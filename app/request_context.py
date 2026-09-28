"""Request ids and structured logs.

Every HTTP request gets an id: the caller's X-Request-ID if it's a sane value,
otherwise a new one. The id goes back in the response's X-Request-ID header,
onto every log line written while handling the request, and (see
app.agent.core) into the Langfuse trace's metadata, so a log line, a trace and
a user's bug report can be matched up.

Logs are JSON lines by default (one object per line, easy to search and ship
to a log store); LOG_FORMAT=text gives readable lines for local work.

The middleware also turns unhandled exceptions into the standard 500 error.
Starlette handles those outside all user middleware, where the request id is
already gone, so doing it here is what lets the 500 response and its log line
carry the id.
"""

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LogFormat = Literal["json", "text"]

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Echoed into headers, logs and trace metadata, so only a short, plain value is accepted.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

# Health probes run every few seconds; logging each would drown real traffic.
_QUIET_PATHS = frozenset({"/livez", "/readyz"})

logger = logging.getLogger("shopsense.http")


def current_request_id() -> str | None:
    """The id of the HTTP request being handled, or None outside a request."""
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    """Adds the current request id to every log record ("-" outside a request)."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the request id; never drops a record."""
        record.request_id = request_id_var.get() or "-"
        return True


# Attributes every LogRecord has; anything else on a record came from `extra=`.
# Uvicorn adds color_message, a copy of the message with terminal color codes.
_RECORD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "request_id", "color_message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, message, request_id, any
    `extra=` fields, and the exception traceback if there is one."""

    def format(self, record: logging.LogRecord) -> str:
        """The record as one line of JSON."""
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", "-")
        if request_id != "-":
            entry["request_id"] = request_id
        entry.update({k: v for k, v in vars(record).items() if k not in _RECORD_FIELDS})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(log_format: LogFormat) -> None:
    """Root at WARNING, ShopSense at INFO. At INFO, third-party clients (httpx,
    the search engines behind ddgs) log every outgoing request, including
    users' full search queries (SR-69)."""
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        JsonFormatter()
        if log_format == "json"
        else logging.Formatter("%(levelname)s %(name)s [%(request_id)s]: %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    logging.getLogger("shopsense").setLevel(logging.INFO)
    # Uvicorn installs its own handlers; route its messages through ours instead,
    # and drop its access log: the middleware below logs each request with its id.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False


def internal_error_response() -> JSONResponse:
    """The standard 500 body; details stay in the server log."""
    body = {"error": {"code": "internal_error", "message": "Something went wrong. Please try again."}}
    return JSONResponse(body, status_code=500)


class RequestContextMiddleware:
    """Pure ASGI middleware (not BaseHTTPMiddleware), so the request id is set
    in the same context the endpoint runs in."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one request with its id set, then log it."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        method, path = scope["method"], scope["path"]
        status = 500
        response_started = False
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            logger.exception("Unhandled error on %s %s", method, path)
            if response_started:
                raise  # too late to send an error response; let the server close the connection
            await internal_error_response()(scope, receive, send_with_id)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.log(
                logging.DEBUG if path in _QUIET_PATHS else logging.INFO,
                "%s %s %s %.1fms",
                method,
                path,
                status,
                duration_ms,
                extra={"method": method, "path": path, "status": status, "duration_ms": duration_ms},
            )
            request_id_var.reset(token)
