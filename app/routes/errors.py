"""One error shape for every non-2xx response:

    {"error": {"code": "session_not_found", "message": "Session abc not found"}}

`code` is stable and machine-readable (the frontend can branch on it);
`message` is for humans. Validation errors add `details`; errors from an agent
run add `trace_url`, the Langfuse trace of the failed turn.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.llm.types import JSONObject

logger = logging.getLogger("shopsense.api")


class ErrorBody(BaseModel):
    """The body of every error response."""

    code: str
    message: str
    details: list[JSONObject] | None = None
    trace_url: str | None = None


class ErrorResponse(BaseModel):
    """Documents the error shape in the OpenAPI schema."""

    error: ErrorBody


def api_error(
    status_code: int,
    code: str,
    message: str,
    *,
    trace_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Raise with: `raise api_error(404, "session_not_found", "...")`."""
    detail: dict[str, str] = {"code": code, "message": message}
    if trace_url:
        detail["trace_url"] = trace_url
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _body(code: str, message: str, details: list[JSONObject] | None = None, trace_url: str | None = None) -> JSONObject:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    if trace_url:
        error["trace_url"] = trace_url
    return {"error": error}


def register_error_handlers(app: FastAPI) -> None:
    """Make HTTP and validation errors use the standard error shape."""

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Starlette types `detail` as str; api_error() puts a dict there (FastAPI allows any JSON).
        detail: object = exc.detail
        if isinstance(detail, dict) and {"code", "message"} <= detail.keys():
            body = _body(detail["code"], detail["message"], trace_url=detail.get("trace_url"))
        else:
            # Framework-raised errors (404 unknown route, 405 wrong method, ...)
            body = _body(f"http_{exc.status_code}", str(exc.detail))
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(_body("validation_error", "Request failed validation.", details), status_code=422)

    # Unhandled exceptions (500 internal_error) are handled in
    # app.request_context.RequestContextMiddleware, where the request id is known.
