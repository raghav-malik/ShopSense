"""One error shape for every non-2xx response:

    {"error": {"code": "session_not_found", "message": "Session abc not found"}}

`code` is stable and machine-readable (the frontend can branch on it);
`message` is for humans. Validation errors add `details`; errors from an agent
run add `trace_url`, the Langfuse trace of the failed turn.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("shopsense.api")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None
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
    detail = {"code": code, "message": message}
    if trace_url:
        detail["trace_url"] = trace_url
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _body(code: str, message: str, details: list[dict] | None = None, trace_url: str | None = None) -> dict:
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    if trace_url:
        error["trace_url"] = trace_url
    return {"error": error}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and {"code", "message"} <= exc.detail.keys():
            body = _body(exc.detail["code"], exc.detail["message"], trace_url=exc.detail.get("trace_url"))
        else:
            # Framework-raised errors (404 unknown route, 405 wrong method, ...)
            body = _body(f"http_{exc.status_code}", str(exc.detail))
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(_body("validation_error", "Request failed validation.", details), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        # Full detail goes to the server log; the client gets no internals.
        logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(_body("internal_error", "Something went wrong. Please try again."), status_code=500)
