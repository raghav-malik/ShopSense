"""The FastAPI app: startup and shutdown, routes, error handling, request ids, and health probes."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import close_db, get_db, init_db
from app.request_context import RequestContextMiddleware, configure_logging
from app.routes import chat, sessions
from app.routes.errors import ErrorResponse, register_error_handlers
from app.tracing.langfuse_setup import get_langfuse, shutdown_langfuse

configure_logging(settings.log_format)
logger = logging.getLogger("shopsense")

# A readiness check must answer fast, so a hung database reads as "not ready".
READINESS_DB_TIMEOUT = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown events."""
    # Startup
    app.state.ready = False
    await init_db()
    # Checking auth up front surfaces a wrong key or region (EU vs US) at
    # startup, instead of traces silently going nowhere. Never blocks startup.
    try:
        langfuse_ok = await asyncio.to_thread(get_langfuse().auth_check)
    except Exception as e:  # noqa: BLE001 - tracing problems are reported, never block startup
        langfuse_ok = False
        logger.warning("Langfuse auth check failed (%s): %s", settings.langfuse_base_url, e)
    app.state.langfuse_ok = langfuse_ok
    app.state.langfuse_project_url = await _langfuse_project_url() if langfuse_ok else None
    logger.info(
        "ShopSense started. Database initialized. LLM: %s/%s. Langfuse: %s",
        settings.llm_provider,
        settings.llm_model,
        "connected" if langfuse_ok else "NOT connected (traces will be lost)",
    )

    app.state.ready = True
    yield
    app.state.ready = False  # stop taking traffic before the database closes

    # Shutdown: flush buffered traces (a blocking call) before closing the DB.
    await asyncio.to_thread(shutdown_langfuse)
    await close_db()
    logger.info("ShopSense stopped. Connections closed.")


app = FastAPI(
    title="ShopSense",
    description="Personal Shopping Concierge Agent API",
    version="0.3.0",
    lifespan=lifespan,
)

register_error_handlers(app)
# Request ids, one log line per request, and the standard 500 for unhandled errors.
app.add_middleware(RequestContextMiddleware)

# Register routes
app.include_router(sessions.router)
app.include_router(chat.router)


async def _langfuse_project_url() -> str | None:
    """Deep link to this project's Langfuse dashboard (right region, right project)."""
    try:
        projects = await asyncio.to_thread(get_langfuse().api.projects.get)
        return f"{settings.langfuse_base_url}/project/{projects.data[0].id}"
    except Exception:  # noqa: BLE001 - the link is a convenience; /health falls back to the base URL
        return None


@app.get("/livez")
async def livez() -> dict[str, str]:
    """Liveness: the process is up and answering. Deliberately checks nothing
    else, so a slow database or LLM provider never gets a healthy process restarted."""
    return {"status": "ok"}


@app.get(
    "/readyz",
    response_model=None,
    responses={503: {"model": ErrorResponse, "description": "Starting, stopping, or the database is unavailable"}},
)
async def readyz() -> JSONResponse:
    """Readiness: can this instance serve requests right now? Checks only what
    every request needs: startup finished and the database answers. The LLM
    provider and Langfuse aren't checked: an outage there would mark every
    instance unready at once, and chat already reports provider errors itself."""
    checks = {"startup": bool(getattr(app.state, "ready", False))}
    # After shutdown the database is closed, and get_db() would reopen it.
    checks["database"] = checks["startup"] and await _database_answers()
    langfuse = "connected" if getattr(app.state, "langfuse_ok", False) else "not connected"
    if all(checks.values()):
        return JSONResponse({"status": "ready", "checks": dict.fromkeys(checks, "ok"), "langfuse": langfuse})
    failed = [{"check": name, "status": "failed"} for name, ok in checks.items() if not ok]
    body = {"error": {"code": "not_ready", "message": "This instance can't serve requests yet.", "details": failed}}
    return JSONResponse(body, status_code=503)


async def _database_answers() -> bool:
    try:
        async with asyncio.timeout(READINESS_DB_TIMEOUT):
            db = await get_db()
            await db.execute("SELECT 1")
    except Exception:
        logger.warning("Readiness: database check failed", exc_info=True)
        return False
    return True


@app.get("/health")
async def health() -> dict[str, Any]:
    """Status, the configured models, and the Langfuse project link (read by the UI)."""
    # The frontend reads the model and dashboard link from here rather than
    # hardcoding them (the spec's UI said "Llama 3.3 70B" and linked the EU region).
    return {
        "status": "ok",
        "service": "shopsense",
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "llm_small": f"{settings.llm_provider}/{settings.llm_small_model}",
        "langfuse_url": getattr(app.state, "langfuse_project_url", None) or settings.langfuse_base_url,
    }
