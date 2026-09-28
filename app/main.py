import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.config import settings
from app.db.database import close_db, init_db
from app.routes import chat, sessions
from app.routes.errors import register_error_handlers
from app.tracing.langfuse_setup import get_langfuse, shutdown_langfuse

# Root at WARNING: at INFO, third-party clients (httpx, the search engines behind
# ddgs) log every outgoing request, including users' full search queries.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shopsense")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown events."""
    # Startup
    await init_db()
    # Checking auth up front surfaces a wrong key or region (EU vs US) at
    # startup, instead of traces silently going nowhere. Never blocks startup.
    try:
        langfuse_ok = await asyncio.to_thread(get_langfuse().auth_check)
    except Exception as e:  # noqa: BLE001 - tracing problems are reported, never block startup
        langfuse_ok = False
        logger.warning("Langfuse auth check failed (%s): %s", settings.langfuse_base_url, e)
    app.state.langfuse_project_url = await _langfuse_project_url() if langfuse_ok else None
    logger.info(
        "ShopSense started. Database initialized. LLM: %s/%s. Langfuse: %s",
        settings.llm_provider,
        settings.llm_model,
        "connected" if langfuse_ok else "NOT connected (traces will be lost)",
    )

    yield

    # Shutdown: flush buffered traces (a blocking call) before closing the DB.
    await asyncio.to_thread(shutdown_langfuse)
    await close_db()
    logger.info("ShopSense stopped. Connections closed.")


app = FastAPI(
    title="ShopSense",
    description="Personal Shopping Concierge Agent API",
    version="0.1.0",
    lifespan=lifespan,
)

register_error_handlers(app)

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


@app.get("/health")
async def health() -> dict[str, Any]:
    # The frontend reads the model and dashboard link from here rather than
    # hardcoding them (the spec's UI said "Llama 3.3 70B" and linked the EU region).
    return {
        "status": "ok",
        "service": "shopsense",
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "llm_small": f"{settings.llm_provider}/{settings.llm_small_model}",
        "langfuse_url": getattr(app.state, "langfuse_project_url", None) or settings.langfuse_base_url,
    }
