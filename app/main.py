import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Imported first: creates the Langfuse client before any traced code runs.
from app.tracing.langfuse_setup import get_langfuse, shutdown_langfuse

from app.config import settings
from app.db.database import close_db, init_db
from app.routes import chat, sessions
from app.routes.errors import register_error_handlers

# Root at WARNING: at INFO, third-party clients (httpx, the search engines behind
# ddgs) log every outgoing request, including users' full search queries.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shopsense")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    await init_db()
    # Checking auth up front surfaces a wrong key or region (EU vs US) at
    # startup, instead of traces silently going nowhere. Never blocks startup.
    try:
        langfuse_ok = await asyncio.to_thread(get_langfuse().auth_check)
    except Exception as e:
        langfuse_ok = False
        logger.warning("Langfuse auth check failed (%s): %s", settings.langfuse_base_url, e)
    logger.info(
        "ShopSense started. Database initialized. LLM: %s/%s. Langfuse: %s",
        settings.llm_provider, settings.llm_model, "connected" if langfuse_ok else "NOT connected (traces will be lost)",
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "shopsense"}
