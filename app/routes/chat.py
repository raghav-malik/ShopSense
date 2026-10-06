"""Per-session endpoints: chat, follow-up suggestions, cart and history."""

import asyncio
import logging
import math

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.agent.core import run_agent
from app.agent.schemas import AgentResponse
from app.agent.suggestions import suggest_follow_ups
from app.db import queries
from app.db.models import MessageRow, Session
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.routes.errors import ErrorResponse, api_error
from app.tracing.langfuse_setup import get_langfuse

logger = logging.getLogger("shopsense.api")

# (status, user-facing message) per LLM failure. The technical reason goes to
# the server log and the Langfuse trace, not to the user.
LLM_ERRORS: list[tuple[type[LLMError], int, str]] = [
    (LLMRateLimitError, 503, "The assistant is busy right now. Please try again in a minute."),
    (LLMTimeoutError, 504, "The assistant took too long to respond. Please try again."),
    (LLMUnavailableError, 502, "The assistant's language model is unavailable right now. Please try again later."),
    (LLMError, 502, "The assistant couldn't process that request. Try rephrasing it."),
]

router = APIRouter(
    prefix="/sessions/{session_id}",
    tags=["chat"],
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
)


class ChatRequest(BaseModel):
    """A user message for the agent."""

    message: str = Field(max_length=4000)


async def _require_session(session_id: str) -> Session:
    session = await queries.get_session(session_id)
    if not session:
        raise api_error(404, "session_not_found", f"Session {session_id} not found")
    return session


@router.post(
    "/chat",
    responses={
        400: {"model": ErrorResponse, "description": "Empty message"},
        500: {"model": ErrorResponse, "description": "Unexpected server error (with trace_url)"},
        502: {"model": ErrorResponse, "description": "The LLM is unavailable or rejected the request"},
        503: {"model": ErrorResponse, "description": "The LLM provider is rate limiting us (Retry-After header)"},
        504: {"model": ErrorResponse, "description": "The LLM didn't respond in time"},
    },
)
async def chat(session_id: str, request: ChatRequest, background_tasks: BackgroundTasks) -> AgentResponse:
    """Send a message and get the agent's response. Memory extraction runs after
    the response is sent, so the answer never waits for it."""
    await _require_session(session_id)

    if not request.message.strip():
        raise api_error(400, "empty_message", "Message cannot be empty")

    # Choose the trace id up front: if the agent fails, its trace (level ERROR,
    # with the failing call) still exists, and the error response links to it.
    langfuse = get_langfuse()
    trace_id = langfuse.create_trace_id()

    try:
        return await run_agent(
            session_id, request.message, schedule=background_tasks.add_task, langfuse_trace_id=trace_id
        )
    except LLMError as e:
        status, message = next((s, m) for cls, s, m in LLM_ERRORS if isinstance(e, cls))
        logger.warning("Agent LLM failure (%s) for session %s: %s", e.code, session_id, e)
        headers = {"Retry-After": str(math.ceil(e.retry_after))} if e.retry_after else None
        raise api_error(status, e.code, message, trace_url=await _trace_url(trace_id), headers=headers) from e
    except Exception as e:  # any other failure becomes a structured 500; details go to the log
        logger.exception("Agent failed for session %s", session_id)
        raise api_error(
            500,
            "agent_error",
            "Something went wrong while answering. Please try again.",
            trace_url=await _trace_url(trace_id),
        ) from e


async def _trace_url(trace_id: str) -> str | None:
    # get_trace_url looks up the project id over HTTP the first time; keep it off the event loop.
    try:
        return await asyncio.to_thread(get_langfuse().get_trace_url, trace_id=trace_id)
    except Exception:  # noqa: BLE001 - the debug link is optional; never mask the real error
        return None


class SuggestionsResponse(BaseModel):
    """Follow-up messages the user might send next."""

    suggestions: list[str]


@router.post("/suggestions")
async def suggestions(session_id: str) -> SuggestionsResponse:
    """2-3 follow-up messages the user might send next. Best-effort: an empty
    list if they can't be made. The UI calls this after showing an answer, so
    the answer itself never waits for suggestions."""
    await _require_session(session_id)
    history = await queries.get_messages(session_id, limit=12)
    return SuggestionsResponse(suggestions=await suggest_follow_ups(session_id, history))


class CartItemOut(BaseModel):
    """A cart item as the API returns it."""

    product_name: str
    price: float | None
    url: str


class CartResponse(BaseModel):
    """The session's cart, its total, and the session's budget if one is set."""

    items: list[CartItemOut]
    total: float
    currency: str = "INR"
    budget: float | None = None


@router.get("/cart")
async def get_cart(session_id: str) -> CartResponse:
    """Get the current cart for a session."""
    session = await _require_session(session_id)

    cart = await queries.get_cart(session_id)
    items = [CartItemOut(product_name=i["product_name"], price=i["price"], url=i["url"]) for i in cart]
    total = sum(i["price"] or 0 for i in cart)
    return CartResponse(items=items, total=total, budget=session.budget)


class SessionInfo(BaseModel):
    """Session details returned with its history."""

    id: str
    created_at: str
    budget: float | None


class HistoryResponse(BaseModel):
    """A session's messages (tool results included) and its details."""

    messages: list[MessageRow]
    session: SessionInfo


@router.get("/history")
async def get_history(session_id: str) -> HistoryResponse:
    """Get conversation history for a session."""
    session = await _require_session(session_id)

    messages = await queries.get_messages(session_id)
    return HistoryResponse(
        messages=messages,
        session=SessionInfo(id=session.id, created_at=session.created_at, budget=session.budget),
    )
