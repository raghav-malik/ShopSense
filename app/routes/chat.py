import logging

import openai
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.core import run_agent
from app.agent.schemas import AgentResponse
from app.db import queries
from app.db.models import Session
from app.routes.errors import ErrorResponse, api_error

logger = logging.getLogger("shopsense.api")

router = APIRouter(
    prefix="/sessions/{session_id}",
    tags=["chat"],
    responses={404: {"model": ErrorResponse, "description": "Session not found"}},
)


class ChatRequest(BaseModel):
    message: str = Field(max_length=4000)


async def _require_session(session_id: str) -> Session:
    session = await queries.get_session(session_id)
    if not session:
        raise api_error(404, "session_not_found", f"Session {session_id} not found")
    return session


@router.post(
    "/chat",
    response_model=AgentResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty message"},
        502: {"model": ErrorResponse, "description": "The LLM provider returned an error"},
        503: {"model": ErrorResponse, "description": "The LLM provider is rate limiting us"},
    },
)
async def chat(session_id: str, request: ChatRequest):
    """Send a message and get the agent's response."""
    await _require_session(session_id)

    if not request.message.strip():
        raise api_error(400, "empty_message", "Message cannot be empty")

    # The failure itself is already on the Langfuse trace (level ERROR); log it
    # here for the server, and give the client a stable code, not internals.
    try:
        return await run_agent(session_id, request.message)
    except openai.RateLimitError:
        logger.warning("LLM rate limited for session %s", session_id)
        raise api_error(503, "llm_rate_limited", "The assistant is busy right now. Please try again in a minute.")
    except openai.APIError:
        logger.exception("LLM provider error for session %s", session_id)
        raise api_error(502, "llm_error", "The assistant couldn't reach its language model. Please try again.")


class CartItemOut(BaseModel):
    product_name: str
    price: float | None
    url: str


class CartResponse(BaseModel):
    items: list[CartItemOut]
    total: float
    currency: str = "INR"


@router.get("/cart", response_model=CartResponse)
async def get_cart(session_id: str):
    """Get the current cart for a session."""
    await _require_session(session_id)

    cart = await queries.get_cart(session_id)
    items = [CartItemOut(product_name=i["product_name"], price=i["price"], url=i["url"]) for i in cart]
    total = sum(i.get("price", 0) or 0 for i in cart)
    return CartResponse(items=items, total=total)


class HistoryResponse(BaseModel):
    messages: list[dict]
    session: dict


@router.get("/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session."""
    session = await _require_session(session_id)

    messages = await queries.get_messages(session_id)
    return HistoryResponse(
        messages=messages,
        session={"id": session.id, "created_at": session.created_at, "budget": session.budget},
    )
