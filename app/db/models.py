"""Database models (Pydantic, for writes) and row types (TypedDict, for reads)."""

import uuid
from datetime import UTC, datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "tool"]


def new_id() -> str:
    """A new random id (UUID4)."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """The current UTC time in ISO 8601, which sorts correctly as text."""
    # datetime.utcnow() is deprecated since Python 3.12.
    return datetime.now(UTC).isoformat()


class Session(BaseModel):
    """A conversation. Nothing sets `budget` (INR) or `context_summary` yet (SR-13, SR-14)."""

    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    budget: float | None = None
    context_summary: str | None = None


class Message(BaseModel):
    """One user, assistant or tool message in a session."""

    id: str = Field(default_factory=new_id)
    session_id: str
    role: MessageRole
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    created_at: str = Field(default_factory=now_iso)
    token_count: int | None = None


class CartItem(BaseModel):
    """A product in a session's cart."""

    id: str = Field(default_factory=new_id)
    session_id: str
    product_name: str
    price: float | None = None
    currency: str = "INR"
    url: str
    source: str | None = None
    added_at: str = Field(default_factory=now_iso)


class Preference(BaseModel):
    """A lasting user preference; `value` is JSON-encoded."""

    id: str = Field(default_factory=new_id)
    key: str
    value: str  # JSON-encoded
    updated_at: str = Field(default_factory=now_iso)


# Rows as returned by queries.py: one key per column, so readers of a row are
# checked against the schema in database.py.


class MessageRow(TypedDict):
    """A row of the messages table, as queries return it."""

    id: str
    session_id: str
    role: MessageRole
    content: str
    tool_name: str | None
    tool_call_id: str | None
    created_at: str
    token_count: int | None


class CartItemRow(TypedDict):
    """A row of the cart_items table, as queries return it."""

    id: str
    session_id: str
    product_name: str
    price: float | None
    currency: str
    url: str
    source: str | None
    added_at: str
