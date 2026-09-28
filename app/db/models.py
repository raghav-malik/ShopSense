import uuid
from datetime import UTC, datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "tool"]


def new_id() -> str:
    return str(uuid.uuid4())


def now_iso() -> str:
    # datetime.utcnow() is deprecated since Python 3.12.
    return datetime.now(UTC).isoformat()


class Session(BaseModel):
    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    budget: float | None = None
    context_summary: str | None = None


class Message(BaseModel):
    id: str = Field(default_factory=new_id)
    session_id: str
    role: MessageRole
    content: str
    tool_name: str | None = None
    tool_call_id: str | None = None
    created_at: str = Field(default_factory=now_iso)
    token_count: int | None = None


class CartItem(BaseModel):
    id: str = Field(default_factory=new_id)
    session_id: str
    product_name: str
    price: float | None = None
    currency: str = "INR"
    url: str
    source: str | None = None
    added_at: str = Field(default_factory=now_iso)


class Preference(BaseModel):
    id: str = Field(default_factory=new_id)
    key: str
    value: str  # JSON-encoded
    updated_at: str = Field(default_factory=now_iso)


# Rows as returned by queries.py: one key per column, so readers of a row are
# checked against the schema in database.py.


class MessageRow(TypedDict):
    id: str
    session_id: str
    role: MessageRole
    content: str
    tool_name: str | None
    tool_call_id: str | None
    created_at: str
    token_count: int | None


class CartItemRow(TypedDict):
    id: str
    session_id: str
    product_name: str
    price: float | None
    currency: str
    url: str
    source: str | None
    added_at: str
