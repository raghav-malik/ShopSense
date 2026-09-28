import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


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
    role: str  # 'user' | 'assistant' | 'tool'
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
