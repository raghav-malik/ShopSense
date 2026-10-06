"""Database models (Pydantic, for writes) and row types (TypedDict, for reads)."""

import re
import uuid
from datetime import UTC, datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "tool"]
# Same values as the CHECK constraints on memories.category and episodes.outcome.
MemoryCategory = Literal[
    "brand_preference",
    "brand_dislike",
    "budget_range",
    "category_interest",
    "retailer_preference",
    "product_feedback",
    "size_info",
    "shopping_style",
    "general",
]
# ShopSense can't place orders, so "purchased" is only known if the user says so.
EpisodeOutcome = Literal["purchased", "carted", "browsed", "abandoned"]


def new_id() -> str:
    """A new random id (UUID4)."""
    return str(uuid.uuid4())


def now_iso() -> str:
    """The current UTC time in ISO 8601, which sorts correctly as text."""
    # datetime.utcnow() is deprecated since Python 3.12.
    return datetime.now(UTC).isoformat()


TitleSource = Literal["placeholder", "llm", "user"]

# user.md: the user's own notes about themselves. Written by the user, read by
# the agent, never written by it. Blank fields mean "not told yet".
USER_MD_TEMPLATE = """# About me

Name:
Call me:
Pronouns:
City:
Notes:
"""


_USER_MD_FIELD = re.compile(r"^(name|call me|pronouns|city|notes)\s*:\s*$", re.IGNORECASE)


def user_md_is_blank(content: str) -> bool:
    """Whether user.md says nothing yet: only the template's heading and empty fields."""
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not _USER_MD_FIELD.match(line):
            return False
    return True


class Session(BaseModel):
    """A conversation. `budget` (INR) is set by the set_budget tool; `context_summary` is unused (SR-14).

    `title` starts as the first message (`placeholder`), is replaced by a short
    generated title (`llm`) unless the user renamed it first (`user`). A deleted
    session has `deleted_at` set and is treated as gone; its rows are kept.
    """

    id: str = Field(default_factory=new_id)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    budget: float | None = None
    context_summary: str | None = None
    title: str | None = None
    title_source: TitleSource | None = None
    deleted_at: str | None = None


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


class Memory(BaseModel):
    """A fact about the user, learned from what they said rather than saved on request.

    `confidence` is 1.0 when stated outright and lower when inferred; a repeated
    mention strengthens it. `source_session` is where it was learned.
    """

    id: str = Field(default_factory=new_id)
    category: MemoryCategory
    content: str
    confidence: float = 1.0
    source_session: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    access_count: int = 0


class Episode(BaseModel):
    """A summary of one past shopping session; `products_*` are JSON-encoded lists of names."""

    id: str = Field(default_factory=new_id)
    session_id: str
    summary: str
    products_searched: str | None = None  # JSON array
    products_carted: str | None = None  # JSON array
    outcome: EpisodeOutcome | None = None
    created_at: str = Field(default_factory=now_iso)


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


class MemoryRow(TypedDict):
    """A row of the memories table, as queries return it."""

    id: str
    category: MemoryCategory
    content: str
    confidence: float
    source_session: str | None
    created_at: str
    updated_at: str
    access_count: int


class EpisodeRow(TypedDict):
    """A row of the episodes table, as queries return it."""

    id: str
    session_id: str
    summary: str
    products_searched: str | None
    products_carted: str | None
    outcome: EpisodeOutcome | None
    created_at: str


class ChatEpisodeRow(EpisodeRow):
    """An episode with its chat's title and last activity: a summary belongs to
    when the chat happened, not to when it was summarized (often later)."""

    chat_title: str | None
    last_active_at: str


class ChatListRow(TypedDict):
    """A chat in the sidebar list."""

    id: str
    title: str | None
    created_at: str
    updated_at: str


class UserProfileRow(TypedDict):
    """The user's own user.md."""

    content: str
    created_at: str
    updated_at: str
