"""Every database read and write: sessions, messages, cart, preferences, memories and episodes."""

import json
from typing import Any, cast

from app.db.database import get_db
from app.db.models import (
    CartItem,
    CartItemRow,
    Episode,
    EpisodeRow,
    Memory,
    MemoryRow,
    Message,
    MessageRow,
    Session,
    new_id,
    now_iso,
)

# ---- Sessions ----


async def create_session(session_id: str | None = None) -> Session:
    """Create a session. `session_id` is for tests and scripts; the API always generates one."""
    db = await get_db()
    session = Session(id=session_id) if session_id else Session()
    await db.execute(
        "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
        (session.id, session.created_at, session.updated_at),
    )
    await db.commit()
    return session


async def get_session(session_id: str) -> Session | None:
    """The session, or None if it doesn't exist."""
    db = await get_db()
    rows = list(await db.execute_fetchall("SELECT * FROM sessions WHERE id = ?", (session_id,)))
    if not rows:
        return None
    r = rows[0]
    return Session(
        id=r["id"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        budget=r["budget"],
        context_summary=r["context_summary"],
    )


async def update_session_budget(session_id: str, budget: float | None) -> None:
    """Set the session's budget (INR), or clear it with None."""
    db = await get_db()
    await db.execute(
        "UPDATE sessions SET budget = ?, updated_at = ? WHERE id = ?",
        (budget, now_iso(), session_id),
    )
    await db.commit()


# ---- Messages ----


async def save_message(msg: Message) -> None:
    """Store a message and mark the session as updated."""
    db = await get_db()
    await db.execute(
        """INSERT INTO messages (id, session_id, role, content, tool_name, tool_call_id, created_at, token_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            msg.id,
            msg.session_id,
            msg.role,
            msg.content,
            msg.tool_name,
            msg.tool_call_id,
            msg.created_at,
            msg.token_count,
        ),
    )
    await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (msg.created_at, msg.session_id))
    await db.commit()


async def get_messages(session_id: str, limit: int = 50) -> list[MessageRow]:
    """Returns the most recent `limit` messages, oldest first.

    Takes the newest rows then flips them: `ORDER BY ... ASC LIMIT n` would return
    the oldest n, and long conversations would lose their recent context.
    rowid breaks ties between messages saved within the same timestamp.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        """SELECT * FROM (
               SELECT rowid AS _seq, * FROM messages
               WHERE session_id = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT ?
           ) ORDER BY created_at ASC, _seq ASC""",
        (session_id, limit),
    )
    # .keys() is required: iterating a sqlite3.Row yields its values, not its column names.
    return [cast(MessageRow, {k: r[k] for k in r.keys() if k != "_seq"}) for r in rows]  # noqa: SIM118


# ---- Cart ----


async def add_to_cart(
    session_id: str, product_name: str, price: float | None, url: str, source: str | None = None
) -> CartItem:
    """Add a product to the session's cart."""
    db = await get_db()
    item = CartItem(session_id=session_id, product_name=product_name, price=price, url=url, source=source)
    await db.execute(
        """INSERT INTO cart_items (id, session_id, product_name, price, currency, url, source, added_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (item.id, item.session_id, item.product_name, item.price, item.currency, item.url, item.source, item.added_at),
    )
    await db.commit()
    return item


async def remove_from_cart(session_id: str, product_name: str) -> bool:
    """Remove a product by exact name; False if it wasn't in the cart."""
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM cart_items WHERE session_id = ? AND product_name = ?",
        (session_id, product_name),
    )
    await db.commit()
    return cursor.rowcount > 0


async def get_cart(session_id: str) -> list[CartItemRow]:
    """The session's cart, oldest item first."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM cart_items WHERE session_id = ? ORDER BY added_at ASC",
        (session_id,),
    )
    return [cast(CartItemRow, dict(r)) for r in rows]


async def clear_cart(session_id: str) -> None:
    """Remove every item from the session's cart."""
    db = await get_db()
    await db.execute("DELETE FROM cart_items WHERE session_id = ?", (session_id,))
    await db.commit()


# ---- Preferences ----


async def get_all_preferences() -> dict[str, Any]:
    """Preference key -> its JSON-decoded value."""
    db = await get_db()
    rows = await db.execute_fetchall("SELECT key, value FROM preferences")
    return {r["key"]: json.loads(r["value"]) for r in rows}


async def set_preference(key: str, value: object) -> None:
    """Save a preference (any JSON value), replacing an existing one with the same key."""
    db = await get_db()
    encoded = json.dumps(value)
    await db.execute(
        """INSERT INTO preferences (id, key, value, updated_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
        (new_id(), key, encoded, now_iso()),
    )
    await db.commit()


# ---- Memories ----


async def get_all_memories(limit: int = 20) -> list[MemoryRow]:
    """The `limit` strongest memories: highest confidence first, then most recently updated."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM memories ORDER BY confidence DESC, updated_at DESC LIMIT ?",
        (limit,),
    )
    return [cast(MemoryRow, dict(r)) for r in rows]


async def get_memories_by_category(category: str, limit: int = 10) -> list[MemoryRow]:
    """The `limit` strongest memories in one category, in the same order as get_all_memories."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM memories WHERE category = ? ORDER BY confidence DESC, updated_at DESC LIMIT ?",
        (category, limit),
    )
    return [cast(MemoryRow, dict(r)) for r in rows]


async def save_memory(memory: Memory) -> None:
    """Store a new memory."""
    db = await get_db()
    await db.execute(
        """INSERT INTO memories (id, category, content, confidence, source_session, created_at, updated_at, access_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            memory.id,
            memory.category,
            memory.content,
            memory.confidence,
            memory.source_session,
            memory.created_at,
            memory.updated_at,
            memory.access_count,
        ),
    )
    await db.commit()


async def update_memory_confidence(memory_id: str, confidence: float) -> None:
    """Strengthen or weaken a memory. A new mention also makes it the most recently updated."""
    db = await get_db()
    await db.execute(
        "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
        (confidence, now_iso(), memory_id),
    )
    await db.commit()


async def replace_memory(memory_id: str, new_content: str) -> None:
    """Rewrite a memory the user has contradicted, keeping its id, category and history."""
    db = await get_db()
    await db.execute(
        "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
        (new_content, now_iso(), memory_id),
    )
    await db.commit()


async def increment_access(memory_id: str) -> None:
    """Count one retrieval of a memory (for pruning later); doesn't change updated_at."""
    db = await get_db()
    await db.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (memory_id,))
    await db.commit()


# ---- Episodes ----


async def save_episode(episode: Episode) -> None:
    """Store a session's summary, replacing an earlier one for the same session.

    One episode per session (UNIQUE session_id), but a session can be reopened
    and continued after it was summarized, so a newer summary replaces the old.
    """
    db = await get_db()
    await db.execute(
        """INSERT INTO episodes (id, session_id, summary, products_searched, products_carted, outcome, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               summary = excluded.summary,
               products_searched = excluded.products_searched,
               products_carted = excluded.products_carted,
               outcome = excluded.outcome,
               created_at = excluded.created_at""",
        (
            episode.id,
            episode.session_id,
            episode.summary,
            episode.products_searched,
            episode.products_carted,
            episode.outcome,
            episode.created_at,
        ),
    )
    await db.commit()


async def get_recent_episodes(limit: int = 5) -> list[EpisodeRow]:
    """The `limit` most recent session summaries, newest first."""
    db = await get_db()
    rows = await db.execute_fetchall("SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,))
    return [cast(EpisodeRow, dict(r)) for r in rows]
