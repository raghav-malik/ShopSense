"""Every database read and write: sessions, messages, cart, preferences, memories, episodes and user.md."""

import json
from typing import Any, cast

from app.db.database import get_db
from app.db.models import (
    USER_MD_TEMPLATE,
    CartItem,
    CartItemRow,
    ChatEpisodeRow,
    ChatListRow,
    Episode,
    Memory,
    MemoryRow,
    Message,
    MessageRow,
    Session,
    TitleSource,
    UserProfileRow,
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
    """The session, or None if it doesn't exist or was deleted."""
    db = await get_db()
    rows = list(await db.execute_fetchall("SELECT * FROM sessions WHERE id = ? AND deleted_at IS NULL", (session_id,)))
    if not rows:
        return None
    r = rows[0]
    return Session(
        id=r["id"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
        budget=r["budget"],
        context_summary=r["context_summary"],
        title=r["title"],
        title_source=r["title_source"],
    )


async def set_session_title(
    session_id: str, title: str, source: TitleSource, *, only_if: TitleSource | None = None
) -> bool:
    """Set the chat's title. With `only_if`, only while the title still has that
    source, so a generated title never overwrites one the user typed meanwhile.

    Doesn't touch updated_at: a title isn't activity, and updated_at orders the
    chat list and decides when a chat needs a new summary.
    """
    db = await get_db()
    query = "UPDATE sessions SET title = ?, title_source = ? WHERE id = ? AND deleted_at IS NULL"
    params: tuple[object, ...] = (title, source, session_id)
    if only_if is not None:
        query += " AND title_source = ?"
        params += (only_if,)
    cursor = await db.execute(query, params)
    await db.commit()
    return cursor.rowcount > 0


async def list_chats(limit: int = 50, offset: int = 0, query: str = "") -> tuple[list[ChatListRow], int]:
    """Chats with at least one message, most recently active first, optionally
    filtered by title (case-insensitive), and how many match in total. Deleted
    chats and the empty sessions every new tab starts with aren't chats."""
    db = await get_db()
    where = """deleted_at IS NULL
               AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = sessions.id AND m.role = 'user')"""
    params: tuple[object, ...] = ()
    if query.strip():
        where += " AND lower(coalesce(title, '')) LIKE ? ESCAPE '\\'"
        escaped = query.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params = (f"%{escaped}%",)
    rows = await db.execute_fetchall(
        f"SELECT id, title, created_at, updated_at FROM sessions WHERE {where} "  # noqa: S608 - fixed SQL, values bound
        "ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    counted = await db.execute_fetchall(f"SELECT COUNT(*) FROM sessions WHERE {where}", params)  # noqa: S608
    total = int(next(iter(counted))[0])
    return [cast(ChatListRow, dict(r)) for r in rows], total


async def delete_session(session_id: str) -> bool:
    """Delete a chat from the user's list (a soft delete: its rows stay, so
    memories learned from it keep their source). False if there was none."""
    db = await get_db()
    cursor = await db.execute(
        "UPDATE sessions SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL", (now_iso(), session_id)
    )
    await db.commit()
    return cursor.rowcount > 0


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
        """INSERT INTO messages (id, session_id, role, content, tool_name, tool_call_id, created_at, token_count, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            msg.id,
            msg.session_id,
            msg.role,
            msg.content,
            msg.tool_name,
            msg.tool_call_id,
            msg.created_at,
            msg.token_count,
            msg.details,
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


async def add_to_latest_answer(session_id: str, extra: dict[str, object]) -> bool:
    """Merge `extra` into the details of the session's latest answer (e.g. its
    follow-up suggestions, made after it was saved). False if there's no answer.
    Doesn't count as activity."""
    db = await get_db()
    rows = list(
        await db.execute_fetchall(
            """SELECT id, details FROM messages WHERE session_id = ? AND role = 'assistant'
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (session_id,),
        )
    )
    if not rows:
        return False
    details = json.loads(rows[0]["details"] or "{}")
    details.update(extra)
    await db.execute("UPDATE messages SET details = ? WHERE id = ?", (json.dumps(details), rows[0]["id"]))
    await db.commit()
    return True


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


async def preferences_updated_at() -> str | None:
    """When a preference was last saved, or None if there are none."""
    db = await get_db()
    rows = list(await db.execute_fetchall("SELECT MAX(updated_at) FROM preferences"))
    return rows[0][0] if rows else None


async def delete_preference(key: str) -> bool:
    """Forget a preference; False if there was none with that key."""
    db = await get_db()
    cursor = await db.execute("DELETE FROM preferences WHERE key = ?", (key,))
    await db.commit()
    return cursor.rowcount > 0


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


async def delete_memory(memory_id: str) -> bool:
    """Forget a memory; False if there was no such memory."""
    db = await get_db()
    cursor = await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    await db.commit()
    return cursor.rowcount > 0


async def update_memory(memory_id: str, category: str, content: str, confidence: float) -> None:
    """Rewrite a memory the user edited; refreshes updated_at."""
    db = await get_db()
    await db.execute(
        "UPDATE memories SET category = ?, content = ?, confidence = ?, updated_at = ? WHERE id = ?",
        (category, content, confidence, now_iso(), memory_id),
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


async def get_sessions_to_summarize(exclude_session_id: str, limit: int = 3) -> list[str]:
    """Ids of the most recently active sessions that need a summary, newest first:
    sessions with a conversation (2+ user or assistant messages) and either no
    episode yet, or one older than their last message (reopened and continued).
    `exclude_session_id` is the session in use, which isn't finished."""
    db = await get_db()
    rows = await db.execute_fetchall(
        """SELECT s.id FROM sessions s
           LEFT JOIN episodes e ON e.session_id = s.id
           WHERE s.id != ?
             AND s.deleted_at IS NULL
             AND (e.id IS NULL OR e.created_at < s.updated_at)
             AND s.id NOT IN (SELECT session_id FROM forgotten_sessions)
             AND (SELECT COUNT(*) FROM messages m
                  WHERE m.session_id = s.id AND m.role IN ('user', 'assistant')) >= 2
           ORDER BY s.updated_at DESC
           LIMIT ?""",
        (exclude_session_id, limit),
    )
    return [r["id"] for r in rows]


async def delete_episode(episode_id: str) -> bool:
    """Forget a session's summary for good: the session is marked forgotten so
    it isn't summarized again. False if there was no such episode."""
    db = await get_db()
    rows = list(await db.execute_fetchall("SELECT session_id FROM episodes WHERE id = ?", (episode_id,)))
    if not rows:
        return False
    await db.execute(
        "INSERT OR IGNORE INTO forgotten_sessions (session_id, forgotten_at) VALUES (?, ?)",
        (rows[0]["session_id"], now_iso()),
    )
    await db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
    await db.commit()
    return True


async def forget_everything() -> None:
    """Delete every preference, memory and session summary, and mark every
    session forgotten so none is summarized again. Chats themselves stay."""
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO forgotten_sessions (session_id, forgotten_at) SELECT id, ? FROM sessions",
        (now_iso(),),
    )
    await db.execute("DELETE FROM episodes")
    await db.execute("DELETE FROM memories")
    await db.execute("DELETE FROM preferences")
    await db.commit()


async def get_recent_episodes(limit: int = 5) -> list[ChatEpisodeRow]:
    """The summaries of the `limit` most recently active chats, newest first.

    Ordered and dated by the chat's last activity, not by when the summary was
    made: a chat from last week summarized today still belongs to last week.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        """SELECT e.*, s.title AS chat_title, s.updated_at AS last_active_at
           FROM episodes e JOIN sessions s ON s.id = e.session_id
           ORDER BY s.updated_at DESC, e.rowid DESC LIMIT ?""",
        (limit,),
    )
    return [cast(ChatEpisodeRow, dict(r)) for r in rows]


async def update_episode_summary(episode_id: str, summary: str) -> bool:
    """Rewrite a chat's summary (the user edited it); False if there's no such episode."""
    db = await get_db()
    cursor = await db.execute("UPDATE episodes SET summary = ? WHERE id = ?", (summary, episode_id))
    await db.commit()
    return cursor.rowcount > 0


# ---- user.md ----


async def get_user_profile() -> UserProfileRow:
    """The user's user.md (created from the template at startup)."""
    db = await get_db()
    rows = list(await db.execute_fetchall("SELECT content, created_at, updated_at FROM user_profile WHERE id = 1"))
    if not rows:  # only if init_db hasn't run, as in some scripts
        now = now_iso()
        return {"content": USER_MD_TEMPLATE, "created_at": now, "updated_at": now}
    return cast(UserProfileRow, dict(rows[0]))


async def set_user_profile(content: str) -> None:
    """Save the user's user.md."""
    db = await get_db()
    now = now_iso()
    await db.execute(
        """INSERT INTO user_profile (id, content, created_at, updated_at) VALUES (1, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at""",
        (content, now, now),
    )
    await db.commit()
