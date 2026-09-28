import json
from typing import Any, cast

from app.db.database import get_db
from app.db.models import CartItem, CartItemRow, Message, MessageRow, Session, new_id, now_iso

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


async def update_session_budget(session_id: str, budget: float) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE sessions SET budget = ?, updated_at = ? WHERE id = ?",
        (budget, now_iso(), session_id),
    )
    await db.commit()


# ---- Messages ----


async def save_message(msg: Message) -> None:
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
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM cart_items WHERE session_id = ? AND product_name = ?",
        (session_id, product_name),
    )
    await db.commit()
    return cursor.rowcount > 0


async def get_cart(session_id: str) -> list[CartItemRow]:
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM cart_items WHERE session_id = ? ORDER BY added_at ASC",
        (session_id,),
    )
    return [cast(CartItemRow, dict(r)) for r in rows]


async def clear_cart(session_id: str) -> None:
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
    db = await get_db()
    encoded = json.dumps(value)
    await db.execute(
        """INSERT INTO preferences (id, key, value, updated_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
        (new_id(), key, encoded, now_iso()),
    )
    await db.commit()


if __name__ == "__main__":
    # Smoke test against a throwaway DB file (your real shopsense.db is untouched).
    # From the project root:  python -m app.db.queries
    import asyncio
    import tempfile
    from pathlib import Path

    from app.config import settings
    from app.db.database import close_db, init_db

    async def _smoke_test() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings.db_path = str(Path(tmp) / "smoke.db")
            await init_db()
            try:
                session = await create_session()
                print(f"created session: {session.id}")
                await save_message(
                    Message(session_id=session.id, role="user", content="Find me wireless earbuds under 3000")
                )
                messages = await get_messages(session.id)
                print(f"retrieved {len(messages)} message(s): {messages[0]['role']}: {messages[0]['content']!r}")
                if messages[0]["content"] != "Find me wireless earbuds under 3000":
                    raise SystemExit("FAILED: retrieved message doesn't match what was saved")
                print("OK")
            finally:
                await close_db()

    asyncio.run(_smoke_test())
