"""The SQLite connection (aiosqlite, WAL mode) and the schema."""

import asyncio
from pathlib import Path

import aiosqlite

from app.config import PROJECT_ROOT, settings
from app.db.models import USER_MD_TEMPLATE, now_iso

# Module-level connection reference
_db: aiosqlite.Connection | None = None
_db_lock = asyncio.Lock()


def _resolve_db_path() -> Path:
    """A relative DB_PATH is relative to the project root, not the CWD — otherwise
    running uvicorn or pytest from another folder silently creates a second, empty DB."""
    path = Path(settings.db_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


async def get_db() -> aiosqlite.Connection:
    """Get the database connection, creating it if needed."""
    global _db
    if _db is None:
        # Two requests arriving at startup would otherwise each open a connection.
        async with _db_lock:
            if _db is None:
                db = await aiosqlite.connect(_resolve_db_path())
                db.row_factory = aiosqlite.Row  # dict-like access on rows
                await db.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
                await db.execute("PRAGMA foreign_keys=ON")
                _db = db
    return _db


async def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    db = await get_db()

    await db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            budget          REAL,
            context_summary TEXT,
            title           TEXT,
            title_source    TEXT CHECK(title_source IN ('placeholder', 'llm', 'user')),
            deleted_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(id),
            role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
            content         TEXT NOT NULL,
            tool_name       TEXT,
            tool_call_id    TEXT,
            created_at      TEXT NOT NULL,
            token_count     INTEGER,
            details         TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, created_at);

        CREATE TABLE IF NOT EXISTS cart_items (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(id),
            product_name    TEXT NOT NULL,
            price           REAL,
            currency        TEXT DEFAULT 'INR',
            url             TEXT NOT NULL,
            source          TEXT,
            added_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cart_session
            ON cart_items(session_id);

        CREATE TABLE IF NOT EXISTS preferences (
            id              TEXT PRIMARY KEY,
            key             TEXT NOT NULL UNIQUE,
            value           TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        -- Memory (MEMORY_DESIGN.md). preferences above holds what the user
        -- explicitly asked to save; memories holds facts learned from what they
        -- said, and episodes one summary per past session.
        CREATE TABLE IF NOT EXISTS memories (
            id              TEXT PRIMARY KEY,
            category        TEXT NOT NULL CHECK(category IN (
                'brand_preference', 'brand_dislike', 'budget_range',
                'category_interest', 'retailer_preference', 'product_feedback',
                'size_info', 'shopping_style', 'general'
            )),
            content         TEXT NOT NULL,
            confidence      REAL DEFAULT 1.0,
            source_session  TEXT REFERENCES sessions(id),
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            access_count    INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category);

        CREATE TABLE IF NOT EXISTS episodes (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL UNIQUE REFERENCES sessions(id),
            summary         TEXT NOT NULL,
            products_searched TEXT,
            products_carted  TEXT,
            outcome         TEXT CHECK(outcome IN (
                'purchased', 'carted', 'browsed', 'abandoned'
            )),
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at DESC);

        -- Sessions whose summary the user deleted: never summarized again, or
        -- the background summarizer would bring the forgotten summary back.
        CREATE TABLE IF NOT EXISTS forgotten_sessions (
            session_id      TEXT PRIMARY KEY REFERENCES sessions(id),
            forgotten_at    TEXT NOT NULL
        );

        -- The user's own user.md (one row): what to call them and other notes
        -- about themselves. Written only by the user.
        CREATE TABLE IF NOT EXISTS user_profile (
            id              INTEGER PRIMARY KEY CHECK(id = 1),
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );
    """)
    await _migrate(db)
    now = now_iso()
    await db.execute(
        "INSERT OR IGNORE INTO user_profile (id, content, created_at, updated_at) VALUES (1, ?, ?, ?)",
        (USER_MD_TEMPLATE, now, now),
    )
    await db.commit()


# Columns added after a table first shipped: CREATE TABLE IF NOT EXISTS doesn't
# add them to an existing database, so they're added here, once.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "messages": [("details", "TEXT")],
    "sessions": [
        ("title", "TEXT"),
        ("title_source", "TEXT CHECK(title_source IN ('placeholder', 'llm', 'user'))"),
        ("deleted_at", "TEXT"),
    ],
}


async def _migrate(db: aiosqlite.Connection) -> None:
    """Add missing columns, then give chats from before titles existed their
    first message as a title."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in await db.execute_fetchall(f"PRAGMA table_info({table})")}
        for name, definition in columns:
            if name not in existing:
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
    await db.execute(
        """UPDATE sessions SET title_source = 'placeholder', title = (
               SELECT substr(content, 1, 60) FROM messages m
               WHERE m.session_id = sessions.id AND m.role = 'user'
               ORDER BY m.created_at, m.rowid LIMIT 1)
           WHERE title IS NULL
             AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = sessions.id AND m.role = 'user')"""
    )


async def close_db() -> None:
    """Close the database connection. Called at app shutdown."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
