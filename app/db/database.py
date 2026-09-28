import asyncio
from pathlib import Path

import aiosqlite

from app.config import PROJECT_ROOT, settings

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


async def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    db = await get_db()

    await db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            budget          REAL,
            context_summary TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL REFERENCES sessions(id),
            role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
            content         TEXT NOT NULL,
            tool_name       TEXT,
            tool_call_id    TEXT,
            created_at      TEXT NOT NULL,
            token_count     INTEGER
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
    """)
    await db.commit()


async def close_db():
    """Close the database connection. Called at app shutdown."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
