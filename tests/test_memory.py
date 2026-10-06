"""Long-term memory: facts learned about the user, and summaries of past sessions."""

import json
import sqlite3

import pytest

from app.db import queries
from app.db.database import get_db
from app.db.models import Episode, Memory, Session

pytestmark = pytest.mark.usefixtures("db")


@pytest.fixture
async def session() -> Session:
    return await queries.create_session()


# ---- memories table ----


async def test_save_and_read_a_memory(session: Session) -> None:
    memory = Memory(category="brand_preference", content="prefers Sony for audio", source_session=session.id)
    await queries.save_memory(memory)
    assert await queries.get_all_memories() == [
        {
            "id": memory.id,
            "category": "brand_preference",
            "content": "prefers Sony for audio",
            "confidence": 1.0,
            "source_session": session.id,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "access_count": 0,
        }
    ]


async def test_unknown_categories_are_rejected() -> None:
    db = await get_db()
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
        await db.execute(
            "INSERT INTO memories (id, category, content, created_at, updated_at) VALUES ('m', 'mood', 'x', 't', 't')"
        )


async def test_memories_come_strongest_first_then_newest() -> None:
    await queries.save_memory(Memory(category="general", content="weak", confidence=0.5, updated_at="2026-10-03"))
    await queries.save_memory(Memory(category="general", content="old", confidence=1.0, updated_at="2026-10-01"))
    await queries.save_memory(Memory(category="general", content="new", confidence=1.0, updated_at="2026-10-02"))
    assert [m["content"] for m in await queries.get_all_memories()] == ["new", "old", "weak"]
    assert [m["content"] for m in await queries.get_all_memories(limit=2)] == ["new", "old"]


async def test_memories_by_category() -> None:
    await queries.save_memory(Memory(category="brand_dislike", content="dislikes boAt"))
    await queries.save_memory(Memory(category="size_info", content="wears size M"))
    rows = await queries.get_memories_by_category("size_info")
    assert [m["content"] for m in rows] == ["wears size M"]


async def test_confidence_update_also_marks_the_memory_as_recent() -> None:
    memory = Memory(category="retailer_preference", content="shops on Amazon", confidence=0.8, updated_at="2026-01-01")
    await queries.save_memory(memory)
    await queries.update_memory_confidence(memory.id, 0.9)
    [row] = await queries.get_all_memories()
    assert row["confidence"] == 0.9 and row["updated_at"] > "2026-01-01"


async def test_replace_keeps_the_id_and_category() -> None:
    memory = Memory(category="brand_dislike", content="dislikes boAt", updated_at="2026-01-01")
    await queries.save_memory(memory)
    await queries.replace_memory(memory.id, "is fine with boAt again")
    [row] = await queries.get_all_memories()
    assert (row["id"], row["category"], row["content"]) == (memory.id, "brand_dislike", "is fine with boAt again")
    assert row["updated_at"] > "2026-01-01" and row["created_at"] == memory.created_at


async def test_access_is_counted_without_touching_updated_at() -> None:
    memory = Memory(category="general", content="x")
    await queries.save_memory(memory)
    await queries.increment_access(memory.id)
    await queries.increment_access(memory.id)
    [row] = await queries.get_all_memories()
    assert row["access_count"] == 2 and row["updated_at"] == memory.updated_at


# ---- episodes table ----


async def test_save_and_read_an_episode(session: Session) -> None:
    episode = Episode(
        session_id=session.id,
        summary="Looked for earbuds under ₹3,000 and carted the boAt Airdopes 141.",
        products_searched=json.dumps(["boAt Airdopes 141", "Noise Buds VS104"]),
        products_carted=json.dumps(["boAt Airdopes 141"]),
        outcome="carted",
    )
    await queries.save_episode(episode)
    assert await queries.get_recent_episodes() == [episode.model_dump()]


async def test_one_episode_per_session_a_new_summary_replaces_the_old(session: Session) -> None:
    await queries.save_episode(Episode(session_id=session.id, summary="first", outcome="browsed"))
    await queries.save_episode(Episode(session_id=session.id, summary="continued and carted", outcome="carted"))
    [row] = await queries.get_recent_episodes()
    assert (row["summary"], row["outcome"]) == ("continued and carted", "carted")


async def test_episodes_need_a_real_session() -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await queries.save_episode(Episode(session_id="no-such-session", summary="x"))


async def test_recent_episodes_newest_first() -> None:
    for day in ("2026-10-01", "2026-10-03", "2026-10-02"):
        s = await queries.create_session()
        await queries.save_episode(Episode(session_id=s.id, summary=day, created_at=day))
    assert [e["summary"] for e in await queries.get_recent_episodes(limit=2)] == ["2026-10-03", "2026-10-02"]
