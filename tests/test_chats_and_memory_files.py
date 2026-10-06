"""The chat list (titles, rename, delete), user.md, memory as markdown files, and dates in the user's time zone."""

import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.agent.core as core
from app import clock
from app.agent import memory_files
from app.agent.memory import extract_memories, summarize_pending_sessions
from app.agent.prompts import build_system_prompt
from app.agent.titles import generate_title, placeholder_title
from app.config import settings
from app.db import database, queries
from app.db.models import USER_MD_TEMPLATE, Episode, Memory, Message, MessageRole, Session, user_md_is_blank
from app.llm.errors import LLMError
from tests.test_agent import FakeLLM, answer
from tests.test_routes import client, error_of, session_id  # noqa: F401 - fixtures

IST_MIDNIGHT_UTC = "2026-10-05T18:45:00+00:00"  # 00:15 on 6 October in India, still the 5th in UTC


@pytest.fixture
async def session(db: None) -> Session:
    return await queries.create_session()


async def chat(*turns: str) -> Session:
    """A session where the user and the assistant take turns saying `turns`."""
    s = await queries.create_session()
    for i, content in enumerate(turns):
        role: MessageRole = "user" if i % 2 == 0 else "assistant"
        await queries.save_message(Message(session_id=s.id, role=role, content=content))
    return s


async def set_last_active(chat_id: str, when: str) -> None:
    db = await database.get_db()
    await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (when, chat_id))
    await db.commit()


# ---- dates in the user's time zone ----


def test_a_utc_timestamp_lands_on_the_users_day() -> None:
    assert settings.timezone == "Asia/Kolkata"
    assert clock.local_date(IST_MIDNIGHT_UTC) == date(2026, 10, 6)
    assert clock.to_local(IST_MIDNIGHT_UTC).strftime("%H:%M") == "00:15"


@pytest.mark.parametrize("stamp", ["2026-10-05T18:45:00Z", "2026-10-05T18:45:00", "2026-10-06T00:15:00+05:30"])
def test_timestamps_in_any_stored_shape_mean_the_same_moment(stamp: str) -> None:
    assert clock.parse_utc(stamp) == datetime(2026, 10, 5, 18, 45, tzinfo=UTC)


def test_the_time_zone_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "timezone", "America/New_York")
    assert clock.local_date(IST_MIDNIGHT_UTC) == date(2026, 10, 5)


def test_an_unknown_time_zone_fails_at_startup() -> None:
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError, match="isn't an IANA time zone"):
        Settings(timezone="Mars/Olympus")


def test_the_agent_is_told_todays_date() -> None:
    prompt = build_system_prompt({}, [], now=datetime.fromisoformat(IST_MIDNIGHT_UTC))
    assert "## Today\nTuesday, 6 October 2026 (Asia/Kolkata)" in prompt


# ---- user.md ----


def test_the_template_is_blank_until_filled_in() -> None:
    assert user_md_is_blank(USER_MD_TEMPLATE)
    assert not user_md_is_blank(USER_MD_TEMPLATE.replace("Call me:", "Call me: Raghav"))


async def test_user_md_starts_from_the_template_and_saves(db: None) -> None:
    assert (await queries.get_user_profile())["content"] == USER_MD_TEMPLATE
    await queries.set_user_profile("Name: Raghav\nCall me: Raghav\n")
    assert (await queries.get_user_profile())["content"] == "Name: Raghav\nCall me: Raghav\n"


def test_user_md_is_in_the_prompt_only_when_filled_in() -> None:
    assert "About the user" not in build_system_prompt({}, [], user_profile=USER_MD_TEMPLATE)
    filled = USER_MD_TEMPLATE.replace("Call me:", "Call me: Raghav")
    prompt = build_system_prompt({}, [], user_profile=filled)
    assert "## About the user" in prompt and "<user.md>" in prompt and "Call me: Raghav" in prompt


async def test_the_agent_reads_user_md(session: Session) -> None:
    await queries.set_user_profile("Call me: Raghav\n")
    llm = FakeLLM(answer("Hi Raghav!"))
    await core.run_agent(session.id, "hey", llm=llm, small_llm=llm)
    assert "Call me: Raghav" in llm.agent_calls[0]["messages"][0]["content"]


async def test_facts_in_user_md_are_already_known_to_the_extractor(session: Session) -> None:
    await queries.set_user_profile("Name: Raghav\nNotes: wears size M\n")
    llm = FakeLLM(answer("[]"))
    await extract_memories("I'm a size M", None, session.id, llm)
    assert "- user.md: Notes: wears size M" in llm.calls[0]["messages"][1]["content"]


# ---- chat titles ----


async def test_a_chat_is_titled_by_its_first_message_then_a_short_title_is_scheduled(session: Session) -> None:
    jobs: list[tuple[Callable[..., Awaitable[object]], tuple[object, ...]]] = []
    llm = FakeLLM(answer("Here are some."), answer("Sure."))
    await core.run_agent(
        session.id,
        "I need running shoes under 4000 for daily jogging",
        llm=llm,
        small_llm=llm,
        schedule=lambda f, *a: jobs.append((f, a)),
    )
    current = await queries.get_session(session.id)
    assert current is not None and (current.title, current.title_source) == (
        "I need running shoes under 4000 for daily jogging",
        "placeholder",
    )
    assert jobs[0] == (generate_title, (session.id, "I need running shoes under 4000 for daily jogging", llm))

    jobs.clear()
    await core.run_agent(
        session.id, "the second one", llm=llm, small_llm=llm, schedule=lambda f, *a: jobs.append((f, a))
    )
    assert generate_title not in [f for f, _ in jobs]  # only after the first answer


def test_long_first_messages_are_shortened_at_a_word() -> None:
    title = placeholder_title("I am looking for a pair of noise cancelling headphones for long flights and the gym")
    assert len(title) <= 62 and title.endswith("…") and " " in title


async def test_generated_titles_are_saved(session: Session) -> None:
    await queries.set_session_title(session.id, "placeholder text", "placeholder")
    llm = FakeLLM(answer('"Running shoes under 4000."'))
    assert await generate_title(session.id, "I need running shoes under 4000", llm) == "Running shoes under 4000"
    current = await queries.get_session(session.id)
    assert current is not None and (current.title, current.title_source) == ("Running shoes under 4000", "llm")
    assert llm.calls[0]["name"] == "generate-chat-title"


async def test_a_generated_title_never_replaces_the_users(session: Session) -> None:
    await queries.set_session_title(session.id, "placeholder text", "placeholder")
    await queries.set_session_title(session.id, "Shoes for mom", "user")  # renamed while the title was made
    assert await generate_title(session.id, "running shoes", FakeLLM(answer("Running shoes"))) is None
    current = await queries.get_session(session.id)
    assert current is not None and current.title == "Shoes for mom"


async def test_titles_dont_count_as_activity(session: Session) -> None:
    await set_last_active(session.id, "2026-10-01T10:00:00+00:00")
    await queries.set_session_title(session.id, "Renamed", "user")
    current = await queries.get_session(session.id)
    assert current is not None and current.updated_at == "2026-10-01T10:00:00+00:00"


@pytest.mark.parametrize("reply", [LLMError("down"), answer(""), answer("   ")])
async def test_title_generation_never_raises(session: Session, reply: Any) -> None:
    await queries.set_session_title(session.id, "placeholder text", "placeholder")
    assert await generate_title(session.id, "shoes", FakeLLM(reply)) is None
    current = await queries.get_session(session.id)
    assert current is not None and current.title == "placeholder text"


# ---- the chat list ----


async def test_the_list_has_chats_most_recently_active_first(db: None) -> None:
    await queries.create_session()  # an empty session (a new tab): not a chat
    older = await chat("phones under 15k", "Here are some.")
    newer = await chat("earbuds", "Here are some.")
    deleted = await chat("laptops", "Here are some.")
    await set_last_active(older.id, "2026-10-01T10:00:00+00:00")
    await set_last_active(newer.id, "2026-10-03T10:00:00+00:00")
    await queries.delete_session(deleted.id)
    rows, total = await queries.list_chats()
    assert [r["id"] for r in rows] == [newer.id, older.id] and total == 2
    rows, total = await queries.list_chats(limit=1, offset=1)
    assert [r["id"] for r in rows] == [older.id] and total == 2


async def test_continuing_an_old_chat_moves_it_to_the_top(db: None) -> None:
    old = await chat("phones", "Here are some.")
    await set_last_active(old.id, "2026-09-01T10:00:00+00:00")
    other = await chat("earbuds", "Here are some.")
    await set_last_active(other.id, "2026-10-01T10:00:00+00:00")
    await queries.save_message(Message(session_id=old.id, role="user", content="and cheaper ones?"))
    rows, _ = await queries.list_chats()
    assert rows[0]["id"] == old.id


async def test_search_matches_titles_and_treats_wildcards_as_text(db: None) -> None:
    shoes = await chat("running shoes", "Here are some.")
    await queries.set_session_title(shoes.id, "Running Shoes under 4k", "llm")
    discount = await chat("deals", "Here are some.")
    await queries.set_session_title(discount.id, "50% off earbuds", "user")
    assert [r["id"] for r in (await queries.list_chats(query="shoes"))[0]] == [shoes.id]
    assert [r["id"] for r in (await queries.list_chats(query="50%"))[0]] == [discount.id]
    assert (await queries.list_chats(query="%"))[1] == 1


async def test_a_deleted_chat_is_gone_everywhere(db: None) -> None:
    current = await queries.create_session()
    old = await chat("phones", "Here are some.")
    assert await queries.delete_session(old.id) is True
    assert await queries.get_session(old.id) is None
    assert await queries.delete_session(old.id) is False
    assert await queries.set_session_title(old.id, "x", "user") is False
    assert await summarize_pending_sessions(current.id, FakeLLM()) == []  # not summarized either


# ---- an existing database gets the new columns ----


async def test_an_old_database_is_migrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as old:  # the schema before titles, deletes and user.md
        old.executescript(
            """CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                   budget REAL, context_summary TEXT);
               CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                   role TEXT NOT NULL, content TEXT NOT NULL, tool_name TEXT, tool_call_id TEXT,
                   created_at TEXT NOT NULL, token_count INTEGER);
               INSERT INTO sessions VALUES ('s1', '2026-09-01T10:00:00+00:00', '2026-09-01T10:05:00+00:00', NULL, NULL);
               INSERT INTO messages VALUES ('m1', 's1', 'user', 'wireless earbuds under 3000 with long battery life and ANC',
                   NULL, NULL, '2026-09-01T10:00:00+00:00', NULL);"""
        )
    monkeypatch.setattr(settings, "db_path", str(path))
    await database.close_db()
    try:
        await database.init_db()
        await database.init_db()  # and again: migrations run once
        current = await queries.get_session("s1")
        assert current is not None and current.title_source == "placeholder"
        assert current.title == "wireless earbuds under 3000 with long battery life and ANC"[:60]
        assert current.updated_at == "2026-09-01T10:05:00+00:00"  # the backfill isn't activity
        assert (await queries.get_user_profile())["content"] == USER_MD_TEMPLATE
        db = await database.get_db()
        assert "details" in {row[1] for row in await db.execute_fetchall("PRAGMA table_info(messages)")}
    finally:
        await database.close_db()


# ---- memory.md ----


async def test_memory_md_groups_facts_under_headings(db: None) -> None:
    await queries.save_memory(Memory(category="brand_dislike", content="dislikes boAt"))
    await queries.save_memory(Memory(category="size_info", content="wears size M in shoes", confidence=0.7))
    await queries.save_memory(Memory(category="brand_dislike", content="avoids Noise", created_at="2026-12-01"))
    file = await memory_files.get_file("memory.md")
    assert file.content == (
        "## Brands you avoid\n- dislikes boAt\n- avoids Noise\n\n## Sizes\n- wears size M in shoes (inferred)\n"
    )
    assert file.updated_at is not None


async def test_saving_memory_md_applies_the_edit(db: None) -> None:
    boat = Memory(category="brand_dislike", content="dislikes boAt")
    size = Memory(category="size_info", content="wears size M in shoes", confidence=0.7)
    amazon = Memory(category="retailer_preference", content="shops on Amazon")
    for memory in (boat, size, amazon):
        await queries.save_memory(memory)

    edited = (
        "## Brands you avoid\n"
        "- dislikes boAt\n"  # unchanged
        "\n## Sizes\n- wears size M in shoes\n"  # "(inferred)" removed: now stated
        "\n## Other\n- shops on Amazon\n"  # moved to another heading
        "- is a college student\n"  # added
    )  # and nothing about Flipkart, so nothing more is stored
    await memory_files.save_file("memory.md", edited)

    rows = {m["content"]: m for m in await queries.get_all_memories()}
    assert set(rows) == {"dislikes boAt", "wears size M in shoes", "shops on Amazon", "is a college student"}
    assert rows["dislikes boAt"]["id"] == boat.id and rows["dislikes boAt"]["updated_at"] == boat.updated_at
    assert rows["wears size M in shoes"]["confidence"] == 1.0
    assert (rows["shops on Amazon"]["id"], rows["shops on Amazon"]["category"]) == (amazon.id, "general")
    assert (rows["is a college student"]["category"], rows["is a college student"]["confidence"]) == ("general", 1.0)

    await memory_files.save_file("memory.md", "## Brands you avoid\n- dislikes boAt\n- avoids JBL (inferred)\n")
    rows = {m["content"]: m for m in await queries.get_all_memories()}
    assert set(rows) == {"dislikes boAt", "avoids JBL"}  # removed lines are forgotten
    assert rows["avoids JBL"]["confidence"] == 0.7


async def test_memory_md_round_trips(db: None) -> None:
    content = "## Brands you like\n- always buys Sony\n\n## Budget\n- usually spends under 5000 (inferred)\n"
    assert (await memory_files.save_file("memory.md", content)).content == content


async def test_memory_md_has_a_size_limit(db: None) -> None:
    too_many = "\n".join(f"- fact number {i}" for i in range(memory_files.MAX_FACTS + 1))
    with pytest.raises(memory_files.MemoryFileError, match="at most"):
        await memory_files.save_file("memory.md", too_many)


# ---- preferences.md ----


async def test_preferences_md_round_trips_and_applies(db: None) -> None:
    await queries.set_preference("preferred_brands", ["Sony", "Samsung"])
    await queries.set_preference("shoe_size", "M")
    file = await memory_files.get_file("preferences.md")
    assert file.content == '- preferred_brands: ["Sony", "Samsung"]\n- shoe_size: M\n'

    await memory_files.save_file("preferences.md", '- preferred_brands: ["Sony"]\n- budget_default: 5000\n')
    assert await queries.get_all_preferences() == {"preferred_brands": ["Sony"], "budget_default": 5000}


# ---- day files (short-term memory) ----


async def summarized(chat_session: Session, summary: str, last_active: str, summarized_at: str) -> Episode:
    await set_last_active(chat_session.id, last_active)
    episode = Episode(session_id=chat_session.id, summary=summary, outcome="browsed", created_at=summarized_at)
    await queries.save_episode(episode)
    return episode


async def test_day_files_follow_when_chats_happened_in_the_users_time_zone(db: None) -> None:
    late = await chat("earbuds", "Here are some.")
    await queries.set_session_title(late.id, "Gym earbuds", "llm")
    # 00:15 on the 6th in India, summarized two days later.
    await summarized(late, "Looked for gym earbuds.", IST_MIDNIGHT_UTC, "2026-10-08T09:00:00+00:00")
    earlier = await chat("phones", "Here are some.")
    await summarized(earlier, "Compared phones.", "2026-10-05T06:00:00+00:00", "2026-10-05T07:00:00+00:00")

    days = [f for f in await memory_files.list_files() if f.kind == "short_term"]
    assert [f.name for f in days] == ["2026-10-06.md", "2026-10-05.md"]  # newest day first
    assert days[0].day == date(2026, 10, 6) and days[0].updated_at == "2026-10-08T09:00:00+00:00"
    assert days[0].content.startswith("# Tuesday, 6 October 2026\n\n## 12:15 AM · Gym earbuds\n")
    assert "Looked for gym earbuds." in days[0].content and "Compared phones." not in days[0].content


async def test_editing_a_day_file_rewrites_and_forgets(db: None) -> None:
    keep = await chat("earbuds", "Here are some.")
    drop = await chat("phones", "Here are some.")
    kept = await summarized(keep, "Looked for earbuds.", "2026-10-05T06:00:00+00:00", "2026-10-05T07:00:00+00:00")
    await summarized(drop, "Compared phones.", "2026-10-05T08:00:00+00:00", "2026-10-05T09:00:00+00:00")
    current = await queries.create_session()

    file = await memory_files.get_file("2026-10-05.md")
    marker = f"<!-- chat {keep.id[:8]} -->"
    edited = (
        file.content.split("\n\n## ")[0] + f"\n\n## 11:30 AM · Earbuds\n{marker}\nLooked for gym earbuds under 2000.\n"
    )
    await memory_files.save_file("2026-10-05.md", edited)

    [row] = await queries.get_recent_episodes()
    assert (row["id"], row["summary"]) == (kept.id, "Looked for gym earbuds under 2000.")
    assert await queries.get_sessions_to_summarize(current.id) == []  # the forgotten chat stays forgotten


async def test_clearing_files(db: None) -> None:
    await queries.set_user_profile("Name: Raghav\n")
    await queries.save_memory(Memory(category="general", content="x"))
    await queries.set_preference("shoe_size", "M")
    old = await chat("earbuds", "Here are some.")
    await summarized(old, "Looked for earbuds.", "2026-10-05T06:00:00+00:00", "2026-10-05T07:00:00+00:00")

    for name in ("user.md", "memory.md", "preferences.md", "2026-10-05.md"):
        await memory_files.clear_file(name)
    assert (await queries.get_user_profile())["content"] == USER_MD_TEMPLATE
    assert await queries.get_all_memories() == [] and await queries.get_all_preferences() == {}
    assert await queries.get_recent_episodes() == []


@pytest.mark.parametrize("name", ["soul.md", "2026-13-01.md", "2026-10-05.md", "../memory.md"])
async def test_unknown_files(db: None, name: str) -> None:
    with pytest.raises(memory_files.MemoryFileError) as e:
        await memory_files.get_file(name)
    assert e.value.code == "unknown_file"


async def test_user_md_is_saved_as_written_within_a_limit(db: None) -> None:
    saved = await memory_files.save_file("user.md", "Name: Raghav\r\nNotes: vegetarian, likes minimal designs")
    assert saved.content == "Name: Raghav\nNotes: vegetarian, likes minimal designs\n"
    assert (await memory_files.save_file("user.md", "   ")).content == USER_MD_TEMPLATE
    with pytest.raises(memory_files.MemoryFileError, match="at most"):
        await memory_files.save_file("user.md", "x" * (memory_files.USER_MD_MAX_CHARS + 1))


# ---- routes ----


def run(test_client: TestClient, func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
    assert test_client.portal is not None
    return test_client.portal.call(func, *args)


def test_health_reports_the_time_zone(client: TestClient) -> None:  # noqa: F811
    assert client.get("/health").json()["timezone"] == "Asia/Kolkata"


def test_chat_list_rename_and_delete(client: TestClient, session_id: str) -> None:  # noqa: F811
    run(client, queries.save_message, Message(session_id=session_id, role="user", content="running shoes"))
    run(client, queries.set_session_title, session_id, "running shoes", "placeholder")

    body = client.get("/sessions").json()
    assert body["total"] == 1 and [(c["id"], c["title"]) for c in body["items"]] == [(session_id, "running shoes")]

    renamed = client.patch(f"/sessions/{session_id}", json={"title": "  Shoes   for mom "})
    assert renamed.status_code == 200 and renamed.json()["title"] == "Shoes for mom"
    assert client.get("/sessions", params={"q": "MOM"}).json()["total"] == 1

    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert client.get("/sessions").json() == {"items": [], "total": 0}
    assert error_of(client.post(f"/sessions/{session_id}/chat", json={"message": "hi"}))["code"] == "session_not_found"
    assert error_of(client.delete(f"/sessions/{session_id}"))["code"] == "session_not_found"
    assert error_of(client.patch(f"/sessions/{session_id}", json={"title": "x"}))["code"] == "session_not_found"


def test_rename_needs_a_title(client: TestClient, session_id: str) -> None:  # noqa: F811
    assert error_of(client.patch(f"/sessions/{session_id}", json={"title": "   "}))["code"] == "empty_title"


def test_memory_files_routes(client: TestClient) -> None:  # noqa: F811
    body = client.get("/memory/files").json()
    assert body["timezone"] == "Asia/Kolkata"
    assert [f["name"] for f in body["files"]] == ["user.md", "memory.md", "preferences.md"]
    assert body["files"][0]["content"] == USER_MD_TEMPLATE and body["files"][1]["updated_at"] is None

    saved = client.put("/memory/files/memory.md", json={"content": "## Sizes\n- wears size M\n"})
    assert saved.status_code == 200 and saved.json()["content"] == "## Sizes\n- wears size M\n"
    assert saved.json()["updated_at"]

    assert client.delete("/memory/files/memory.md").status_code == 204
    assert client.get("/memory/files").json()["files"][1]["content"] == ""

    missing = client.put("/memory/files/soul.md", json={"content": "x"})
    assert missing.status_code == 404 and error_of(missing)["code"] == "unknown_file"
    too_long = client.put("/memory/files/user.md", json={"content": "x" * 5000})
    assert too_long.status_code == 400 and error_of(too_long)["code"] == "too_long"
    assert json.loads(json.dumps(client.get("/memory/files").json()))  # plain JSON throughout
