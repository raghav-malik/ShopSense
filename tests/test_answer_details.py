"""What the UI shows under an answer (product links, price checks, trace details,
follow-up suggestions) is saved with it, so a reopened chat shows it again."""

import json
from typing import Any

import pytest

import app.agent.core as core
import app.agent.price_check as price_check
from app.config import settings
from app.db import queries
from app.db.models import Message, Session
from tests.test_agent import (  # noqa: F401 - stub_search is autouse
    FakeLLM,
    answer,
    search_call,
    stub_search,
    tool_calls,
)

AMAZON = "https://www.amazon.in/boAt-Airdopes-141/dp/B09N3ZNHTY"


@pytest.fixture
async def session(db: None) -> Session:
    return await queries.create_session()


async def last_answer_details(chat_id: str) -> dict[str, Any]:
    [answer_row] = [m for m in await queries.get_messages(chat_id) if m["role"] == "assistant"][-1:]
    assert answer_row["details"] is not None
    details: dict[str, Any] = json.loads(answer_row["details"])
    return details


async def test_an_answer_is_saved_with_its_products_price_checks_and_trace_details(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def store_page(url: str) -> tuple[float | None, bool | None, str | None]:
        return 1099.0, True, "boAt Airdopes 141"

    monkeypatch.setattr(price_check, "_live_details", store_page)
    llm = FakeLLM(tool_calls(search_call()), answer(f"**boAt Airdopes 141** — ₹1,099. [Buy on Amazon.in]({AMAZON})"))
    result = await core.run_agent(session.id, "earbuds under 3000", llm=llm, small_llm=llm)

    details = await last_answer_details(session.id)
    assert details["products_found"] == json.loads(json.dumps(result.products_found))
    assert details["tool_calls_made"] == ["search_products"]
    assert details["step_count"] == 2 and details["total_tokens"] == result.total_tokens
    assert [(c["url"], c["status"], c["live_price"]) for c in details["price_checks"]] == [(AMAZON, "verified", 1099.0)]
    assert "response" not in details  # the text is the message itself


async def test_an_answer_at_a_turn_limit_is_saved_with_its_details_too(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_agent_steps", 1)
    llm = FakeLLM(tool_calls(search_call()), answer("From what I found: the boAt Airdopes 141."))
    await core.run_agent(session.id, "earbuds", llm=llm, small_llm=llm)
    details = await last_answer_details(session.id)
    assert details["products_found"] and details["step_count"] == 1


async def test_saved_products_are_capped(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    many = [
        {"title": f"Product {i}", "url": f"https://shop.example/{i}", "snippet": "x", "source": "x"} for i in range(80)
    ]
    monkeypatch.setattr("app.tools.search._ddgs_text", lambda query, max_results: many[:max_results])
    llm = FakeLLM(tool_calls(*(search_call(f"c{i}", f"q{i}") for i in range(8))), answer("Here are some."))
    result = await core.run_agent(session.id, "earbuds", llm=llm, small_llm=llm)
    assert len(result.products_found) > core._SAVED_PRODUCTS  # the reply itself has them all
    assert len((await last_answer_details(session.id))["products_found"]) == core._SAVED_PRODUCTS


async def test_suggestions_are_added_to_the_latest_answer(session: Session) -> None:
    for role, content in (
        ("user", "earbuds"),
        ("assistant", "Here are some."),
        ("user", "cheaper?"),
        ("assistant", "Sure."),
    ):
        await queries.save_message(Message(session_id=session.id, role=role, content=content))  # type: ignore[arg-type]
    before = (await queries.get_session(session.id)).updated_at  # type: ignore[union-attr]

    assert await queries.add_to_latest_answer(session.id, {"suggestions": ["Show me more"]}) is True
    assert await queries.add_to_latest_answer(session.id, {"seen": True}) is True  # merged, not replaced
    rows = [m for m in await queries.get_messages(session.id) if m["role"] == "assistant"]
    assert rows[0]["details"] is None
    assert json.loads(rows[1]["details"] or "{}") == {"suggestions": ["Show me more"], "seen": True}
    assert (await queries.get_session(session.id)).updated_at == before  # type: ignore[union-attr]


async def test_no_answer_no_suggestions(session: Session) -> None:
    assert await queries.add_to_latest_answer(session.id, {"suggestions": ["x"]}) is False
