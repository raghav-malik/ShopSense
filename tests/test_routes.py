"""API endpoints through FastAPI's TestClient. The agent is mocked where a route
would call it, so these tests never reach an LLM."""

import json
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Protocol

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.routes.chat as chat_routes
import app.routes.sessions as session_routes
from app.agent.schemas import AgentResponse
from app.db import queries
from app.db.models import Episode, Memory, Message, MessageRole
from app.llm.errors import LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import JSONObject
from app.main import app

AMAZON = "https://www.amazon.in/boAt-Airdopes-141/dp/B09N3ZNHTY"


class _OfflineLangfuse:
    """Startup checks Langfuse auth; answer locally instead of calling the network."""

    def auth_check(self) -> bool:
        return False


@pytest.fixture
def client(isolated_db_path: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Synchronous test client for FastAPI, with its own database."""
    monkeypatch.setattr(main_module, "get_langfuse", lambda: _OfflineLangfuse())
    monkeypatch.setattr(main_module, "shutdown_langfuse", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def session_id(client: TestClient) -> str:
    sid: str = client.post("/sessions").json()["session_id"]
    return sid


class _Response(Protocol):
    """What error_of needs from TestClient's response (httpx or httpx2, by what's installed)."""

    def json(self) -> Any: ...


def error_of(response: _Response) -> JSONObject:
    body = response.json()
    assert set(body) == {"error"}, body
    error: JSONObject = body["error"]
    return error


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm"] == "openai/gpt-6-luna"
    assert body["langfuse_url"]


def test_create_session(client: TestClient) -> None:
    response = client.post("/sessions")
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data


def test_sessions_are_unique(client: TestClient) -> None:
    assert client.post("/sessions").json()["session_id"] != client.post("/sessions").json()["session_id"]


def test_a_new_session_summarizes_earlier_ones_in_the_background(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    async def summarize_pending(new_session_id: str) -> list[Episode]:
        started.append(new_session_id)
        return []

    monkeypatch.setattr(session_routes, "summarize_pending_sessions", summarize_pending)
    response = client.post("/sessions")
    assert response.status_code == 201 and started == [response.json()["session_id"]]


# ---- session summary ----


def test_summarize_returns_the_episode(client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    small_llm = object()
    seen: list[tuple[str, object]] = []

    async def summarize(sid: str, llm: object) -> Episode:
        seen.append((sid, llm))
        return Episode(
            session_id=sid,
            summary="Looked for earbuds under ₹3,000 and carted the boAt Airdopes 141.",
            products_searched='["wireless earbuds"]',
            products_carted='["boAt Airdopes 141"]',
            outcome="carted",
            created_at="2026-10-06T08:00:00+00:00",
        )

    monkeypatch.setattr(chat_routes, "get_small_llm_adapter", lambda: small_llm)
    monkeypatch.setattr(chat_routes, "summarize_session", summarize)
    response = client.post(f"/sessions/{session_id}/summarize")
    assert response.status_code == 200
    assert response.json() == {
        "session_id": session_id,
        "summary": "Looked for earbuds under ₹3,000 and carted the boAt Airdopes 141.",
        "products_searched": ["wireless earbuds"],
        "products_carted": ["boAt Airdopes 141"],
        "outcome": "carted",
        "created_at": "2026-10-06T08:00:00+00:00",
    }
    assert seen == [(session_id, small_llm)]  # on the small model


def test_summarize_returns_null_when_there_is_nothing_to_summarize(
    client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def too_short(sid: str, llm: object) -> None:
        return None

    monkeypatch.setattr(chat_routes, "get_small_llm_adapter", lambda: object())
    monkeypatch.setattr(chat_routes, "summarize_session", too_short)
    response = client.post(f"/sessions/{session_id}/summarize")
    assert response.status_code == 200 and response.json() is None


def test_summarize_unknown_session(client: TestClient) -> None:
    response = client.post("/sessions/nonexistent/summarize")
    assert response.status_code == 404
    assert error_of(response)["code"] == "session_not_found"


# ---- chat ----


def test_chat_invalid_session(client: TestClient) -> None:
    response = client.post("/sessions/nonexistent/chat", json={"message": "hello"})
    assert response.status_code == 404
    assert error_of(response)["code"] == "session_not_found"


@pytest.mark.parametrize("message", ["", "   "])
def test_chat_empty_message(client: TestClient, session_id: str, message: str) -> None:
    response = client.post(f"/sessions/{session_id}/chat", json={"message": message})
    assert response.status_code == 400
    assert error_of(response) == {"code": "empty_message", "message": "Message cannot be empty"}


def test_chat_missing_message_field(client: TestClient, session_id: str) -> None:
    response = client.post(f"/sessions/{session_id}/chat", json={})
    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"] == ["body", "message"]


def test_chat_message_too_long(client: TestClient, session_id: str) -> None:
    response = client.post(f"/sessions/{session_id}/chat", json={"message": "a" * 4001})
    assert response.status_code == 422


def test_chat_returns_agent_response(client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, JSONObject]] = []

    async def fake_run_agent(sid: str, message: str, **kwargs: Any) -> AgentResponse:
        calls.append((sid, message, kwargs))
        return AgentResponse(
            response="Try the boAt Airdopes 141.",
            tool_calls_made=["search_products"],
            products_found=[{"title": "boAt", "url": "https://x"}],
            step_count=2,
            total_tokens=1234,
        )

    monkeypatch.setattr(chat_routes, "run_agent", fake_run_agent)
    response = client.post(f"/sessions/{session_id}/chat", json={"message": "earbuds"})
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Try the boAt Airdopes 141."
    assert body["products_found"] and "suggestions" not in body  # fetched separately, after the answer
    # The route picks the Langfuse trace id up front so failures can link to it.
    ((sid, message, kwargs),) = calls
    assert sid == session_id and message == "earbuds" and len(kwargs["langfuse_trace_id"]) == 32


def test_chat_runs_scheduled_work_after_the_response(
    client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory extraction goes through BackgroundTasks: it runs after the answer is sent."""
    ran: list[str] = []

    async def learn(message: str) -> None:
        ran.append(message)

    async def fake_run_agent(sid: str, message: str, **kwargs: Any) -> AgentResponse:
        kwargs["schedule"](learn, message)
        assert ran == []  # scheduled, not run while the agent is still answering
        return AgentResponse(response="ok", step_count=1)

    monkeypatch.setattr(chat_routes, "run_agent", fake_run_agent)
    response = client.post(f"/sessions/{session_id}/chat", json={"message": "I always buy Sony"})
    assert response.status_code == 200 and ran == ["I always buy Sony"]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMRateLimitError("rate limited", retry_after=12), 503, "llm_rate_limited"),
        (LLMTimeoutError("no response"), 504, "llm_timeout"),
        (LLMUnavailableError("model missing"), 502, "llm_unavailable"),
        (RuntimeError("secret internal detail: password=hunter2"), 500, "agent_error"),
    ],
)
def test_chat_errors_are_structured_with_trace_url(
    client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int, code: str
) -> None:
    async def failing_run_agent(*args: object, **kwargs: object) -> AgentResponse:
        raise error

    async def fake_trace_url(trace_id: str) -> str:
        return f"https://langfuse.example/traces/{trace_id}"

    monkeypatch.setattr(chat_routes, "run_agent", failing_run_agent)
    monkeypatch.setattr(chat_routes, "_trace_url", fake_trace_url)
    response = client.post(f"/sessions/{session_id}/chat", json={"message": "earbuds"})

    assert response.status_code == status
    body = error_of(response)
    assert body["code"] == code
    assert body["trace_url"].startswith("https://langfuse.example/traces/")
    assert "hunter2" not in response.text  # internals never reach the client
    if status == 503:
        assert response.headers["retry-after"] == "12"


# ---- cart & history ----


# ---- suggestions (fetched after the answer) ----


def test_suggestions_for_the_saved_conversation(
    client: TestClient, session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, list[Any]]] = []

    async def fake_suggest(sid: str, history: list[Any]) -> list[str]:
        seen.append((sid, history))
        return ["Compare these two", "Show cheaper options"]

    monkeypatch.setattr(chat_routes, "suggest_follow_ups", fake_suggest)
    response = client.post(f"/sessions/{session_id}/suggestions")
    assert response.status_code == 200
    assert response.json() == {"suggestions": ["Compare these two", "Show cheaper options"]}
    assert seen == [(session_id, [])]  # a new session has no messages yet


def test_suggestions_nonexistent_session(client: TestClient) -> None:
    response = client.post("/sessions/nonexistent/suggestions")
    assert response.status_code == 404
    assert error_of(response)["code"] == "session_not_found"


def test_cart_empty(client: TestClient, session_id: str) -> None:
    response = client.get(f"/sessions/{session_id}/cart")
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert response.json()["budget"] is None


def test_cart_reports_the_session_budget(client: TestClient, session_id: str) -> None:
    import asyncio

    from app.db import queries

    asyncio.run(queries.update_session_budget(session_id, 3000))
    assert client.get(f"/sessions/{session_id}/cart").json()["budget"] == 3000


def test_cart_nonexistent_session(client: TestClient) -> None:
    response = client.get("/sessions/nonexistent/cart")
    assert response.status_code == 404
    assert error_of(response)["code"] == "session_not_found"


def test_history_new_session(client: TestClient, session_id: str) -> None:
    response = client.get(f"/sessions/{session_id}/history")
    assert response.status_code == 200
    assert response.json()["messages"] == []
    assert response.json()["session"]["id"] == session_id


def test_history_nonexistent_session(client: TestClient) -> None:
    response = client.get("/sessions/nonexistent/history")
    assert response.status_code == 404


def test_unknown_route_uses_error_shape(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert error_of(response)["code"] == "http_404"


# ---- memory ----


def run(client: TestClient, func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
    """Run a query in the app's event loop, where its database connection lives."""
    assert client.portal is not None
    return client.portal.call(func, *args)


@pytest.fixture
def remembered(client: TestClient, session_id: str) -> dict[str, str]:
    """A preference, a learned fact and a past-session summary."""
    memory = Memory(category="brand_dislike", content="dislikes boAt", confidence=0.8)
    episode = Episode(session_id=session_id, summary="Looked for earbuds under ₹3,000.", outcome="browsed")
    run(client, queries.set_preference, "preferred_brands", ["Sony"])
    run(client, queries.save_memory, memory)
    run(client, queries.save_episode, episode)
    return {"memory": memory.id, "episode": episode.id}


def test_memory_overview(client: TestClient, session_id: str, remembered: dict[str, str]) -> None:
    body = client.get("/memory").json()
    assert body["preferences"] == {"preferred_brands": ["Sony"]}
    [memory] = body["memories"]
    assert (memory["id"], memory["category"], memory["content"], memory["confidence"]) == (
        remembered["memory"],
        "brand_dislike",
        "dislikes boAt",
        0.8,
    )
    [episode] = body["episodes"]
    assert (episode["id"], episode["session_id"], episode["summary"], episode["outcome"]) == (
        remembered["episode"],
        session_id,
        "Looked for earbuds under ₹3,000.",
        "browsed",
    )


def test_memory_overview_when_empty(client: TestClient) -> None:
    assert client.get("/memory").json() == {"preferences": {}, "memories": [], "episodes": []}


@pytest.mark.parametrize(
    ("path", "kind", "code"),
    [
        ("/memory/memories/{memory}", "memories", "memory_not_found"),
        ("/memory/episodes/{episode}", "episodes", "episode_not_found"),
    ],
)
def test_forget_one_item(client: TestClient, remembered: dict[str, str], path: str, kind: str, code: str) -> None:
    url = path.format(**remembered)
    assert client.delete(url).status_code == 204
    assert client.get("/memory").json()[kind] == []
    response = client.delete(url)  # already gone
    assert response.status_code == 404 and error_of(response)["code"] == code


def test_forget_a_preference(client: TestClient, remembered: dict[str, str]) -> None:
    assert client.delete("/memory/preferences/preferred_brands").status_code == 204
    assert client.get("/memory").json()["preferences"] == {}
    assert error_of(client.delete("/memory/preferences/preferred_brands"))["code"] == "preference_not_found"


def test_forget_everything(client: TestClient, remembered: dict[str, str]) -> None:
    assert client.delete("/memory").status_code == 204
    assert client.get("/memory").json() == {"preferences": {}, "memories": [], "episodes": []}


# ---- answer details in the history ----


def say(test_client: TestClient, chat_id: str, role: MessageRole, content: str, **fields: Any) -> None:
    run(test_client, queries.save_message, Message(session_id=chat_id, role=role, content=content, **fields))


def test_suggestions_are_saved_and_come_back_with_the_history(
    client: TestClient,
    session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def suggest(sid: str, history: list[Any]) -> list[str]:
        return ["Show me cheaper options", "Compare these two"]

    monkeypatch.setattr(chat_routes, "suggest_follow_ups", suggest)
    say(client, session_id, "user", "earbuds")
    say(client, session_id, "assistant", "Here are some.", details=json.dumps({"products_found": [], "step_count": 1}))
    assert client.post(f"/sessions/{session_id}/suggestions").json() == {
        "suggestions": ["Show me cheaper options", "Compare these two"]
    }
    [_, reply] = client.get(f"/sessions/{session_id}/history").json()["messages"]
    assert json.loads(reply["details"]) == {
        "products_found": [],
        "step_count": 1,
        "suggestions": ["Show me cheaper options", "Compare these two"],
    }


def test_older_answers_get_their_links_back_from_the_search_results(
    client: TestClient,
    session_id: str,
) -> None:
    hits = [{"title": "boAt Airdopes 141", "url": AMAZON, "snippet": "₹1,099", "source": "amazon.in"}]
    say(client, session_id, "user", "earbuds")
    say(client, session_id, "tool", json.dumps({"results": hits}), tool_name="search_products", tool_call_id="c1")
    say(client, session_id, "assistant", f"[boAt]({AMAZON})")  # saved before details existed
    say(client, session_id, "user", "thanks")
    say(client, session_id, "assistant", "You're welcome!")  # no search this turn: nothing to recover

    messages = client.get(f"/sessions/{session_id}/history").json()["messages"]
    answers = [m for m in messages if m["role"] == "assistant"]
    assert json.loads(answers[0]["details"]) == {"products_found": hits}
    assert answers[1]["details"] is None
