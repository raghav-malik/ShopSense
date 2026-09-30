"""API endpoints through FastAPI's TestClient. The agent is mocked where a route
would call it, so these tests never reach an LLM."""

from collections.abc import Iterator
from typing import Any, Protocol

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.routes.chat as chat_routes
from app.agent.schemas import AgentResponse
from app.llm.errors import LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import JSONObject
from app.main import app


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
