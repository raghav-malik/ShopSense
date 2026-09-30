"""Request ids, structured logs, the unhandled-error path, and /livez /readyz."""

import asyncio
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.agent.trace_attributes import trace_attributes
from app.db import queries
from app.llm.adapter import ChatCompletionsAdapter
from app.main import app
from app.request_context import JsonFormatter, RequestIdFilter, current_request_id, request_id_var


class _OfflineLangfuse:
    def auth_check(self) -> bool:
        return False


@pytest.fixture
def client(isolated_db_path: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(main_module, "get_langfuse", lambda: _OfflineLangfuse())
    monkeypatch.setattr(main_module, "shutdown_langfuse", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class Collect(logging.Handler):
    """Log records as our real handler sees them (with the request id filter)."""

    def __init__(self) -> None:
        super().__init__()
        self.addFilter(RequestIdFilter())
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def logs() -> Iterator[Collect]:
    handler = Collect()
    shopsense = logging.getLogger("shopsense")
    shopsense.addHandler(handler)
    try:
        yield handler
    finally:
        shopsense.removeHandler(handler)


# ---- request ids ----


def test_every_response_gets_a_request_id(client: TestClient) -> None:
    first = client.get("/livez").headers["x-request-id"]
    second = client.get("/livez").headers["x-request-id"]
    assert len(first) == 32 and first != second


def test_a_sane_incoming_request_id_is_kept(client: TestClient) -> None:
    response = client.get("/livez", headers={"X-Request-ID": "frontend-abc.123"})
    assert response.headers["x-request-id"] == "frontend-abc.123"


@pytest.mark.parametrize("bad", ["x" * 65, "has space", "new\nline", "<script>", ""])
def test_an_odd_incoming_request_id_is_replaced(client: TestClient, bad: str) -> None:
    response = client.get("/livez", headers={"X-Request-ID": bad} if "\n" not in bad else {})
    assert response.headers["x-request-id"] != bad and len(response.headers["x-request-id"]) == 32


def test_code_handling_a_request_sees_its_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []
    real_get_cart = queries.get_cart

    async def get_cart(session_id: str) -> list:  # type: ignore[type-arg]
        seen.append(current_request_id())
        return await real_get_cart(session_id)

    monkeypatch.setattr(queries, "get_cart", get_cart)
    session_id = client.post("/sessions").json()["session_id"]
    response = client.get(f"/sessions/{session_id}/cart", headers={"X-Request-ID": "req-42"})
    assert response.status_code == 200 and seen == ["req-42"]


def test_each_request_is_logged_once_with_its_id(client: TestClient, logs: Collect) -> None:
    client.post("/sessions", headers={"X-Request-ID": "req-7"})
    (line,) = [r for r in logs.records if r.name == "shopsense.http"]
    assert line.request_id == "req-7"  # type: ignore[attr-defined]
    assert (line.method, line.path, line.status) == ("POST", "/sessions", 201)  # type: ignore[attr-defined]
    assert line.duration_ms >= 0  # type: ignore[attr-defined]


def test_probes_are_not_logged_at_info(client: TestClient, logs: Collect) -> None:
    client.get("/livez")
    client.get("/readyz")
    assert [r for r in logs.records if r.name == "shopsense.http" and r.levelno >= logging.INFO] == []


def test_agent_traces_carry_the_request_id() -> None:
    llm = ChatCompletionsAdapter()
    token = request_id_var.set("req-99")
    try:
        inside = trace_attributes(trace_name="run-agent", session_id="s", llm=llm)
    finally:
        request_id_var.reset(token)
    assert inside["metadata"]["request_id"] == "req-99"
    outside = trace_attributes(trace_name="run-agent", session_id="s", llm=llm)
    assert "request_id" not in outside["metadata"]  # scripts and evals have no request


# ---- unhandled errors ----


def test_unhandled_error_is_a_standard_500_with_the_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, logs: Collect
) -> None:
    async def broken(session_id: str) -> None:
        raise RuntimeError("secret internal detail: password=hunter2")

    session_id = client.post("/sessions").json()["session_id"]
    monkeypatch.setattr(queries, "get_cart", broken)
    response = client.get(f"/sessions/{session_id}/cart", headers={"X-Request-ID": "req-500"})

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Something went wrong. Please try again."}
    }
    assert "hunter2" not in response.text
    assert response.headers["x-request-id"] == "req-500"
    (error,) = [r for r in logs.records if r.levelno == logging.ERROR]
    assert error.request_id == "req-500" and error.exc_info is not None  # type: ignore[attr-defined]


# ---- probes ----


def test_livez(client: TestClient) -> None:
    assert client.get("/livez").json() == {"status": "ok"}


def test_readyz_when_ready(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready" and body["checks"] == {"startup": "ok", "database": "ok"}
    assert body["langfuse"] == "not connected"  # reported, but doesn't decide readiness


def test_readyz_when_the_database_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable() -> None:
        raise OSError("disk I/O error")

    monkeypatch.setattr(main_module, "get_db", unavailable)
    response = client.get("/readyz")
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "not_ready" and error["details"] == [{"check": "database", "status": "failed"}]
    assert "disk" not in response.text  # the reason goes to the log, not the response


def test_readyz_when_the_database_hangs(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def hang() -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(main_module, "get_db", hang)
    monkeypatch.setattr(main_module, "READINESS_DB_TIMEOUT", 0.05)
    assert client.get("/readyz").status_code == 503


def test_readyz_before_startup_finishes_or_after_shutdown_begins(client: TestClient) -> None:
    app.state.ready = False
    try:
        response = client.get("/readyz")
    finally:
        app.state.ready = True
    assert response.status_code == 503
    assert {"check": "startup", "status": "failed"} in response.json()["error"]["details"]


# ---- log format ----


def test_json_log_lines() -> None:
    record = logging.makeLogRecord(
        {"name": "shopsense.http", "levelno": logging.INFO, "levelname": "INFO", "msg": "GET %s", "args": ("/livez",)}
    )
    record.request_id = "req-1"
    record.status = 200
    record.color_message = "GET [1m%s[0m"  # uvicorn's colored copy: dropped
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "GET /livez" and line["request_id"] == "req-1" and line["status"] == 200
    assert line["level"] == "INFO" and line["logger"] == "shopsense.http" and line["ts"].endswith("+00:00")
    assert "color_message" not in line


def test_json_log_lines_include_the_traceback() -> None:
    error = ValueError("boom")
    record = logging.getLogger("shopsense").makeRecord(
        "shopsense", logging.ERROR, __file__, 1, "failed", (), exc_info=(ValueError, error, None)
    )
    line = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in line["exception"] and "request_id" not in line  # outside a request
