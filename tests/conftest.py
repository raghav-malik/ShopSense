"""Shared test setup.

Tests are hermetic by default:
- config comes from the values below, not your .env, so dummy keys guarantee no
  test can call OpenAI or spend money;
- Langfuse tracing is off, so test runs don't show up in your dashboards;
- every test gets its own throwaway SQLite file, never your shopsense.db.

Only tests marked `network` touch the internet (DuckDuckGo, real product pages).
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

# Must run before any `app` import: Settings() and the Langfuse client are
# created at import time.
os.environ.update(
    {
        "LLM_PROVIDER": "openai",
        "LLM_API": "chat_completions",
        "OPENAI_API_KEY": "sk-test-not-a-real-key",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
        "LANGFUSE_BASE_URL": "http://localhost:1",
        "LANGFUSE_TRACING_ENABLED": "false",
        "MAX_AGENT_STEPS": "10",
    }
)

import pytest

import app.agent.core as agent_core
from app.config import settings
from app.db import database


@pytest.fixture
async def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """A fresh, initialized database for one test."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    await database.close_db()
    await database.init_db()
    yield
    await database.close_db()


@pytest.fixture
def isolated_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """For TestClient tests: the app's lifespan opens and closes the DB itself."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    return settings.db_path


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent must get its models from the test (llm=..., small_llm=...).
    Without this, a forgotten argument silently tries the real provider."""

    def refuse() -> None:
        raise AssertionError("run_agent needs llm= and small_llm= in tests; it tried to use a real provider")

    monkeypatch.setattr(agent_core, "get_llm_adapter", refuse)
    monkeypatch.setattr(agent_core, "get_small_llm_adapter", refuse)
