"""Shared test setup.

Tests are hermetic by default:
- config comes from the values below, not your .env, so dummy keys guarantee no
  test can call OpenAI or spend money;
- Langfuse tracing is off, so test runs don't show up in your dashboards;
- every test gets its own throwaway SQLite file, never your shopsense.db.

Only tests marked `network` touch the internet (DuckDuckGo, real product pages).
"""

import os

# Must run before any `app` import: Settings() and the Langfuse client are
# created at import time.
os.environ.update({
    "LLM_PROVIDER": "openai",
    "LLM_API": "chat_completions",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "LANGFUSE_BASE_URL": "http://localhost:1",
    "LANGFUSE_TRACING_ENABLED": "false",
    "MAX_AGENT_STEPS": "10",
})

import pytest  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import database  # noqa: E402


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """A fresh, initialized database for one test."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    await database.close_db()
    await database.init_db()
    yield
    await database.close_db()


@pytest.fixture
def isolated_db_path(tmp_path, monkeypatch):
    """For TestClient tests: the app's lifespan opens and closes the DB itself."""
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    return settings.db_path
