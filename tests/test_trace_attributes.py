"""Every trace carries the model, provider, API, limits and app version (tags,
metadata and version), like Airtap's per-call traces do."""

import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import app.agent.core as core
import app.agent.suggestions as suggestions_module
from app import __version__
from app.agent.trace_attributes import trace_attributes
from app.config import settings
from app.db import queries
from app.db.models import Session
from app.llm.adapter import ChatCompletionsAdapter, LLMAdapter, ResponsesAdapter
from app.llm.types import ChatMessage, JSONObject, LLMResponse


class DescribedLLM(LLMAdapter):
    """A fake that answers once and describes itself like a real adapter."""

    def __init__(self, model: str) -> None:
        self.model = model

    def describe(self) -> dict[str, str]:
        return {"model": self.model, "provider": "openai", "api": "chat_completions", "reasoning_effort": "none"}

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[JSONObject] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
        trace_metadata: JSONObject | None = None,
    ) -> LLMResponse:
        usage = {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}
        return LLMResponse(content="Try the boAt Airdopes 141.", finish_reason="stop", usage=usage, model=self.model)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The keyword arguments each trace was started with."""
    calls: list[dict[str, Any]] = []

    @contextmanager
    def record(**kwargs: Any) -> Iterator[None]:
        calls.append(kwargs)
        yield

    monkeypatch.setattr(core, "propagate_attributes", record)
    monkeypatch.setattr(suggestions_module, "propagate_attributes", record)
    return calls


def test_the_app_version_matches_pyproject() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def test_adapters_describe_how_they_call_their_model() -> None:
    assert ChatCompletionsAdapter().describe() == {
        "model": settings.llm_model,
        "provider": "openai",
        "api": "chat_completions",
        "reasoning_effort": "none",
    }
    assert ResponsesAdapter(reasoning_effort="medium").describe()["api"] == "responses"
    assert ChatCompletionsAdapter(model="gpt-6-sol").describe()["model"] == "gpt-6-sol"


def test_trace_attributes() -> None:
    attributes = trace_attributes(
        trace_name="run-agent", session_id="s1", llm=DescribedLLM("gpt-6-sol"), small_llm=DescribedLLM("gpt-6-luna")
    )
    assert attributes["trace_name"] == "run-agent" and attributes["session_id"] == "s1"
    assert attributes["version"] == __version__
    assert attributes["tags"] == ["model:gpt-6-sol", "provider:openai", "api:chat_completions"]
    metadata = attributes["metadata"]
    assert metadata["model"] == "gpt-6-sol" and metadata["small_model"] == "gpt-6-luna"
    assert metadata["max_agent_steps"] == str(settings.max_agent_steps)
    # Langfuse's rule for propagated metadata: strings of at most 200 characters.
    assert all(isinstance(v, str) and len(v) <= 200 for v in metadata.values())


async def test_agent_traces_carry_the_attributes(session: Session, captured: list[dict[str, Any]]) -> None:
    await core.run_agent(session.id, "earbuds", llm=DescribedLLM("gpt-6-sol"), small_llm=DescribedLLM("gpt-6-luna"))
    (trace,) = captured
    assert trace["trace_name"] == "run-agent" and trace["session_id"] == session.id
    assert "model:gpt-6-sol" in trace["tags"] and trace["metadata"]["small_model"] == "gpt-6-luna"


async def test_suggestion_traces_carry_the_small_model(session: Session, captured: list[dict[str, Any]]) -> None:
    await suggestions_module.suggest_follow_ups(session.id, await queries.get_messages(session.id), DescribedLLM("m"))
    (trace,) = captured
    assert trace["trace_name"] == "suggest-follow-ups" and trace["tags"][0] == "model:m"


@pytest.fixture
async def session(db: None) -> Session:
    return await queries.create_session()
