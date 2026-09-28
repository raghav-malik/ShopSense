"""API keys never show up in printed settings, logs or traces; the clients
still get the real value."""

import logging

import pytest
from langfuse.types import MaskOtelSpansParams, OtelSpanData, OtelSpanIdentifier
from pydantic import SecretStr, ValidationError

from app.config import Settings, settings
from app.llm.adapter import ChatCompletionsAdapter
from app.tracing.langfuse_setup import mask_otel_spans

# Built at runtime so this file holds no key-shaped strings.
OPENAI_KEY = "sk-proj-" + "Q7x" * 12
GEMINI_KEY = "AQ." + "Zy9" * 12  # not the classic "AIza..." shape the regexes know
LANGFUSE_SECRET = "sk-lf-" + "4c1d" * 8


def configured() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key=OPENAI_KEY,  # type: ignore[arg-type]
        gemini_api_key=GEMINI_KEY,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key=LANGFUSE_SECRET,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "render",
    [repr, str, lambda s: s.model_dump_json(), lambda s: str(s.model_dump()), lambda s: f"{s}"],
    ids=["repr", "str", "model_dump_json", "model_dump", "f-string"],
)
def test_keys_never_appear_when_settings_are_printed(render: object) -> None:
    text = render(configured())  # type: ignore[operator]
    for key in (OPENAI_KEY, GEMINI_KEY, LANGFUSE_SECRET):
        assert key not in text
    assert "**********" in text


def test_keys_never_appear_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        logging.getLogger("shopsense").info("config: %s", configured())
    assert OPENAI_KEY not in caplog.text and LANGFUSE_SECRET not in caplog.text


def test_the_llm_client_gets_the_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(OPENAI_KEY))
    assert ChatCompletionsAdapter().client.api_key == OPENAI_KEY


def test_an_empty_key_counts_as_missing() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        Settings(_env_file=None, llm_provider="openai", openai_api_key="")  # type: ignore[arg-type]


def test_traces_redact_configured_keys_in_any_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(GEMINI_KEY))
    identifier = OtelSpanIdentifier(trace_id="f" * 32, span_id="1" * 16)
    span = OtelSpanData(
        trace_id="f" * 32,
        span_id="1" * 16,
        parent_span_id=None,
        name="generate-agent-response",
        instrumentation_scope_name="langfuse",
        instrumentation_scope_version=None,
        attributes={"langfuse.observation.output": f"error: invalid key {GEMINI_KEY} for this project"},
        resource_attributes={},
    )
    result = mask_otel_spans(params=MaskOtelSpansParams(spans={identifier: span}))
    assert result is not None
    patch = result.span_patches[identifier]
    assert patch is not None
    assert patch.set_attributes == {"langfuse.observation.output": "error: invalid key [SECRET] for this project"}
