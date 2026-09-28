"""Tracing must never break the app: GenerationTrace's guards, and the PII /
secret masking applied to every span before export."""

import logging
from types import TracebackType
from typing import Any

import httpx2
import openai
import pytest
from langfuse.types import MaskOtelSpansParams, OtelSpanData, OtelSpanIdentifier

import app.tracing.generation as generation_module
from app.tracing.generation import GenerationTrace
from app.tracing.langfuse_setup import _redact, mask_otel_spans

# ---- GenerationTrace ----


class FakeGeneration:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeObservation:
    """What start_as_current_observation returns: a context manager around the generation."""

    def __init__(self, generation: FakeGeneration, fail_on_exit: bool) -> None:
        self.generation = generation
        self.fail_on_exit = fail_on_exit

    def __enter__(self) -> FakeGeneration:
        return self.generation

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self.fail_on_exit:
            raise RuntimeError("export failed")


class FakeLangfuse:
    def __init__(self, *, fail_on_start: bool = False, fail_on_exit: bool = False) -> None:
        self.fail_on_start = fail_on_start
        self.fail_on_exit = fail_on_exit
        self.generation = FakeGeneration()
        self.started: list[dict[str, Any]] = []

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail_on_start:
            raise RuntimeError("langfuse misconfigured")
        self.started.append(kwargs)
        return FakeObservation(self.generation, self.fail_on_exit)


def install(monkeypatch: pytest.MonkeyPatch, **failures: bool) -> FakeLangfuse:
    fake = FakeLangfuse(**failures)
    monkeypatch.setattr(generation_module, "get_langfuse", lambda: fake)
    return fake


def test_success_records_output_and_keeps_start_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    with GenerationTrace("generate-response", model="m", input="hi", metadata={"step": 1}) as trace:
        trace.success(lambda: {"output": "Hello", "metadata": {"finish_reason": "stop"}})

    (started,) = fake.started
    assert started["as_type"] == "generation" and started["name"] == "generate-response"
    assert started["model"] == "m" and started["input"] == "hi"
    # Langfuse replaces metadata on update, so the start keys are sent again.
    assert fake.generation.updates == [{"output": "Hello", "metadata": {"step": 1, "finish_reason": "stop"}}]


def test_error_records_level_and_provider_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://llm.test"))
    body = {"message": "Invalid tool schema", "code": "invalid_request"}
    error = openai.BadRequestError("Invalid tool schema", response=response, body=body)

    with GenerationTrace("generate-response", model="m", input="hi") as trace:
        trace.error(error)

    (update,) = fake.generation.updates
    assert update["level"] == "ERROR"
    assert update["status_message"] == "BadRequestError: Invalid tool schema"
    assert update["output"] == {
        "error": "Invalid tool schema",
        "error_type": "BadRequestError",
        "status_code": 400,
        "provider_error": body,
    }


def test_caller_exception_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    with pytest.raises(ValueError, match="the real error"), GenerationTrace("g", model="m", input="x"):
        raise ValueError("the real error")


def test_start_failure_continues_untraced(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    install(monkeypatch, fail_on_start=True)
    with GenerationTrace("g", model="m", input="x") as trace:
        result = "the LLM's answer"
        trace.success(lambda: {"output": result})  # a no-op without a generation
    assert result == "the LLM's answer"
    assert "continuing untraced" in caplog.text


def test_failing_success_builder_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = install(monkeypatch)

    def unexpected_response_shape() -> dict[str, Any]:
        raise KeyError("usage")

    with GenerationTrace("g", model="m", input="x") as trace:
        trace.success(unexpected_response_shape)
    assert fake.generation.updates == []
    assert "Couldn't record Langfuse generation success" in caplog.text


def test_end_failure_is_logged_not_raised(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    install(monkeypatch, fail_on_exit=True)
    with caplog.at_level(logging.ERROR), GenerationTrace("g", model="m", input="x") as trace:
        trace.success(lambda: {"output": "ok"})
    assert "Couldn't end Langfuse generation" in caplog.text


# ---- masking ----

# Built at runtime so the test file itself holds no key-shaped strings.
OPENAI_KEY = "sk-proj-" + "A1b2" * 8
LANGFUSE_KEY = "sk-lf-" + "0a1b2c3d" * 4
GROQ_KEY = "gsk_" + "Ab3" * 10
GOOGLE_KEY = "AIza" + "B" * 35


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail me at priya.k+deals@example.co.in", "mail me at [EMAIL]"),
        (f"key={OPENAI_KEY}", "key=[API_KEY]"),
        (f"LANGFUSE_SECRET_KEY={LANGFUSE_KEY}", "LANGFUSE_SECRET_KEY=[LANGFUSE_KEY]"),
        (f"groq {GROQ_KEY}", "groq [GROQ_KEY]"),
        (f"gemini {GOOGLE_KEY}", "gemini [GOOGLE_KEY]"),
        ("Authorization: Bearer abcdefghijklmnop.qrstuv", "Authorization: Bearer [TOKEN]"),
        # Ordinary shopping text is left alone.
        ("boAt Airdopes 141 at ₹1,099, sk-type model, 4.1 / 5", "boAt Airdopes 141 at ₹1,099, sk-type model, 4.1 / 5"),
    ],
    ids=["email", "openai-key", "langfuse-key", "groq-key", "google-key", "bearer", "untouched"],
)
def test_redact(text: str, expected: str) -> None:
    assert _redact(text) == expected


def span(attributes: dict[str, Any], span_id: str = "0" * 16) -> tuple[OtelSpanIdentifier, OtelSpanData]:
    trace_id = "f" * 32
    identifier = OtelSpanIdentifier(trace_id=trace_id, span_id=span_id)
    data = OtelSpanData(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name="generate-response",
        instrumentation_scope_name="langfuse",
        instrumentation_scope_version=None,
        attributes=attributes,
        resource_attributes={},
    )
    return identifier, data


def test_mask_patches_only_attributes_that_change() -> None:
    dirty_id, dirty = span(
        {"langfuse.observation.input": "email me: a@b.co", "langfuse.observation.output": "Sure!", "tokens": 12},
        span_id="1" * 16,
    )
    clean_id, clean = span({"langfuse.observation.input": "earbuds under 3000"}, span_id="2" * 16)

    result = mask_otel_spans(params=MaskOtelSpansParams(spans={dirty_id: dirty, clean_id: clean}))

    assert result is not None
    assert set(result.span_patches) == {dirty_id}
    patch = result.span_patches[dirty_id]
    assert patch is not None
    assert patch.set_attributes == {"langfuse.observation.input": "email me: [EMAIL]"}


def test_mask_returns_none_when_nothing_to_redact() -> None:
    identifier, data = span({"langfuse.observation.input": "earbuds under 3000", "step": 1})
    assert mask_otel_spans(params=MaskOtelSpansParams(spans={identifier: data})) is None
