"""The adapter's retry policy and SDK-error mapping, offline: `_create` is
replaced by a script of responses and exceptions, and asyncio.sleep records
the waits instead of sleeping."""

import asyncio
from collections.abc import Callable

import httpx2  # the OpenAI SDK's HTTP client; its exceptions are built from these types
import openai
import pytest
from openai.types.chat import ChatCompletion

from app.llm.adapter import (
    DEFAULT_RETRY_WAIT_SECONDS,
    MAX_RETRY_WAIT_SECONDS,
    TRANSIENT_RETRY_WAIT_SECONDS,
    ChatCompletionsAdapter,
)
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import JSONObject

REQUEST = httpx2.Request("POST", "https://llm.test/v1/chat/completions")

COMPLETION = ChatCompletion.model_validate(
    {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi!"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
)


def status_error[E: openai.APIStatusError](
    cls: type[E], status: int, headers: dict[str, str] | None = None, body: object = None
) -> E:
    response = httpx2.Response(status, request=REQUEST, headers=headers)
    return cls(f"HTTP {status}", response=response, body=body)


def rate_limited(retry_after: str | None = None) -> openai.RateLimitError:
    return status_error(openai.RateLimitError, 429, {"retry-after": retry_after} if retry_after else None)


def timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=REQUEST)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=REQUEST)


def overloaded() -> openai.InternalServerError:
    return status_error(openai.InternalServerError, 503)


class Scripted:
    """Stands in for the SDK call: returns or raises the scripted items in order."""

    def __init__(self, *script: ChatCompletion | Exception) -> None:
        self.script = list(script)
        self.calls = 0

    async def __call__(self, kwargs: JSONObject) -> ChatCompletion:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The retry waits the adapter asked for; nothing actually sleeps."""
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def record(seconds: float) -> None:
        waits.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", record)
    return waits


def adapter_with(
    monkeypatch: pytest.MonkeyPatch, *script: ChatCompletion | Exception
) -> tuple[ChatCompletionsAdapter, Scripted]:
    adapter = ChatCompletionsAdapter()
    scripted = Scripted(*script)
    monkeypatch.setattr(adapter, "_create", scripted)
    return adapter, scripted


# ---- one retry, then success ----


@pytest.mark.parametrize(
    ("first_failure", "expected_wait"),
    [
        (overloaded, TRANSIENT_RETRY_WAIT_SECONDS),
        (timeout, TRANSIENT_RETRY_WAIT_SECONDS),
        (connection_error, TRANSIENT_RETRY_WAIT_SECONDS),
        (lambda: rate_limited("3"), 3.0),  # the provider's retry-after is honoured
        (lambda: rate_limited(), DEFAULT_RETRY_WAIT_SECONDS),
        (lambda: rate_limited("Wed, 21 Oct 2026 07:28:00 GMT"), DEFAULT_RETRY_WAIT_SECONDS),  # HTTP-date form
    ],
    ids=["503", "timeout", "connection", "429-retry-after", "429-no-header", "429-http-date"],
)
async def test_transient_failure_is_retried_once(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    first_failure: Callable[[], Exception],
    expected_wait: float,
) -> None:
    adapter, scripted = adapter_with(monkeypatch, first_failure(), COMPLETION)
    response = await adapter.chat([{"role": "user", "content": "hi"}])
    assert response.content == "Hi!"
    assert scripted.calls == 2
    assert sleeps == [expected_wait]


# ---- failures mapped to provider-agnostic errors ----


@pytest.mark.parametrize(
    ("script", "expected", "calls"),
    [
        ((timeout(), timeout()), LLMTimeoutError, 2),
        ((connection_error(), connection_error()), LLMUnavailableError, 2),
        ((overloaded(), overloaded()), LLMUnavailableError, 2),
        ((rate_limited("1"), rate_limited("1")), LLMRateLimitError, 2),
        # Not transient: retrying can't fix these, so there's no second call.
        ((status_error(openai.NotFoundError, 404),), LLMUnavailableError, 1),
        ((status_error(openai.AuthenticationError, 401),), LLMUnavailableError, 1),
        ((status_error(openai.PermissionDeniedError, 403),), LLMUnavailableError, 1),
        ((status_error(openai.BadRequestError, 400),), LLMError, 1),
    ],
    ids=["timeout", "connection", "503", "429", "404-model", "401-key", "403-key", "400-request"],
)
async def test_failures_map_to_llm_errors(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    script: tuple[Exception, ...],
    expected: type[LLMError],
    calls: int,
) -> None:
    adapter, scripted = adapter_with(monkeypatch, *script)
    with pytest.raises(LLMError) as raised:
        await adapter.chat([{"role": "user", "content": "hi"}])
    assert type(raised.value) is expected  # exact class: the API maps each to its own status
    assert isinstance(raised.value.__cause__, openai.APIError)  # SDK error kept for logs and traces
    assert scripted.calls == calls
    assert len(sleeps) == calls - 1


async def test_long_rate_limit_wait_fails_fast(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    # A daily-limit 429 can ask for minutes; holding the request open that long is worse than failing.
    wait = MAX_RETRY_WAIT_SECONDS + 1
    adapter, scripted = adapter_with(monkeypatch, rate_limited(str(wait)))
    with pytest.raises(LLMRateLimitError) as raised:
        await adapter.chat([{"role": "user", "content": "hi"}])
    assert raised.value.retry_after == wait  # the API turns this into a Retry-After header
    assert scripted.calls == 1 and sleeps == []


async def test_unexpected_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    adapter, scripted = adapter_with(monkeypatch, RuntimeError("bug in our code"))
    with pytest.raises(RuntimeError, match="bug in our code"):
        await adapter.chat([{"role": "user", "content": "hi"}])
    assert scripted.calls == 1 and sleeps == []
