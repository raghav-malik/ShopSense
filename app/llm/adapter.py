import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache

import openai
from openai import AsyncOpenAI, RateLimitError

from app.config import settings
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import LLMResponse, ToolCall, ToolCallFunction
from app.tracing.langfuse_setup import get_langfuse

langfuse = get_langfuse()

# A per-minute limit (e.g. Groq free tier: 8K tokens/min) can ask for up to ~60s,
# but daily token limits can ask for many minutes. Past this cap we fail fast
# rather than hang the request.
MAX_RETRY_WAIT_SECONDS = 60.0
DEFAULT_RETRY_WAIT_SECONDS = 5.0
# Transient failures (timeouts, dropped connections, provider 5xx) get one quick retry.
TRANSIENT_RETRY_WAIT_SECONDS = 2.0


class LLMAdapter(ABC):
    """Base class for LLM providers. Subclass and implement chat()."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Send messages to the LLM and get a response.

        `name` labels the call's Langfuse generation. Keep it stable: evaluators
        and dashboards filter on it. `tool_choice="none"` forces a text answer
        even when tools are offered.

        Raises the provider-agnostic errors in app.llm.errors.
        """
        ...


class OpenAICompatibleAdapter(LLMAdapter):
    """Any provider that speaks the OpenAI Chat Completions API: Groq and OpenAI
    today, picked with LLM_PROVIDER. Includes 429 retry."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            # The SDK retries 429s twice on its own by default; disable that so
            # the retry-once policy below is the only one.
            max_retries=0,
            # The SDK's default read timeout is 600s; with no SDK retries, one
            # hung call would hold the request for 10 minutes.
            timeout=settings.llm_timeout,
        )
        self.model = settings.llm_model
        self.reasoning_effort = settings.llm_reasoning_effort

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            # OpenAI's current models reject the older `max_tokens`; Groq accepts both.
            "max_completion_tokens": settings.llm_max_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        # OpenAI reasoning models only accept a custom temperature with
        # reasoning disabled ("Only the default (1) value is supported").
        if self.reasoning_effort in (None, "none"):
            kwargs["temperature"] = settings.llm_temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # Retry once on transient failures; fail fast with a clear, provider-
        # agnostic error on ones retrying can't fix.
        # Pattern from Airtap's omniNormalizeProviderError
        provider = f"{settings.llm_provider} model {self.model!r}"
        max_retries = 1
        for attempt in range(max_retries + 1):
            last_attempt = attempt == max_retries
            try:
                response = await self._traced_completion(kwargs, name=name, attempt=attempt + 1)
                break
            except RateLimitError as e:
                wait = _retry_after_seconds(e)
                if last_attempt or wait > MAX_RETRY_WAIT_SECONDS:
                    raise LLMRateLimitError(f"Rate limited by {provider}; retry after ~{wait:.0f}s", retry_after=wait) from e
            except openai.APITimeoutError as e:  # subclass of APIConnectionError: check first
                if last_attempt:
                    raise LLMTimeoutError(f"{provider} didn't respond within {settings.llm_timeout:g}s") from e
                wait = TRANSIENT_RETRY_WAIT_SECONDS
            except openai.APIConnectionError as e:
                if last_attempt:
                    raise LLMUnavailableError(f"Couldn't connect to {settings.llm_base_url}") from e
                wait = TRANSIENT_RETRY_WAIT_SECONDS
            except openai.InternalServerError as e:  # 5xx, incl. 503 "overloaded"
                if last_attempt:
                    raise LLMUnavailableError(f"{provider} is unavailable (HTTP {e.status_code})") from e
                wait = TRANSIENT_RETRY_WAIT_SECONDS
            except openai.NotFoundError as e:
                # e.g. Groq retiring llama-3.3-70b-versatile: a config problem, not transient.
                raise LLMUnavailableError(f"{provider} doesn't exist or this key can't use it; check LLM_MODEL") from e
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                raise LLMUnavailableError(f"{settings.llm_provider} rejected the API key (HTTP {e.status_code}); check {settings.llm_provider.upper()}_API_KEY") from e
            except openai.APIStatusError as e:  # 400/413/422...: the request itself was rejected
                raise LLMError(f"{provider} rejected the request (HTTP {e.status_code}): {e.message}") from e
            await asyncio.sleep(wait)

        choice = response.choices[0]

        # Parse tool calls if present
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in choice.message.tool_calls
            ]

        usage = response.usage
        return LLMResponse(
            content=choice.message.content,
            # Groq returns reasoning models' thinking as a non-OpenAI field.
            reasoning=(choice.message.model_extra or {}).get("reasoning"),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            model=response.model,
        )

    async def _traced_completion(self, kwargs: dict, *, name: str, attempt: int):
        """One API call = one Langfuse generation.

        Usage and cost are recorded in `finally` so every attempt is logged,
        including one that fails (e.g. a 429 before the retry).
        Pattern from Airtap's omniGenerate() finally block.
        """
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=name,
            model=kwargs["model"],
            input=_generation_input(kwargs),
            model_parameters=_model_parameters(kwargs),
            metadata={"attempt": attempt, "provider": settings.llm_provider},
        ) as generation:
            response = None
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                generation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
                raise
            finally:
                if response is not None:
                    choice = response.choices[0]
                    # Langfuse prices these buckets with its model definition for
                    # response.model (built in for OpenAI models; added manually
                    # for openai/gpt-oss-120b), which is where the cost comes from.
                    generation.update(
                        model=response.model,
                        output=_generation_output(choice.message),
                        usage_details=_usage_details(response.usage),
                        # Langfuse picks the price tier (standard/flex/priority)
                        # from service_tier, and only the response says which ran.
                        model_parameters=_model_parameters(kwargs, getattr(response, "service_tier", None)),
                        metadata={"attempt": attempt, "provider": settings.llm_provider, "finish_reason": choice.finish_reason},
                    )


def _model_parameters(kwargs: dict, service_tier: str | None = None) -> dict:
    params = {
        k: kwargs[k]
        for k in ("temperature", "max_completion_tokens", "reasoning_effort", "tool_choice")
        if k in kwargs
    }
    if service_tier:
        params["service_tier"] = service_tier
    return params


def _generation_input(kwargs: dict):
    """OpenAI chat format, the same shape Langfuse's own OpenAI integration logs,
    so the UI renders a conversation and tool definitions rather than raw JSON."""
    if kwargs.get("tools"):
        return {"messages": kwargs["messages"], "tools": kwargs["tools"]}
    return kwargs["messages"]


def _generation_output(message) -> dict:
    output = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        output["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls
        ]
    # The model's thinking: why it answered or picked a tool. Langfuse renders a
    # message-level reasoning field in its formatted view.
    reasoning = (message.model_extra or {}).get("reasoning")
    if reasoning:
        output["reasoning"] = reasoning
    return output


def _usage_details(usage) -> dict[str, int] | None:
    """Groq/OpenAI counts are inclusive (prompt_tokens includes cache reads and
    writes, completion_tokens includes reasoning tokens). Langfuse wants mutually
    exclusive buckets, so split them out before sending. Bucket names match the
    price keys in Langfuse's model definitions (GPT-6 bills cache writes separately)."""
    if usage is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    cached = (getattr(prompt_details, "cached_tokens", None) or 0) if prompt_details else 0
    cache_writes = (getattr(prompt_details, "cache_write_tokens", None) or 0) if prompt_details else 0
    reasoning = (getattr(completion_details, "reasoning_tokens", None) or 0) if completion_details else 0
    details = {
        "input": usage.prompt_tokens - cached - cache_writes,
        "output": usage.completion_tokens - reasoning,
        "total": usage.total_tokens,
    }
    if cached:
        details["input_cached_tokens"] = cached
    if cache_writes:
        details["cache_write_tokens"] = cache_writes
    if reasoning:
        details["output_reasoning_tokens"] = reasoning
    return details


def _retry_after_seconds(error: RateLimitError) -> float:
    """Seconds to wait, from the 429's retry-after header (Groq and OpenAI send seconds)."""
    value = error.response.headers.get("retry-after")
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_RETRY_WAIT_SECONDS


@lru_cache(maxsize=1)
def get_llm_adapter() -> LLMAdapter:
    """Factory function. Set LLM_PROVIDER to swap between OpenAI-compatible
    providers; a provider with a different API needs its own LLMAdapter subclass.

    Cached so every caller shares one HTTP client and connection pool, instead
    of opening a new one per agent turn and per suggestion call.
    """
    return OpenAICompatibleAdapter()


if __name__ == "__main__":
    # Connection smoke test. From the project root:  python -m app.llm.adapter
    import sys

    # Windows consoles default to cp1252 and crash on the model's emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def _smoke_test() -> None:
        llm = get_llm_adapter()
        response = await llm.chat([{"role": "user", "content": "Hello, what can you do?"}])
        print(f"model:         {response.model}")
        print(f"finish_reason: {response.finish_reason}")
        print(f"usage:         {response.usage}")
        print(f"reasoning:     {(response.reasoning or '')[:200]!r}")
        print(f"\n{response.content}")

    from app.tracing.langfuse_setup import shutdown_langfuse

    try:
        asyncio.run(_smoke_test())
    finally:
        shutdown_langfuse()  # send the generation before the script exits
