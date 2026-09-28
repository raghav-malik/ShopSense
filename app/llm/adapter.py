import asyncio
from abc import ABC, abstractmethod

from openai import AsyncOpenAI, RateLimitError

from app.config import settings
from app.llm.types import LLMResponse, ToolCall, ToolCallFunction

# Groq free tier: per-minute limits reset in seconds, but daily token limits can
# ask for a wait of many minutes. Past this cap we fail fast rather than hang
# the request.
MAX_RETRY_WAIT_SECONDS = 30.0
DEFAULT_RETRY_WAIT_SECONDS = 5.0


class LLMAdapter(ABC):
    """Base class for LLM providers. Subclass and implement chat()."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response."""
        ...


class GroqAdapter(LLMAdapter):
    """Groq LLM adapter using OpenAI-compatible API. Includes 429 retry."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.llm_base_url,
            # The SDK retries 429s twice on its own by default; disable that so
            # the retry-once policy below is the only one.
            max_retries=0,
        )
        self.model = settings.llm_model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Retry once on 429 (Groq free tier rate limit)
        # Pattern from Airtap's omniNormalizeProviderError
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError as e:
                wait = _retry_after_seconds(e)
                if attempt == max_retries or wait > MAX_RETRY_WAIT_SECONDS:
                    raise
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


def _retry_after_seconds(error: RateLimitError) -> float:
    """Seconds to wait, from the 429's retry-after header (Groq sends seconds)."""
    value = error.response.headers.get("retry-after")
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_RETRY_WAIT_SECONDS


def get_llm_adapter() -> LLMAdapter:
    """Factory function. Change this to swap providers."""
    return GroqAdapter()


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

    asyncio.run(_smoke_test())
