import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache

import openai
from openai import AsyncOpenAI, RateLimitError

from app.config import settings
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import PROVIDER_ITEMS_KEY, LLMResponse, ToolCall, ToolCallFunction
from app.tracing.generation import GenerationTrace

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
        trace_metadata: dict | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response.

        `messages` and `tools` are in OpenAI Chat Completions format; adapters
        for other APIs convert them. `name` labels the call's Langfuse
        generation. Keep it stable: evaluators and dashboards filter on it.
        `tool_choice="none"` forces a text answer even when tools are offered.
        `trace_metadata` (e.g. {"step": 3, "operation": "agent_step"}) is
        attached to the generation.

        Raises the provider-agnostic errors in app.llm.errors.
        """
        ...


class _OpenAISDKAdapter(LLMAdapter):
    """Shared by every adapter built on the OpenAI Python SDK: one client, the
    retry policy, the SDK-error to LLMError mapping, and generation tracing.
    Subclasses implement one API's request, call, and response mapping."""

    api_name: str

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

    # --- implemented per API ---

    @abstractmethod
    def _build_request(self, messages: list[dict], tools: list[dict] | None, tool_choice: str) -> dict: ...

    @abstractmethod
    async def _create(self, kwargs: dict): ...

    @abstractmethod
    def _trace_input(self, kwargs: dict): ...

    @abstractmethod
    def _model_parameters(self, kwargs: dict, service_tier: str | None = None) -> dict: ...

    @abstractmethod
    def _success_update(self, kwargs: dict, response) -> dict: ...

    @abstractmethod
    def _to_llm_response(self, response) -> LLMResponse: ...

    # --- shared ---

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
        trace_metadata: dict | None = None,
    ) -> LLMResponse:
        kwargs = self._build_request(messages, tools, tool_choice)

        # Retry once on transient failures; fail fast with a clear, provider-
        # agnostic error on ones retrying can't fix.
        # Pattern from Airtap's omniNormalizeProviderError
        provider = f"{settings.llm_provider} model {self.model!r}"
        max_retries = 1
        for attempt in range(max_retries + 1):
            last_attempt = attempt == max_retries
            try:
                response = await self._traced_call(kwargs, name=name, attempt=attempt + 1, trace_metadata=trace_metadata or {})
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

        return self._to_llm_response(response)

    async def _traced_call(self, kwargs: dict, *, name: str, attempt: int, trace_metadata: dict):
        """One API call = one Langfuse generation, opened before the call and
        completed after it succeeds or fails (every attempt, including a 429
        before the retry, is its own generation). Pattern from Airtap's omniTracing."""
        with GenerationTrace(
            name,
            model=kwargs["model"],
            input=self._trace_input(kwargs),
            model_parameters=self._model_parameters(kwargs),
            metadata={**trace_metadata, "attempt": attempt, "provider": settings.llm_provider, "api": self.api_name},
        ) as trace:
            try:
                response = await self._create(kwargs)
            except Exception as e:
                trace.error(e)
                raise
            # Langfuse prices the usage buckets with its model definition for
            # the response's model (built in for OpenAI models; added manually
            # for openai/gpt-oss-120b), which is where the cost comes from.
            trace.success(lambda: self._success_update(kwargs, response))
            return response


class ChatCompletionsAdapter(_OpenAISDKAdapter):
    """OpenAI Chat Completions API: Groq, and OpenAI with LLM_API=chat_completions."""

    api_name = "chat_completions"

    def _build_request(self, messages, tools, tool_choice):
        kwargs = {
            "model": self.model,
            # Drop adapter-private keys (e.g. Responses reasoning items) the API would reject.
            "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
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
        return kwargs

    async def _create(self, kwargs):
        return await self.client.chat.completions.create(**kwargs)

    def _trace_input(self, kwargs):
        """OpenAI chat format, the same shape Langfuse's own OpenAI integration logs,
        so the UI renders a conversation and tool definitions rather than raw JSON."""
        if kwargs.get("tools"):
            return {"messages": kwargs["messages"], "tools": kwargs["tools"]}
        return kwargs["messages"]

    def _model_parameters(self, kwargs, service_tier=None):
        return _pick_params(kwargs, ("temperature", "max_completion_tokens", "reasoning_effort", "tool_choice"), service_tier)

    def _success_update(self, kwargs, response):
        choice = response.choices[0]
        message = choice.message
        output = _assistant_output(
            message.content,
            [(tc.id, tc.function.name, tc.function.arguments) for tc in message.tool_calls or []],
            # Groq returns reasoning models' thinking as a non-OpenAI field.
            (message.model_extra or {}).get("reasoning"),
        )
        usage = response.usage
        return {
            "model": response.model,
            "output": output,
            "usage_details": _usage_buckets(
                input_total=usage.prompt_tokens,
                cached=_detail(usage, "prompt_tokens_details", "cached_tokens"),
                cache_writes=_detail(usage, "prompt_tokens_details", "cache_write_tokens"),
                output_total=usage.completion_tokens,
                reasoning=_detail(usage, "completion_tokens_details", "reasoning_tokens"),
                total=usage.total_tokens,
            ) if usage else None,
            # Langfuse picks the price tier (standard/flex/priority) from
            # service_tier, and only the response says which ran.
            "model_parameters": self._model_parameters(kwargs, getattr(response, "service_tier", None)),
            "metadata": {"finish_reason": choice.finish_reason},
        }

    def _to_llm_response(self, response):
        choice = response.choices[0]
        tool_calls = [
            ToolCall(id=tc.id, type=tc.type, function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments))
            for tc in choice.message.tool_calls or []
        ] or None
        usage = response.usage
        return LLMResponse(
            content=choice.message.content,
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


class ResponsesAdapter(_OpenAISDKAdapter):
    """OpenAI Responses API (LLM_API=responses): reasoning *and* function tools
    together, which Chat Completions doesn't allow on GPT-6, with reasoning
    summaries recorded on each generation. Structure follows Airtap's
    omniResponses.ts.

    Stateless (store=False): every call sends the whole turn. Reasoning items
    come back encrypted (include=["reasoning.encrypted_content"]) and must be
    passed back with the function calls they preceded, so the model keeps its
    chain of thought across tool calls; they travel on the assistant message
    under PROVIDER_ITEMS_KEY.
    """

    api_name = "responses"

    def _build_request(self, messages, tools, tool_choice):
        instructions, items = _messages_to_responses_input(messages)
        kwargs = {
            "model": self.model,
            "input": items,
            "store": False,  # nothing retained on OpenAI's side; we resend context
            "include": ["reasoning.encrypted_content"],
            # Counts reasoning tokens too; a response that runs out is "incomplete".
            "max_output_tokens": settings.llm_max_tokens,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        else:
            # Responses rejects `temperature` whenever reasoning is on (tested on gpt-6-luna).
            kwargs["temperature"] = settings.llm_temperature
        if tools:
            kwargs["tools"] = [_chat_tool_to_responses(t) for t in tools]
            kwargs["tool_choice"] = tool_choice
        return kwargs

    async def _create(self, kwargs):
        return await self.client.responses.create(**kwargs)

    def _trace_input(self, kwargs):
        """Instructions as a system message, then the input items: the same shape
        Langfuse's OpenAI integration logs for Responses calls."""
        messages = ([{"role": "system", "content": kwargs["instructions"]}] if kwargs.get("instructions") else []) + kwargs["input"]
        if kwargs.get("tools"):
            return {"messages": messages, "tools": kwargs["tools"]}
        return messages

    def _model_parameters(self, kwargs, service_tier=None):
        params = _pick_params(kwargs, ("temperature", "max_output_tokens", "tool_choice"), service_tier)
        if kwargs.get("reasoning"):
            params["reasoning_effort"] = kwargs["reasoning"]["effort"]
        return params

    def _success_update(self, kwargs, response):
        text, calls, reasoning = _parse_responses_output(response)
        usage = response.usage
        return {
            "model": response.model,
            # Chat-style assistant message so Langfuse renders tool calls and the
            # reasoning summary the same way for both APIs.
            "output": _assistant_output(text, [(c.call_id, c.name, c.arguments) for c in calls], reasoning),
            "usage_details": _usage_buckets(
                input_total=usage.input_tokens,
                cached=_detail(usage, "input_tokens_details", "cached_tokens"),
                cache_writes=_detail(usage, "input_tokens_details", "cache_write_tokens"),
                output_total=usage.output_tokens,
                reasoning=_detail(usage, "output_tokens_details", "reasoning_tokens"),
                total=usage.total_tokens,
            ) if usage else None,
            "model_parameters": self._model_parameters(kwargs, getattr(response, "service_tier", None)),
            "metadata": {"status": response.status, **_incomplete_reason(response)},
        }

    def _to_llm_response(self, response):
        text, calls, reasoning = _parse_responses_output(response)
        tool_calls = [
            ToolCall(id=c.call_id, function=ToolCallFunction(name=c.name, arguments=c.arguments))
            for c in calls
        ] or None
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"
        usage = response.usage
        return LLMResponse(
            content=text or None,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            model=response.model,
            # Everything but the final text message: reasoning items and function
            # calls, to be passed back verbatim on the next call of this turn.
            provider_items=[
                item.model_dump(exclude_none=True) for item in response.output
                if item.type in ("reasoning", "function_call")
            ] or None,
        )


# ---- Responses API conversions ----

def _chat_tool_to_responses(tool: dict) -> dict:
    """Chat Completions nests the definition under "function"; Responses is flat.
    strict=False keeps validation behaviour identical to Chat Completions: our
    Pydantic schemas have optional fields, and the registry validates anyway."""
    fn = tool["function"]
    return {"type": "function", "name": fn["name"], "description": fn.get("description", ""),
            "parameters": fn["parameters"], "strict": False}


def _messages_to_responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Chat-format history -> (instructions, input items).

    - The leading system message becomes `instructions`; later system messages
      (e.g. the step-limit note) stay in place as developer messages.
    - An assistant message carrying provider items replays them verbatim
      (reasoning + function calls); otherwise tool_calls become function_call items.
    - Tool results become function_call_output items.
    """
    instructions = None
    items: list[dict] = []
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            if i == 0:
                instructions = m["content"]
            else:
                items.append({"role": "developer", "content": m["content"]})
        elif role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            if m.get(PROVIDER_ITEMS_KEY):
                items.extend(m[PROVIDER_ITEMS_KEY])
            else:
                for tc in m.get("tool_calls") or []:
                    items.append({"type": "function_call", "call_id": tc["id"],
                                  "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]})
            if m.get("content") and not m.get("tool_calls"):
                items.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
    return instructions, items


def _parse_responses_output(response):
    """-> (text, function_call items, reasoning summary text)."""
    calls = [item for item in response.output if item.type == "function_call"]
    summaries = [
        part.text.strip()
        for item in response.output if item.type == "reasoning"
        for part in (item.summary or []) if part.text and part.text.strip()
    ]
    return response.output_text or "", calls, "\n\n".join(summaries) or None


def _incomplete_reason(response) -> dict:
    details = getattr(response, "incomplete_details", None)
    return {"incomplete_reason": details.reason} if details and getattr(details, "reason", None) else {}


# ---- shared helpers ----

def _pick_params(kwargs: dict, keys: tuple[str, ...], service_tier: str | None) -> dict:
    params = {k: kwargs[k] for k in keys if k in kwargs}
    if "reasoning_effort" in kwargs:
        params["reasoning_effort"] = kwargs["reasoning_effort"]
    if service_tier:
        params["service_tier"] = service_tier
    return params


def _assistant_output(content: str | None, calls: list[tuple[str, str, str]], reasoning: str | None) -> dict:
    output = {"role": "assistant", "content": content}
    if calls:
        output["tool_calls"] = [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            for call_id, name, arguments in calls
        ]
    # The model's thinking: why it answered or picked a tool. Langfuse renders a
    # message-level reasoning field in its formatted view.
    if reasoning:
        output["reasoning"] = reasoning
    return output


def _detail(usage, details_field: str, key: str) -> int:
    details = getattr(usage, details_field, None)
    return (getattr(details, key, None) or 0) if details else 0


def _usage_buckets(*, input_total: int, cached: int, cache_writes: int, output_total: int, reasoning: int, total: int) -> dict[str, int]:
    """Provider counts are inclusive (input includes cache reads and writes,
    output includes reasoning tokens). Langfuse wants mutually exclusive
    buckets, so split them out before sending. Bucket names match the price
    keys in Langfuse's model definitions (GPT-6 bills cache writes separately)."""
    details = {
        "input": input_total - cached - cache_writes,
        "output": output_total - reasoning,
        "total": total,
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


# Backwards-compatible name for the Chat Completions adapter.
OpenAICompatibleAdapter = ChatCompletionsAdapter


@lru_cache(maxsize=1)
def get_llm_adapter() -> LLMAdapter:
    """Factory function. LLM_PROVIDER picks the provider and LLM_API picks the
    OpenAI API (chat_completions or responses); a provider with a different SDK
    needs its own LLMAdapter subclass.

    Cached so every caller shares one HTTP client and connection pool, instead
    of opening a new one per agent turn and per suggestion call.
    """
    return ResponsesAdapter() if settings.llm_api == "responses" else ChatCompletionsAdapter()


if __name__ == "__main__":
    # Connection smoke test. From the project root:  python -m app.llm.adapter
    import sys

    # Windows consoles default to cp1252 and crash on the model's emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def _smoke_test() -> None:
        llm = get_llm_adapter()
        response = await llm.chat([{"role": "user", "content": "Hello, what can you do?"}])
        print(f"api:           {type(llm).__name__}")
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
