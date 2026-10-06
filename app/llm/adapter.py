"""LLM adapters: one interface over OpenAI Chat Completions, the OpenAI Responses API, Groq and Gemini,
with one retry policy, provider-agnostic errors, and a Langfuse generation per call."""

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import cast, override

import openai
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.responses import Response, ResponseFunctionToolCall, ResponseReasoningItem

from app.config import PROVIDER_DEFAULTS, settings
from app.llm.errors import LLMError, LLMRateLimitError, LLMTimeoutError, LLMUnavailableError
from app.llm.types import (
    PROVIDER_ITEMS_KEY,
    ChatMessage,
    ChatToolCall,
    JSONObject,
    LLMResponse,
    ToolCall,
    ToolCallFunction,
)
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

    def describe(self) -> dict[str, str]:
        """How this adapter calls its model (model, provider, api, reasoning_effort), for traces."""
        return {}

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[JSONObject] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
        trace_metadata: JSONObject | None = None,
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


class _OpenAISDKAdapter[ResponseT](LLMAdapter):
    """Shared by every adapter built on the OpenAI Python SDK: one client, the
    retry policy, the SDK-error to LLMError mapping, and generation tracing.
    Subclasses implement one API's request, call, and response mapping;
    ResponseT is that API's SDK response type."""

    api_name: str

    def __init__(self, *, model: str | None = None, reasoning_effort: str | None = None) -> None:
        """`model` and `reasoning_effort` default to the configured agent model."""
        api_key = settings.active_api_key  # checked at startup for the configured provider
        self.client = AsyncOpenAI(
            api_key=api_key.get_secret_value() if api_key else None,
            base_url=settings.llm_base_url,
            # The SDK retries 429s twice on its own by default; disable that so
            # the retry-once policy below is the only one.
            max_retries=0,
            # The SDK's default read timeout is 600s; with no SDK retries, one
            # hung call would hold the request for 10 minutes.
            timeout=settings.llm_timeout,
        )
        self.model = model or settings.llm_model
        self.reasoning_effort = reasoning_effort or settings.llm_reasoning_effort

    @override
    def describe(self) -> dict[str, str]:
        details = {
            "model": self.model,
            "provider": settings.llm_provider,
            "api": self.api_name,
            "reasoning_effort": self.reasoning_effort,
        }
        return {key: value for key, value in details.items() if value}

    # --- implemented per API ---

    @abstractmethod
    def _build_request(
        self, messages: list[ChatMessage], tools: list[JSONObject] | None, tool_choice: str
    ) -> JSONObject: ...

    @abstractmethod
    async def _create(self, kwargs: JSONObject) -> ResponseT: ...

    @abstractmethod
    def _trace_input(self, kwargs: JSONObject) -> object: ...

    @abstractmethod
    def _model_parameters(self, kwargs: JSONObject, service_tier: str | None = None) -> JSONObject: ...

    @abstractmethod
    def _success_update(self, kwargs: JSONObject, response: ResponseT) -> JSONObject: ...

    @abstractmethod
    def _to_llm_response(self, response: ResponseT) -> LLMResponse: ...

    # --- shared ---

    @override
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[JSONObject] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
        trace_metadata: JSONObject | None = None,
    ) -> LLMResponse:
        kwargs = self._build_request(messages, tools, tool_choice)

        # Retry once on transient failures; fail fast with a clear, provider-
        # agnostic error on ones retrying can't fix.
        provider = f"{settings.llm_provider} model {self.model!r}"
        max_retries = 1
        for attempt in range(max_retries + 1):
            last_attempt = attempt == max_retries
            try:
                response = await self._traced_call(
                    kwargs, name=name, attempt=attempt + 1, trace_metadata=trace_metadata or {}
                )
                return self._to_llm_response(response)
            except RateLimitError as e:
                wait = _retry_after_seconds(e)
                if last_attempt or wait > MAX_RETRY_WAIT_SECONDS:
                    raise LLMRateLimitError(
                        f"Rate limited by {provider}; retry after ~{wait:.0f}s", retry_after=wait
                    ) from e
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
                raise LLMUnavailableError(
                    f"{settings.llm_provider} rejected the API key (HTTP {e.status_code}); "
                    f"check {settings.llm_provider.upper()}_API_KEY"
                ) from e
            except openai.APIStatusError as e:  # 400/413/422...: the request itself was rejected
                raise LLMError(f"{provider} rejected the request (HTTP {e.status_code}): {e.message}") from e
            await asyncio.sleep(wait)
        raise AssertionError("unreachable: the last attempt returns or raises")

    async def _traced_call(
        self, kwargs: JSONObject, *, name: str, attempt: int, trace_metadata: JSONObject
    ) -> ResponseT:
        """One API call = one Langfuse generation, opened before the call and
        completed after it succeeds or fails (every attempt, including a 429
        before the retry, is its own generation)."""
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


class ChatCompletionsAdapter(_OpenAISDKAdapter[ChatCompletion]):
    """OpenAI Chat Completions API: Groq, and OpenAI with LLM_API=chat_completions."""

    api_name = "chat_completions"

    @override
    def _build_request(
        self, messages: list[ChatMessage], tools: list[JSONObject] | None, tool_choice: str
    ) -> JSONObject:
        kwargs: JSONObject = {
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

    @override
    async def _create(self, kwargs: JSONObject) -> ChatCompletion:
        return await self.client.chat.completions.create(**kwargs)

    @override
    def _trace_input(self, kwargs: JSONObject) -> object:
        """OpenAI chat format, the same shape Langfuse's own OpenAI integration logs,
        so the UI renders a conversation and tool definitions rather than raw JSON."""
        if kwargs.get("tools"):
            return {"messages": kwargs["messages"], "tools": kwargs["tools"]}
        return kwargs["messages"]

    @override
    def _model_parameters(self, kwargs: JSONObject, service_tier: str | None = None) -> JSONObject:
        return _pick_params(
            kwargs, ("temperature", "max_completion_tokens", "reasoning_effort", "tool_choice"), service_tier
        )

    @override
    def _success_update(self, kwargs: JSONObject, response: ChatCompletion) -> JSONObject:
        choice = response.choices[0]
        message = choice.message
        output = _assistant_output(
            message.content,
            [(tc.id, tc.function.name, tc.function.arguments) for tc in _function_calls(message)],
            _extra_reasoning(message),
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
            )
            if usage
            else None,
            # Langfuse picks the price tier (standard/flex/priority) from
            # service_tier, and only the response says which ran.
            "model_parameters": self._model_parameters(kwargs, response.service_tier),
            "metadata": {"finish_reason": choice.finish_reason},
        }

    @override
    def _to_llm_response(self, response: ChatCompletion) -> LLMResponse:
        choice = response.choices[0]
        tool_calls = [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments),
            )
            for tc in _function_calls(choice.message)
        ] or None
        usage = response.usage
        return LLMResponse(
            content=choice.message.content,
            reasoning=_extra_reasoning(choice.message),
            tool_calls=tool_calls,
            # Typed as always present, but OpenAI-compatible providers can omit it.
            finish_reason=cast(str | None, choice.finish_reason) or "stop",
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                # Billed at a lower rate; the agent's cost estimate needs it.
                "cached_tokens": _detail(usage, "prompt_tokens_details", "cached_tokens"),
            },
            model=response.model,
        )


class GeminiAdapter(ChatCompletionsAdapter):
    """Google Gemini through its OpenAI-compatible endpoint (LLM_PROVIDER=gemini).

    Same request/response shape as Chat Completions, with two Gemini 3 rules:
    - temperature is left at the model default: Google "strongly recommends"
      1.0 for all Gemini 3 models; lower values can cause looping.
    - thought signatures: Gemini 3 attaches opaque extra data to each tool call
      (its encrypted reasoning) that must come back unchanged on the next
      request. The raw tool calls travel on the assistant message under
      PROVIDER_ITEMS_KEY (as with the Responses API) and are replayed verbatim.
    """

    @override
    def _build_request(
        self, messages: list[ChatMessage], tools: list[JSONObject] | None, tool_choice: str
    ) -> JSONObject:
        replayed: list[ChatMessage] = []
        for m in messages:
            if m["role"] == "assistant" and m.get(PROVIDER_ITEMS_KEY):
                m = m.copy()
                # Same shape as ChatToolCall plus Gemini's `extra_content`.
                m["tool_calls"] = cast(list[ChatToolCall], m[PROVIDER_ITEMS_KEY])
            replayed.append(m)
        kwargs = super()._build_request(replayed, tools, tool_choice)
        kwargs.pop("temperature", None)
        return kwargs

    @override
    def _to_llm_response(self, response: ChatCompletion) -> LLMResponse:
        result = super()._to_llm_response(response)
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            # Full tool calls including any provider extras (the thought signature).
            result.provider_items = [tc.model_dump(exclude_none=True) for tc in tool_calls]
        return result

    @override
    def _success_update(self, kwargs: JSONObject, response: ChatCompletion) -> JSONObject:
        """Gemini's OpenAI endpoint leaves thinking tokens out of completion_tokens
        (and completion_tokens_details is null); they only show up in total_tokens.
        Measured: 18 prompt + 11 completion but total 227. Google bills thinking
        as output, so put the difference in the reasoning bucket or Langfuse would
        under-report cost."""
        update = super()._success_update(kwargs, response)
        usage = response.usage
        hidden_thinking = usage.total_tokens - usage.prompt_tokens - usage.completion_tokens if usage else 0
        buckets: dict[str, int] | None = update.get("usage_details")
        if hidden_thinking > 0 and buckets:
            buckets["output_reasoning_tokens"] = buckets.get("output_reasoning_tokens", 0) + hidden_thinking
        return update


class OllamaAdapter(ChatCompletionsAdapter):
    """Ollama Cloud through its OpenAI-compatible endpoint (LLM_PROVIDER=ollama).

    Ollama's Chat Completions API differs from OpenAI's in two ways that
    ShopSense hits (docs.ollama.com/api/openai-compatibility, October 2026):
    - `tool_choice` isn't supported. "auto" is the default anyway; to force a
      text answer (tool_choice="none", at a turn limit) the tools are left out.
    - The output cap is `max_tokens`, not `max_completion_tokens`.
    """

    @override
    def _build_request(
        self, messages: list[ChatMessage], tools: list[JSONObject] | None, tool_choice: str
    ) -> JSONObject:
        kwargs = super()._build_request(messages, tools if tool_choice != "none" else None, tool_choice)
        kwargs.pop("tool_choice", None)
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return kwargs

    @override
    def _model_parameters(self, kwargs: JSONObject, service_tier: str | None = None) -> JSONObject:
        return _pick_params(kwargs, ("temperature", "max_tokens", "reasoning_effort"), service_tier)


class ResponsesAdapter(_OpenAISDKAdapter[Response]):
    """OpenAI Responses API (LLM_API=responses): reasoning *and* function tools
    together, which Chat Completions doesn't allow on GPT-6, with reasoning
    summaries recorded on each generation.

    Stateless (store=False): every call sends the whole turn. Reasoning items
    come back encrypted (include=["reasoning.encrypted_content"]) and must be
    passed back with the function calls they preceded, so the model keeps its
    chain of thought across tool calls; they travel on the assistant message
    under PROVIDER_ITEMS_KEY.
    """

    api_name = "responses"

    @override
    def _build_request(
        self, messages: list[ChatMessage], tools: list[JSONObject] | None, tool_choice: str
    ) -> JSONObject:
        instructions, items = _messages_to_responses_input(messages)
        kwargs: JSONObject = {
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

    @override
    async def _create(self, kwargs: JSONObject) -> Response:
        return await self.client.responses.create(**kwargs)

    @override
    def _trace_input(self, kwargs: JSONObject) -> object:
        """Instructions as a system message, then the input items: the same shape
        Langfuse's OpenAI integration logs for Responses calls."""
        system = [{"role": "system", "content": kwargs["instructions"]}] if kwargs.get("instructions") else []
        messages = [*system, *kwargs["input"]]
        if kwargs.get("tools"):
            return {"messages": messages, "tools": kwargs["tools"]}
        return messages

    @override
    def _model_parameters(self, kwargs: JSONObject, service_tier: str | None = None) -> JSONObject:
        params = _pick_params(kwargs, ("temperature", "max_output_tokens", "tool_choice"), service_tier)
        if kwargs.get("reasoning"):
            params["reasoning_effort"] = kwargs["reasoning"]["effort"]
        return params

    @override
    def _success_update(self, kwargs: JSONObject, response: Response) -> JSONObject:
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
            )
            if usage
            else None,
            "model_parameters": self._model_parameters(kwargs, response.service_tier),
            "metadata": {"status": response.status, **_incomplete_reason(response)},
        }

    @override
    def _to_llm_response(self, response: Response) -> LLMResponse:
        text, calls, reasoning = _parse_responses_output(response)
        tool_calls = [
            ToolCall(id=c.call_id, function=ToolCallFunction(name=c.name, arguments=c.arguments)) for c in calls
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
                "cached_tokens": _detail(usage, "input_tokens_details", "cached_tokens"),
            },
            model=response.model,
            # Everything but the final text message: reasoning items and function
            # calls, to be passed back verbatim on the next call of this turn.
            provider_items=[
                item.model_dump(exclude_none=True)
                for item in response.output
                if isinstance(item, ResponseReasoningItem | ResponseFunctionToolCall)
            ]
            or None,
        )


# ---- Chat Completions helpers ----


def _function_calls(message: ChatCompletionMessage) -> list[ChatCompletionMessageFunctionToolCall]:
    """Function tool calls only. The SDK types tool_calls as function | custom;
    we only ever register function tools."""
    return [tc for tc in message.tool_calls or [] if isinstance(tc, ChatCompletionMessageFunctionToolCall)]


def _extra_reasoning(message: ChatCompletionMessage) -> str | None:
    """Groq returns reasoning models' thinking as a non-OpenAI `reasoning` field."""
    reasoning = (message.model_extra or {}).get("reasoning")
    return reasoning if isinstance(reasoning, str) else None


# ---- Responses API conversions ----


def _chat_tool_to_responses(tool: JSONObject) -> JSONObject:
    """Chat Completions nests the definition under "function"; Responses is flat.
    strict=False keeps validation behaviour identical to Chat Completions: our
    Pydantic schemas have optional fields, and the registry validates anyway."""
    fn = tool["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn["parameters"],
        "strict": False,
    }


def _messages_to_responses_input(messages: list[ChatMessage]) -> tuple[str | None, list[JSONObject]]:
    """Chat-format history -> (instructions, input items).

    - The leading system message becomes `instructions`; later system messages
      (e.g. the step-limit note) stay in place as developer messages.
    - An assistant message carrying provider items replays them verbatim
      (reasoning + function calls); otherwise tool_calls become function_call items.
    - Tool results become function_call_output items.
    """
    instructions: str | None = None
    items: list[JSONObject] = []
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
            if provider_items := m.get(PROVIDER_ITEMS_KEY):
                items.extend(provider_items)
            else:
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in m.get("tool_calls", [])
                )
            if m["content"] and not m.get("tool_calls"):
                items.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"), "output": m["content"]})
    return instructions, items


def _parse_responses_output(response: Response) -> tuple[str, list[ResponseFunctionToolCall], str | None]:
    """-> (text, function_call items, reasoning summary text)."""
    calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
    summaries = [
        part.text.strip()
        for item in response.output
        if isinstance(item, ResponseReasoningItem)
        for part in item.summary
        if part.text.strip()
    ]
    return response.output_text or "", calls, "\n\n".join(summaries) or None


def _incomplete_reason(response: Response) -> JSONObject:
    details = response.incomplete_details
    return {"incomplete_reason": details.reason} if details and details.reason else {}


# ---- shared helpers ----


def _pick_params(kwargs: JSONObject, keys: tuple[str, ...], service_tier: str | None) -> JSONObject:
    params = {k: kwargs[k] for k in keys if k in kwargs}
    if service_tier:
        params["service_tier"] = service_tier
    return params


def _assistant_output(content: str | None, calls: list[tuple[str, str, str]], reasoning: str | None) -> JSONObject:
    output: JSONObject = {"role": "assistant", "content": content}
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


def _detail(usage: object, details_field: str, key: str) -> int:
    """A token count from an optional *_details object (absent on some providers)."""
    details = getattr(usage, details_field, None)
    value = getattr(details, key, None) if details is not None else None
    return value if isinstance(value, int) else 0


def _usage_buckets(
    *, input_total: int, cached: int, cache_writes: int, output_total: int, reasoning: int, total: int
) -> dict[str, int]:
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
    if value is None:
        return DEFAULT_RETRY_WAIT_SECONDS
    try:
        return max(float(value), 0.0)
    except ValueError:  # an HTTP-date instead of seconds
        return DEFAULT_RETRY_WAIT_SECONDS


# Names from the spec: the Groq adapter is the plain Chat Completions adapter
# (OpenAI uses it too); OpenAICompatibleAdapter is its earlier name here.
GroqAdapter = OpenAICompatibleAdapter = ChatCompletionsAdapter


@lru_cache(maxsize=1)
def get_llm_adapter() -> LLMAdapter:
    """Factory function. LLM_PROVIDER picks the provider (openai, groq, gemini, ollama)
    and LLM_API picks OpenAI's API (chat_completions or responses); a provider
    with a different SDK needs its own LLMAdapter subclass.

    Cached so every caller shares one HTTP client and connection pool, instead
    of opening a new one per agent turn and per suggestion call.
    """
    if settings.llm_provider == "gemini":
        return GeminiAdapter()
    if settings.llm_provider == "ollama":
        return OllamaAdapter()
    if settings.llm_api == "responses":
        return ResponsesAdapter()
    return ChatCompletionsAdapter()


@lru_cache(maxsize=1)
def get_small_llm_adapter() -> LLMAdapter:
    """The adapter for side jobs that need no tools (LLM_SMALL_MODEL): always
    Chat Completions, with the provider's default reasoning setting ("none" on
    OpenAI) even when the agent itself reasons through the Responses API, so
    side jobs stay fast and cheap."""
    reasoning_effort = PROVIDER_DEFAULTS[settings.llm_provider][2]
    if settings.llm_provider == "gemini":
        return GeminiAdapter(model=settings.llm_small_model, reasoning_effort=reasoning_effort)
    if settings.llm_provider == "ollama":
        return OllamaAdapter(model=settings.llm_small_model, reasoning_effort=reasoning_effort)
    return ChatCompletionsAdapter(model=settings.llm_small_model, reasoning_effort=reasoning_effort)
