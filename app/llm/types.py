"""Types shared by the agent and the LLM adapters.

Messages and tool calls stay plain dicts (the OpenAI Chat Completions wire
format the agent builds and the adapters send), typed with TypedDicts so the
checker knows every key. Adapters for other APIs convert from this format.
"""

from typing import Any, Final, Literal, NotRequired, TypedDict

from pydantic import BaseModel

# Free-form JSON: tool schemas, tool results, raw provider items. Named, so a
# `dict[str, Any]` in a signature reads as "JSON by design", not "untyped".
type JSONObject = dict[str, Any]

Role = Literal["system", "developer", "user", "assistant", "tool"]

# Message key carrying LLMResponse.provider_items through the chat-format
# history. Adapters that don't use it strip it before sending.
PROVIDER_ITEMS_KEY: Final = "_provider_items"


class ChatToolCallFunction(TypedDict):
    """The function part of a tool call in Chat Completions format."""

    name: str
    arguments: str  # JSON string


class ChatToolCall(TypedDict):
    """A tool call on an assistant message, in Chat Completions format."""

    id: str
    type: Literal["function"]
    function: ChatToolCallFunction


class ChatMessage(TypedDict):
    """One message in the agent's history, in Chat Completions format."""

    role: Role
    content: str | None
    tool_calls: NotRequired[list[ChatToolCall]]
    tool_call_id: NotRequired[str]  # on role="tool"
    name: NotRequired[str]
    # Opaque items to replay verbatim on the next call of this turn (see PROVIDER_ITEMS_KEY).
    _provider_items: NotRequired[list[JSONObject]]


class ToolCallFunction(BaseModel):
    """The function an LLM asked to call, with its arguments as a JSON string."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from an LLM response."""

    id: str
    type: str = "function"
    function: ToolCallFunction


class LLMResponse(BaseModel):
    """One LLM response, in the same shape for every provider and API."""

    content: str | None = None
    reasoning: str | None = None  # model's thinking when returned (Groq gpt-oss; OpenAI Responses summaries)
    tool_calls: list[ToolCall] | None = None
    finish_reason: str  # 'stop' | 'tool_calls' | 'length'
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens, cached_tokens (part of prompt)
    model: str
    # Opaque output items the provider needs back on the next call of this turn
    # (Responses API: reasoning items + function calls; Gemini: signed tool calls).
    # Callers attach them to the assistant message as PROVIDER_ITEMS_KEY.
    provider_items: list[JSONObject] | None = None
