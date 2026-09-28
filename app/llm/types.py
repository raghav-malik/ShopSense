from pydantic import BaseModel


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class LLMMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # for tool result messages


class LLMResponse(BaseModel):
    content: str | None = None
    reasoning: str | None = None  # model's thinking when returned (Groq gpt-oss; OpenAI Responses summaries)
    tool_calls: list[ToolCall] | None = None
    finish_reason: str  # 'stop' | 'tool_calls' | 'length'
    usage: dict  # {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    model: str
    # Opaque output items the provider needs back on the next call of this turn
    # (Responses API: reasoning items with encrypted content + function calls).
    # Callers attach them to the assistant message as PROVIDER_ITEMS_KEY.
    provider_items: list[dict] | None = None


# Message key carrying LLMResponse.provider_items through the chat-format history.
# Adapters that don't use it strip it before sending.
PROVIDER_ITEMS_KEY = "_provider_items"
