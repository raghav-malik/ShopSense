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
    reasoning: str | None = None  # model's thinking when the provider returns it (Groq gpt-oss; not OpenAI Chat Completions)
    tool_calls: list[ToolCall] | None = None
    finish_reason: str  # 'stop' | 'tool_calls' | 'length'
    usage: dict  # {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    model: str
