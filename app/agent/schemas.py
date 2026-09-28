from pydantic import BaseModel


class AgentResponse(BaseModel):
    """The response returned by the agent to the API layer."""
    response: str
    tool_calls_made: list[str] = []
    products_found: list[dict] = []
    suggestions: list[str] = []  # Follow-on suggestions for the user
    trace_url: str | None = None
    step_count: int = 0
    total_tokens: int = 0
