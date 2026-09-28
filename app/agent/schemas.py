"""The agent's response to the API layer."""

from pydantic import BaseModel

from app.llm.types import JSONObject


class AgentResponse(BaseModel):
    """The response returned by the agent to the API layer."""

    response: str
    tool_calls_made: list[str] = []
    products_found: list[JSONObject] = []
    trace_url: str | None = None
    step_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None  # None when a model's price isn't known
