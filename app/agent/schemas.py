"""The agent's response to the API layer."""

from typing import Literal

from pydantic import BaseModel

from app.llm.types import JSONObject

PriceStatus = Literal["verified", "corrected", "unavailable", "unchecked", "search_page"]


class PriceCheck(BaseModel):
    """A link in the answer, checked against the live store page (see app.agent.price_check)."""

    url: str
    store: str
    product: str | None = None  # the link text, or the store page's product name
    status: PriceStatus = "unchecked"
    stated_price: float | None = None  # what the answer said, in INR
    live_price: float | None = None  # what the store page says now
    checked_at: str


class AgentResponse(BaseModel):
    """The response returned by the agent to the API layer."""

    response: str
    tool_calls_made: list[str] = []
    products_found: list[JSONObject] = []
    trace_url: str | None = None
    step_count: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None  # None when a model's price isn't known
    price_checks: list[PriceCheck] = []  # each link in the answer, checked against its store page
