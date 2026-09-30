"""The set_budget tool: save the user's budget for this shopping session."""

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from app.db import queries
from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema


class SetBudgetInput(BaseModel):
    """Input schema for the set_budget tool."""

    reasoning: str = Field(..., description="Explain WHY you are setting this budget. What did the user say?")
    # Injected by the registry, never sent by the LLM (see manage_cart).
    session_id: SkipJsonSchema[str] = Field(default="", description="Injected by the registry, not sent by the LLM")
    amount_inr: float | None = Field(
        ...,
        gt=0,
        description="The budget in INR (e.g. 5000 for '5k'). Use null to clear it when the user drops their budget.",
    )


BUDGET_SCHEMA = pydantic_to_tool_schema(
    name="set_budget",
    description=(
        "Save the user's budget for this shopping session when they state one ('under 5k', 'my budget is 3000'), "
        "so it applies to the rest of the conversation. Update it when they change it; clear it (null) when they "
        "say it no longer applies."
    ),
    input_model=SetBudgetInput,
)


async def set_budget(session_id: str, amount_inr: float | None) -> JSONObject:
    """Save or clear the session's budget."""
    await queries.update_session_budget(session_id, amount_inr)
    if amount_inr is None:
        return {"budget_inr": None, "message": "Budget cleared."}
    return {"budget_inr": amount_inr, "message": f"Budget set to ₹{amount_inr:,.0f} for this session."}
