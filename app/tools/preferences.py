"""The manage_preferences tool: read or save the user's lasting preferences."""

import json
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.db import queries
from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema


class PreferencesInput(BaseModel):
    """Input schema for the manage_preferences tool."""

    reasoning: str = Field(
        ...,
        description="Explain WHY you are reading or updating preferences. What preference did the user express?",
    )
    action: Literal["get", "set"] = Field(
        ...,
        description="'get' to retrieve all preferences, 'set' to save one",
    )
    key: str | None = Field(
        default=None,
        description="Preference key: 'preferred_brands', 'budget_default', 'category_history', 'preferred_sources'",
    )
    value: str | None = Field(
        default=None,
        description="Preference value as JSON string (required for set)",
    )

    @model_validator(mode="after")
    def check_set_fields(self) -> Self:
        """Saving a preference needs both a key and a value."""
        if self.action == "set" and (not self.key or self.value is None):
            raise ValueError("key and value are required when action is 'set'")
        return self


PREFERENCES_SCHEMA = pydantic_to_tool_schema(
    name="manage_preferences",
    description="Read or update user preferences. Use 'get' to retrieve all stored preferences. Use 'set' to save a new preference (e.g., when the user says 'I prefer Samsung' or 'my budget is usually 5000').",
    input_model=PreferencesInput,
)


async def handle_preferences(
    action: str,
    key: str | None = None,
    value: str | None = None,
) -> JSONObject:
    """Read or update user preferences."""

    if action == "get":
        prefs = await queries.get_all_preferences()
        return {"preferences": prefs}

    elif action == "set":
        if not key or value is None:
            return {"error": "key and value are required for set"}
        # The LLM sometimes sends a bare string ("Samsung") instead of JSON.
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        await queries.set_preference(key, parsed_value)
        prefs = await queries.get_all_preferences()
        return {"message": f"Preference '{key}' updated.", "preferences": prefs}

    return {"error": f"Unknown action: {action}"}
