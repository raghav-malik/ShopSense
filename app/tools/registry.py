"""The tool registry: schemas for the LLM, and validated, error-safe dispatch of its tool calls."""

import json
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from app.llm.types import JSONObject
from app.tools.cart import CART_SCHEMA, ManageCartInput, manage_cart
from app.tools.compare import COMPARE_SCHEMA, CompareProductsInput, compare_products
from app.tools.extract import EXTRACT_SCHEMA, ExtractProductInput, extract_product_info
from app.tools.preferences import PREFERENCES_SCHEMA, PreferencesInput, handle_preferences
from app.tools.search import SEARCH_SCHEMA, SearchProductsInput, search_products


class ToolSpec(NamedTuple):
    """Everything the registry needs to show a tool to the LLM and run it."""

    # Called with the validated input's fields as keyword arguments.
    executor: Callable[..., Awaitable[JSONObject]]
    schema: JSONObject  # OpenAI function-calling schema shown to the LLM
    input_model: type[BaseModel]  # validates the LLM's arguments before execution
    needs_session: bool  # session_id is injected server-side, never taken from the LLM


TOOL_MAP: dict[str, ToolSpec] = {
    "search_products": ToolSpec(search_products, SEARCH_SCHEMA, SearchProductsInput, needs_session=False),
    "extract_product_info": ToolSpec(extract_product_info, EXTRACT_SCHEMA, ExtractProductInput, needs_session=False),
    "compare_products": ToolSpec(compare_products, COMPARE_SCHEMA, CompareProductsInput, needs_session=False),
    "manage_cart": ToolSpec(manage_cart, CART_SCHEMA, ManageCartInput, needs_session=True),
    "get_preferences": ToolSpec(handle_preferences, PREFERENCES_SCHEMA, PreferencesInput, needs_session=False),
}

# Fields that exist for the LLM's benefit and never reach an executor. `reasoning`
# is recorded on the tool's trace instead.
LLM_ONLY_FIELDS = {"reasoning"}


def get_tool_schemas() -> list[JSONObject]:
    """Returns the list of tool schemas in OpenAI function-calling format."""
    return [spec.schema for spec in TOOL_MAP.values()]


async def execute_tool(name: str, arguments: str, session_id: str) -> str:
    """
    Execute a tool by name with JSON arguments.

    Validation flow (mirrors Airtap's OmniToolUseValidationError pattern):
    1. Check tool exists → structured error listing available tools
    2. Parse JSON → structured error on malformed JSON
    3. Validate against Pydantic model → structured error with field-level
       details AND the expected schema, so the LLM self-corrects in one retry
    4. Execute with validated, typed input → catch runtime errors

    Returns: JSON string (always — even errors are JSON so the LLM can parse them)
    """
    # Step 1: Unknown tool
    if name not in TOOL_MAP:
        return json.dumps(
            {
                "error": "unknown_tool",
                "message": f"No tool named '{name}'.",
                "available_tools": list(TOOL_MAP.keys()),
            }
        )

    spec = TOOL_MAP[name]

    # Step 2: Parse JSON
    try:
        raw_args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "error": "invalid_json",
                "message": f"Could not parse tool arguments as JSON: {e!s}",
                "raw_arguments": arguments[:500],  # truncate for safety
            }
        )
    if not isinstance(raw_args, dict):
        return json.dumps(
            {
                "error": "invalid_json",
                "message": f"Tool arguments must be a JSON object, got {type(raw_args).__name__}.",
                "raw_arguments": arguments[:500],
            }
        )

    # Step 3: Validate against Pydantic model
    # Inject session_id before validation for tools that need it
    if spec.needs_session:
        raw_args["session_id"] = session_id

    try:
        validated = spec.input_model.model_validate(raw_args)
    except ValidationError as e:
        return json.dumps(
            {
                "error": "validation_failed",
                "message": "Tool arguments failed validation. Fix the errors below and retry.",
                "validation_errors": e.errors(include_url=False),
                "expected_schema": spec.input_model.model_json_schema(),
            },
            default=str,
        )  # model_validator errors carry the raw ValueError in ctx

    # Step 4: Execute with validated input
    try:
        result = await spec.executor(**validated.model_dump(exclude=LLM_ONLY_FIELDS))
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:  # noqa: BLE001 - tool errors go back to the LLM as JSON, never crash the loop
        return json.dumps(
            {
                "error": "execution_failed",
                "message": f"Tool '{name}' threw an error: {e!s}",
            }
        )
