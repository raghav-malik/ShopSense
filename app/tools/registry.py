import json
from typing import Callable, Type

from pydantic import BaseModel, ValidationError

from app.tools.cart import CART_SCHEMA, ManageCartInput, manage_cart
from app.tools.compare import COMPARE_SCHEMA, CompareProductsInput, compare_products
from app.tools.extract import EXTRACT_SCHEMA, ExtractProductInput, extract_product_info
from app.tools.preferences import PREFERENCES_SCHEMA, PreferencesInput, handle_preferences
from app.tools.search import SEARCH_SCHEMA, SearchProductsInput, search_products

# Tool name → (executor, schema, Pydantic input model, needs_session_id)
TOOL_MAP: dict[str, tuple[Callable, dict, Type[BaseModel], bool]] = {
    "search_products":      (search_products,      SEARCH_SCHEMA,      SearchProductsInput,  False),
    "extract_product_info": (extract_product_info, EXTRACT_SCHEMA,     ExtractProductInput,  False),
    "compare_products":     (compare_products,     COMPARE_SCHEMA,     CompareProductsInput, False),
    "manage_cart":          (manage_cart,          CART_SCHEMA,        ManageCartInput,      True),
    "get_preferences":      (handle_preferences,   PREFERENCES_SCHEMA, PreferencesInput,     False),
}

# Fields that exist for the LLM's benefit and never reach an executor. `reasoning`
# is recorded on the tool's trace instead.
LLM_ONLY_FIELDS = {"reasoning"}


def get_tool_schemas() -> list[dict]:
    """Returns the list of tool schemas in OpenAI function-calling format."""
    return [schema for _, schema, _, _ in TOOL_MAP.values()]


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
        return json.dumps({
            "error": "unknown_tool",
            "message": f"No tool named '{name}'.",
            "available_tools": list(TOOL_MAP.keys()),
        })

    executor, _, input_model, needs_session = TOOL_MAP[name]

    # Step 2: Parse JSON
    try:
        raw_args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": "invalid_json",
            "message": f"Could not parse tool arguments as JSON: {str(e)}",
            "raw_arguments": arguments[:500],  # truncate for safety
        })
    if not isinstance(raw_args, dict):
        return json.dumps({
            "error": "invalid_json",
            "message": f"Tool arguments must be a JSON object, got {type(raw_args).__name__}.",
            "raw_arguments": arguments[:500],
        })

    # Step 3: Validate against Pydantic model
    # Inject session_id before validation for tools that need it
    if needs_session:
        raw_args["session_id"] = session_id

    try:
        validated: BaseModel = input_model.model_validate(raw_args)
    except ValidationError as e:
        return json.dumps({
            "error": "validation_failed",
            "message": "Tool arguments failed validation. Fix the errors below and retry.",
            "validation_errors": e.errors(include_url=False),
            "expected_schema": input_model.model_json_schema(),
        }, default=str)  # model_validator errors carry the raw ValueError in ctx

    # Step 4: Execute with validated input
    try:
        result = await executor(**validated.model_dump(exclude=LLM_ONLY_FIELDS))
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({
            "error": "execution_failed",
            "message": f"Tool '{name}' threw an error: {str(e)}",
        })
