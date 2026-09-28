from pydantic import BaseModel

from app.llm.types import JSONObject


def pydantic_to_tool_schema(
    name: str,
    description: str,
    input_model: type[BaseModel],
) -> JSONObject:
    """
    Build an OpenAI-compatible tool schema from a Pydantic model.

    The schema and the validation are always in sync — change the
    Pydantic model, both the LLM's view and the runtime check update.

    This replaces hand-written JSON schemas that can drift from the
    actual function signatures.
    """
    # model_json_schema() produces a JSON Schema dict from the Pydantic model.
    # It includes type, properties, required, descriptions — everything
    # the OpenAI tool format needs.
    json_schema = input_model.model_json_schema()

    # Remove Pydantic-specific keys the OpenAI API doesn't expect
    json_schema.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": json_schema,
        },
    }
