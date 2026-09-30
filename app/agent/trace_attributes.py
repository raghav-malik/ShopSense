"""What every Langfuse trace says about how it was produced.

In Airtap each trace is a single LLM call, so the model and settings sit at the
top of every trace. Here a trace is a whole turn, so the same facts are set at
trace level explicitly, and propagated to every observation in it:

- tags (shown in the trace list, usable as filters): model, provider, API;
- metadata: the models (agent and side jobs), provider, API, reasoning effort,
  the turn limits, and the HTTP request id;
- version: the app version, to compare quality across releases.

Langfuse requires propagated values to be strings of at most 200 characters.
"""

from typing import Any

from app import __version__
from app.config import settings
from app.llm.adapter import LLMAdapter
from app.request_context import current_request_id


def trace_attributes(
    *, trace_name: str, session_id: str, llm: LLMAdapter, small_llm: LLMAdapter | None = None
) -> dict[str, Any]:
    """Keyword arguments for `propagate_attributes()` for a trace made with `llm`
    (and `small_llm` for side jobs, when the trace uses one)."""
    agent = llm.describe()
    metadata = {
        "model": agent.get("model"),
        "provider": agent.get("provider"),
        "api": agent.get("api"),
        "reasoning_effort": agent.get("reasoning_effort"),
        "small_model": small_llm.describe().get("model") if small_llm else None,
        "max_agent_steps": str(settings.max_agent_steps),
        "max_turn_tokens": str(settings.max_turn_tokens),
        "max_turn_cost_usd": str(settings.max_turn_cost_usd),
        "request_id": current_request_id(),
    }
    tags = [f"{key}:{agent[key]}" for key in ("model", "provider", "api") if agent.get(key)]
    return {
        "trace_name": trace_name,
        "session_id": session_id,
        "version": __version__,
        "tags": tags,
        "metadata": {key: value[:200] for key, value in metadata.items() if value},
    }
