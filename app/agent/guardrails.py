"""Per-turn guardrails: a token and cost budget, and no repeated tool calls.

MAX_AGENT_STEPS bounds how many LLM calls a turn makes, but not what they
cost: every step resends the growing context, so a turn that pulls in long
pages or a pricier model can use far more than a normal turn (typically about
15-20K tokens and $0.001-0.002 on gpt-6-luna, measured in Langfuse). The
budget is a runaway guard, set well above normal turns; when it's reached the
agent stops researching and answers from what it has, like at the step limit.

The repeat guard stops the loop seen in SR-64: the model calling the same
search again and again while hunting for a "better" link.
"""

import json
from dataclasses import dataclass, field

from app.llm.types import JSONObject

# USD per 1M tokens (input, cached input, output), from the providers' pricing
# pages on 2026-09-29, like Airtap's per-model pricing in its model registry.
# Cached input matters: in a multi-step turn most of each prompt is the
# previous step's context, and about 75% of a typical turn's input tokens were
# cached (Langfuse). Gemini's cached rate isn't listed here, so it's billed at
# the full input rate (an overestimate, the safe side for a budget). Models not
# listed are still bounded by the token budget.
MODEL_PRICES_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gemini-3.8-flash": (0.75, 0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 0.25, 1.50),
}


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float | None:
    """Cost of one call from its usage, or None for a model without a price.
    Output includes reasoning tokens, which providers bill as output."""
    prices = MODEL_PRICES_USD_PER_1M.get(model) or next(
        # Responses carry dated names like "gpt-6-luna-2026-07-01".
        (p for name, p in MODEL_PRICES_USD_PER_1M.items() if model.startswith(f"{name}-")),
        None,
    )
    if prices is None:
        return None
    input_price, cached_price, output_price = prices
    prompt = usage.get("prompt_tokens", 0)
    cached = min(usage.get("cached_tokens", 0), prompt)
    output = max(usage.get("total_tokens", 0) - prompt, 0)  # completion plus any hidden thinking
    return ((prompt - cached) * input_price + cached * cached_price + output * output_price) / 1_000_000


@dataclass
class TurnBudget:
    """Tokens and estimated cost of the agent's LLM calls in one turn."""

    max_tokens: int
    max_cost_usd: float
    tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0

    def add(self, model: str, usage: dict[str, int]) -> None:
        self.tokens += usage.get("total_tokens", 0)
        cost = estimate_cost_usd(model, usage)
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += cost

    def limit_reached(self) -> str | None:
        """Why the budget is used up, or None while there's room left."""
        if self.tokens >= self.max_tokens:
            return f"turn token budget reached ({self.tokens:,} of {self.max_tokens:,} tokens)"
        if self.cost_usd >= self.max_cost_usd:
            return f"turn cost budget reached (${self.cost_usd:.4f} of ${self.max_cost_usd:.2f})"
        return None


def _normalized(value: object) -> object:
    """Arguments as the model means them: "Noise Buds  price" and
    "noise buds price" are the same search."""
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return {k: _normalized(v) for k, v in value.items() if k != "reasoning"}
    if isinstance(value, list):
        return [_normalized(v) for v in value]
    return value


@dataclass
class RepeatGuard:
    """Remembers the tool calls made this turn, by tool and arguments (the
    `reasoning` text doesn't count, and case and spacing are ignored)."""

    _seen: dict[str, int] = field(default_factory=dict)

    def first_seen_step(self, tool_name: str, arguments: str, step: int) -> int | None:
        """The step that already made this call, or None (and it's recorded)."""
        try:
            args: object = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return None  # the registry reports bad JSON; nothing to compare
        key = json.dumps([tool_name, _normalized(args)], sort_keys=True)
        if key in self._seen:
            return self._seen[key]
        self._seen[key] = step
        return None


def repeat_result(tool_name: str, first_step: int) -> str:
    """The tool result the agent gets instead of running the same call again."""
    result: JSONObject = {
        "error": "repeated_call",
        "message": f"Not run again: {tool_name} was already called with these arguments at step {first_step}.",
        "hint": "Use that earlier result. If it wasn't enough, try different arguments or answer with what you have.",
    }
    return json.dumps(result)
