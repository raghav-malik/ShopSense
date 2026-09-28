"""Turn budget (tokens and estimated cost) and the repeated-call guard, as
units; test_agent.py covers them inside the agent loop."""

import json

import pytest

from app.agent.guardrails import RepeatGuard, TurnBudget, estimate_cost_usd, repeat_result


def usage(prompt: int, completion: int, hidden: int = 0) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion + hidden}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-6-luna", (10_000 * 0.10 + 1_000 * 0.50) / 1e6),
        ("gpt-6-luna-2026-07-01", (10_000 * 0.10 + 1_000 * 0.50) / 1e6),  # dated names match their family
        ("gpt-6-sol", (10_000 * 2.00 + 1_000 * 10.00) / 1e6),
        ("some-unpriced-model", None),
    ],
)
def test_estimate_cost(model: str, expected: float | None) -> None:
    cost = estimate_cost_usd(model, usage(10_000, 1_000))
    assert cost == pytest.approx(expected) if expected is not None else cost is None


def test_cached_input_is_billed_at_the_cached_rate() -> None:
    cached_usage = {**usage(10_000, 1_000), "cached_tokens": 8_000}
    assert estimate_cost_usd("gpt-6-luna", cached_usage) == pytest.approx(
        (2_000 * 0.10 + 8_000 * 0.01 + 1_000 * 0.50) / 1e6
    )


def test_hidden_thinking_tokens_are_billed_as_output() -> None:
    # Gemini reports thinking only in total_tokens (SR-75).
    assert estimate_cost_usd("gemini-3.8-flash", usage(1_000, 100, hidden=900)) == pytest.approx(
        (1_000 * 0.75 + 1_000 * 3.75) / 1e6
    )


def test_budget_by_tokens() -> None:
    budget = TurnBudget(max_tokens=50_000, max_cost_usd=1.0)
    budget.add("gpt-6-luna", usage(30_000, 1_000))
    assert budget.limit_reached() is None
    budget.add("gpt-6-luna", usage(18_000, 1_000))
    reason = budget.limit_reached()
    assert reason is not None and "token budget" in reason


def test_budget_by_cost() -> None:
    budget = TurnBudget(max_tokens=10_000_000, max_cost_usd=0.05)
    budget.add("gpt-6-sol", usage(20_000, 2_000))  # $0.04 + $0.02
    assert (reason := budget.limit_reached()) is not None and "cost budget" in reason


def test_unpriced_models_are_bounded_by_tokens_only() -> None:
    budget = TurnBudget(max_tokens=10_000, max_cost_usd=0.0001)
    budget.add("some-unpriced-model", usage(5_000, 100))
    assert budget.limit_reached() is None and budget.unpriced_calls == 1
    budget.add("some-unpriced-model", usage(5_000, 100))
    reason = budget.limit_reached()
    assert reason is not None and "token budget" in reason


def args(**kwargs: object) -> str:
    return json.dumps(kwargs)


def test_repeat_guard_matches_the_same_call() -> None:
    guard = RepeatGuard()
    assert guard.first_seen_step("search_products", args(reasoning="a", query="Noise Buds  price"), 1) is None
    # Different reasoning, case and spacing: still the same search.
    assert guard.first_seen_step("search_products", args(reasoning="b", query="noise buds price"), 3) == 1


@pytest.mark.parametrize(
    ("second_tool", "second_args"),
    [
        ("search_products", args(query="noise buds price", max_results=3)),  # different arguments
        ("extract_product_info", args(query="noise buds price")),  # different tool
    ],
)
def test_repeat_guard_lets_different_calls_through(second_tool: str, second_args: str) -> None:
    guard = RepeatGuard()
    guard.first_seen_step("search_products", args(query="noise buds price"), 1)
    assert guard.first_seen_step(second_tool, second_args, 2) is None


def test_repeat_result_points_to_the_earlier_step() -> None:
    result = json.loads(repeat_result("search_products", 2))
    assert result["error"] == "repeated_call" and "step 2" in result["message"] and result["hint"]
