"""The ReAct loop with a scripted fake LLM: we control exactly which tool calls
come back, then check what the loop did with them. Tools run for real through
the registry (search is stubbed at the DuckDuckGo boundary); the DB is real
SQLite, isolated per test. The fake is passed in with run_agent(..., llm=...)."""

import copy
import json
from typing import override

import pytest

import app.agent.core as core
from app.config import settings
from app.db import queries
from app.db.models import Session
from app.llm.adapter import LLMAdapter
from app.llm.errors import LLMError, LLMRateLimitError
from app.llm.types import PROVIDER_ITEMS_KEY, ChatMessage, JSONObject, LLMResponse, ToolCall, ToolCallFunction
from app.tools import search

SEARCH_HITS = [
    {
        "title": "boAt Airdopes 141",
        "href": "https://www.amazon.in/dp/B09N3ZNHTY",
        "body": "₹1,099. 42H playback, ENx mics.",
    },
    {"title": "Noise Buds VS104", "href": "https://www.flipkart.com/noise-vs104/p/1", "body": "₹999. 45H playback."},
]


def usage(total: int) -> dict[str, int]:
    return {"prompt_tokens": total - 10, "completion_tokens": 10, "total_tokens": total}


def call(name: str, args: JSONObject, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=json.dumps(args)))


def tool_calls(*calls: ToolCall, tokens: int = 100, provider_items: list[JSONObject] | None = None) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=list(calls),
        finish_reason="tool_calls",
        usage=usage(tokens),
        model="fake",
        provider_items=provider_items,
    )


def answer(text: str, tokens: int = 200) -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop", usage=usage(tokens), model="fake")


class FakeLLM(LLMAdapter):
    """Returns scripted responses in order and records every call it gets."""

    def __init__(self, *script: LLMResponse | Exception) -> None:
        self.script = list(script)
        self.calls: list[JSONObject] = []

    @override
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[JSONObject] | None = None,
        *,
        name: str = "generate-response",
        tool_choice: str = "auto",
        trace_metadata: JSONObject | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "name": name,
                "tool_choice": tool_choice,
                "trace_metadata": trace_metadata,
            }
        )
        if name == "generate-suggestions":
            return answer("1. Compare these two\n- Show cheaper options")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def agent_calls(self) -> list[JSONObject]:
        return [c for c in self.calls if c["name"] != "generate-suggestions"]


@pytest.fixture(autouse=True)
def stub_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here should hit DuckDuckGo; search_products still runs for real."""
    monkeypatch.setattr(search, "_ddgs_text", lambda query, max_results: SEARCH_HITS[:max_results])


@pytest.fixture
async def session(db: None) -> Session:
    return await queries.create_session("test-session-id")


def search_call(call_id: str = "call_1", query: str = "wireless earbuds under 3000") -> ToolCall:
    return call("search_products", {"reasoning": "user wants budget earbuds", "query": query}, call_id)


# ---- the basic loop ----


async def test_tool_call_then_answer(session: Session) -> None:
    llm = FakeLLM(tool_calls(search_call(), tokens=150), answer("The boAt Airdopes 141 at ₹1,099 is my pick."))

    result = await core.run_agent(session.id, "find me wireless earbuds under 3000", llm=llm)

    assert result.response == "The boAt Airdopes 141 at ₹1,099 is my pick."
    assert result.tool_calls_made == ["search_products"]
    assert result.step_count == 2
    assert result.total_tokens == 150 + 200  # agent calls only, not suggestions
    assert [p["url"] for p in result.products_found] == [h["href"] for h in SEARCH_HITS]
    assert result.suggestions == ["Compare these two", "Show cheaper options"]  # list markers stripped

    # Second LLM call sees the assistant's tool call and the matching tool result.
    second = llm.agent_calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1] == {"role": "tool", "content": second[-1]["content"], "tool_call_id": "call_1"}
    assert json.loads(second[-1]["content"])["result_count"] == 2

    # Steps are numbered on the generations for tracing.
    assert [c["trace_metadata"]["step"] for c in llm.agent_calls] == [1, 2]

    # Everything is persisted: user message, tool result, final answer.
    saved = await queries.get_messages(session.id)
    assert [m["role"] for m in saved] == ["user", "tool", "assistant"]
    assert saved[1]["tool_name"] == "search_products" and saved[1]["tool_call_id"] == "call_1"


async def test_answer_without_tools(session: Session) -> None:
    llm = FakeLLM(answer("Could you tell me your budget?"))
    result = await core.run_agent(session.id, "I need earbuds", llm=llm)
    assert result.response == "Could you tell me your budget?"
    assert result.tool_calls_made == [] and result.step_count == 1


async def test_parallel_tool_calls_all_run(session: Session) -> None:
    llm = FakeLLM(
        tool_calls(search_call("call_a", "boAt earbuds"), search_call("call_b", "Noise earbuds")),
        answer("Here are both."),
    )
    result = await core.run_agent(session.id, "compare boAt and Noise", llm=llm)
    assert result.tool_calls_made == ["search_products", "search_products"]
    tool_msgs = [m for m in llm.agent_calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_a", "call_b"]


# ---- tool errors go back to the model, the loop keeps going ----


async def test_unknown_tool_error_is_fed_back(session: Session) -> None:
    llm = FakeLLM(tool_calls(call("buy_now", {"reasoning": "x"}, "call_1")), answer("Sorry, I can't buy directly."))
    result = await core.run_agent(session.id, "buy it", llm=llm)
    assert result.response == "Sorry, I can't buy directly."
    fed_back = json.loads(llm.agent_calls[1]["messages"][-1]["content"])
    assert fed_back["error"] == "unknown_tool"


async def test_validation_error_lets_the_model_self_correct(session: Session) -> None:
    bad = call(
        "manage_cart", {"reasoning": "user asked", "action": "add", "product_name": "boAt Airdopes 141"}, "call_1"
    )
    fixed = call(
        "manage_cart",
        {
            "reasoning": "retry with url",
            "action": "add",
            "product_name": "boAt Airdopes 141",
            "price": 1099,
            "url": "https://www.amazon.in/dp/B09N3ZNHTY",
        },
        "call_2",
    )
    llm = FakeLLM(tool_calls(bad), tool_calls(fixed), answer("Added to your cart."))

    result = await core.run_agent(session.id, "add the boAt to my cart", llm=llm)

    first_error = json.loads(llm.agent_calls[1]["messages"][-1]["content"])
    assert first_error["error"] == "validation_failed" and "expected_schema" in first_error
    assert result.response == "Added to your cart."
    cart = await queries.get_cart(session.id)
    assert [(i["product_name"], i["price"]) for i in cart] == [("boAt Airdopes 141", 1099.0)]


# ---- guardrail: max steps ----


async def test_max_steps_forces_a_final_answer(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_agent_steps", 2)
    llm = FakeLLM(
        tool_calls(search_call("call_1")),
        tool_calls(search_call("call_2", "earbuds ANC")),
        answer("From what I found, the boAt Airdopes 141 fits best."),  # the forced final call
    )
    result = await core.run_agent(session.id, "find earbuds", llm=llm)

    assert result.step_count == 2
    assert result.response == "From what I found, the boAt Airdopes 141 fits best."
    final = llm.agent_calls[-1]
    assert final["name"] == "answer-at-step-limit" and final["tool_choice"] == "none"
    assert final["messages"][-1]["role"] == "system" and "research limit" in final["messages"][-1]["content"]
    assert result.suggestions  # still generated on this path


async def test_max_steps_falls_back_to_results_if_final_call_fails(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "max_agent_steps", 1)
    llm = FakeLLM(tool_calls(search_call()), LLMError("final call failed"))
    result = await core.run_agent(session.id, "find earbuds", llm=llm)
    assert result.response.startswith("I ran out of research steps")
    assert SEARCH_HITS[0]["href"] in result.response  # the user still gets the links


# ---- failures and edge cases ----


async def test_llm_error_propagates_to_the_caller(session: Session) -> None:
    llm = FakeLLM(LLMRateLimitError("rate limited", retry_after=5))
    with pytest.raises(LLMRateLimitError):
        await core.run_agent(session.id, "find earbuds", llm=llm)
    # The user's message was saved before the failure.
    assert [m["role"] for m in await queries.get_messages(session.id)] == ["user"]


async def test_unknown_session(db: None) -> None:
    llm = FakeLLM()
    result = await core.run_agent("no-such-session", "hello", llm=llm)
    assert result.response.startswith("Session not found")
    assert llm.calls == []


# ---- context building ----


async def test_system_prompt_includes_preferences_cart_and_budget(session: Session) -> None:
    await queries.set_preference("preferred_brands", ["Samsung"])
    await queries.add_to_cart(session.id, "Noise Buds VS104", 999, "https://x/noise")
    await queries.update_session_budget(session.id, 3000)
    llm = FakeLLM(answer("Noted."))

    await core.run_agent(session.id, "hi", llm=llm)

    system = llm.agent_calls[0]["messages"][0]
    assert system["role"] == "system"
    assert '"Samsung"' in system["content"]
    assert "Noise Buds VS104: ₹999" in system["content"]
    assert "budget of ₹3000" in system["content"]


async def test_history_is_replayed_as_text_without_tool_rows(session: Session) -> None:
    llm = FakeLLM(tool_calls(search_call()), answer("The boAt Airdopes 141 is my pick."))
    await core.run_agent(session.id, "find earbuds", llm=llm)

    llm = FakeLLM(answer("Added."))
    await core.run_agent(session.id, "add it to my cart", llm=llm)

    roles = [(m["role"], m.get("content")) for m in llm.agent_calls[0]["messages"][1:]]
    # Tool rows are in the DB but not replayed: a tool result without its
    # assistant tool_calls message is invalid in the OpenAI format (SR-40).
    assert roles == [
        ("user", "find earbuds"),
        ("assistant", "The boAt Airdopes 141 is my pick."),
        ("user", "add it to my cart"),
    ]


async def test_provider_items_are_carried_to_the_next_call(session: Session) -> None:
    # Responses API: reasoning items must go back with the function calls they preceded.
    items: list[JSONObject] = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
        {"type": "function_call", "call_id": "call_1", "name": "search_products", "arguments": "{}"},
    ]
    llm = FakeLLM(tool_calls(search_call(), provider_items=items), answer("Done."))
    await core.run_agent(session.id, "find earbuds", llm=llm)
    assistant = llm.agent_calls[1]["messages"][-2]
    assert assistant[PROVIDER_ITEMS_KEY] == items


async def test_suggestions_see_the_final_answer(session: Session) -> None:
    llm = FakeLLM(answer("The boAt Airdopes 141 is my pick."))
    await core.run_agent(session.id, "find earbuds", llm=llm)
    suggestion_call = next(c for c in llm.calls if c["name"] == "generate-suggestions")
    assert "The boAt Airdopes 141 is my pick." in suggestion_call["messages"][-1]["content"]  # SR-41
