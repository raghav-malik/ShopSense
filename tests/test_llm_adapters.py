"""Adapter request/response mapping, offline: no LLM is called. These pin down
the provider-specific rules (Gemini thought signatures and temperature,
Responses API item conversion) that the agent core relies on staying hidden."""

import pytest
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

import app.llm.adapter as adapter_module
from app.config import Settings, settings
from app.llm.adapter import (
    ChatCompletionsAdapter,
    GeminiAdapter,
    ResponsesAdapter,
    _messages_to_responses_input,
)
from app.llm.types import PROVIDER_ITEMS_KEY
from app.tools.registry import get_tool_schemas

SIGNED_TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "search_products", "arguments": '{"reasoning":"r","query":"earbuds"}'},
    "extra_content": {"google": {"thought_signature": "SIG_abc123"}},
}

HISTORY = [
    {"role": "system", "content": "SYSTEM PROMPT"},
    {"role": "user", "content": "earbuds under 3000"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{k: v for k, v in SIGNED_TOOL_CALL.items() if k != "extra_content"}],
        PROVIDER_ITEMS_KEY: [SIGNED_TOOL_CALL],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"results": []}'},
]


def gemini_completion(tool_calls=None, content=None) -> ChatCompletion:
    """A Chat Completions response as Gemini's OpenAI-compatible endpoint returns it."""
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gemini-3.8-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
    )


# ---- config ----


def test_gemini_provider_defaults():
    s = Settings(llm_provider="gemini", gemini_api_key="AIza-test")
    assert s.llm_model == "gemini-3.8-flash"
    assert s.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert s.llm_reasoning_effort is None  # Gemini 3 thinking can't be disabled; keep its default
    assert s.llm_api_key == "AIza-test"


def test_gemini_requires_its_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="GEMINI_API_KEY is required"):
        Settings(_env_file=None, llm_provider="gemini")


def test_google_api_key_alias(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-from-alias")
    assert Settings(llm_provider="gemini").llm_api_key == "AIza-from-alias"


@pytest.mark.parametrize(
    ("provider", "api", "expected"),
    [
        ("openai", "chat_completions", ChatCompletionsAdapter),
        ("openai", "responses", ResponsesAdapter),
        ("gemini", "chat_completions", GeminiAdapter),
    ],
)
def test_get_llm_adapter_picks_by_config(monkeypatch, provider, api, expected):
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "llm_api", api)
    monkeypatch.setattr(settings, "gemini_api_key", "AIza-test")
    adapter_module.get_llm_adapter.cache_clear()
    try:
        assert type(adapter_module.get_llm_adapter()) is expected
    finally:
        adapter_module.get_llm_adapter.cache_clear()


# ---- Gemini ----


def test_gemini_request_replays_thought_signatures_and_omits_temperature():
    kwargs = GeminiAdapter()._build_request(HISTORY, get_tool_schemas(), "auto")
    assistant = kwargs["messages"][2]
    # The tool call goes back exactly as Gemini sent it, signature included.
    assert assistant["tool_calls"] == [SIGNED_TOOL_CALL]
    assert all(PROVIDER_ITEMS_KEY not in m for m in kwargs["messages"])
    assert "temperature" not in kwargs  # Google: keep Gemini 3 at its default of 1.0
    assert kwargs["tools"] and kwargs["tool_choice"] == "auto"


def test_gemini_response_keeps_full_tool_calls_as_provider_items():
    response = GeminiAdapter()._to_llm_response(gemini_completion(tool_calls=[SIGNED_TOOL_CALL]))
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call_1"
    assert response.provider_items[0]["extra_content"] == {"google": {"thought_signature": "SIG_abc123"}}


def test_gemini_counts_hidden_thinking_tokens_as_reasoning():
    # Real shape from Gemini's OpenAI endpoint: thinking only appears in total_tokens.
    completion = gemini_completion(content="Neither; both weigh a kilogram.")
    completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens = 18, 11, 227
    buckets = GeminiAdapter()._success_update({"model": "gemini-3.8-flash"}, completion)["usage_details"]
    assert buckets == {"input": 18, "output": 11, "output_reasoning_tokens": 198, "total": 227}
    assert buckets["input"] + buckets["output"] + buckets["output_reasoning_tokens"] == buckets["total"]


def test_gemini_text_response_has_no_provider_items():
    response = GeminiAdapter()._to_llm_response(gemini_completion(content="Here you go."))
    assert response.content == "Here you go." and response.provider_items is None


def test_chat_completions_strips_provider_items_and_sets_temperature():
    kwargs = ChatCompletionsAdapter()._build_request(HISTORY, None, "auto")
    assert all(PROVIDER_ITEMS_KEY not in m for m in kwargs["messages"])
    assert "extra_content" not in kwargs["messages"][2]["tool_calls"][0]
    assert kwargs["temperature"] == settings.llm_temperature
    assert kwargs["max_completion_tokens"] == settings.llm_max_tokens


# ---- Responses API ----


def test_responses_input_conversion():
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}
    call = {"type": "function_call", "call_id": "call_1", "name": "search_products", "arguments": "{}"}
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "earbuds"},
        {"role": "assistant", "content": None, "tool_calls": [SIGNED_TOOL_CALL], PROVIDER_ITEMS_KEY: [reasoning, call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "assistant", "content": "Here are two."},
        {"role": "system", "content": "STEP LIMIT NOTE"},
    ]
    instructions, items = _messages_to_responses_input(messages)
    assert instructions == "SYSTEM PROMPT"
    assert items == [
        {"role": "user", "content": "earbuds"},
        reasoning,  # replayed before the call it preceded
        call,
        {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
        {"role": "assistant", "content": "Here are two."},
        {"role": "developer", "content": "STEP LIMIT NOTE"},  # later system notes stay in place
    ]


def test_responses_request_shape(monkeypatch):
    monkeypatch.setattr(settings, "llm_reasoning_effort", "medium")
    adapter = ResponsesAdapter()
    adapter.reasoning_effort = "medium"
    kwargs = adapter._build_request(HISTORY, get_tool_schemas(), "none")
    assert kwargs["store"] is False and kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "temperature" not in kwargs  # rejected by the API when reasoning is on
    tool = kwargs["tools"][0]
    assert set(tool) == {"type", "name", "description", "parameters", "strict"} and tool["strict"] is False
    assert kwargs["tool_choice"] == "none"
