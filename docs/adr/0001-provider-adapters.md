# 1. One adapter interface over several LLM providers and APIs

**Status:** Accepted (2026-09-28)

## Context

The project started on Groq, whose chosen model was then shut down. It moved to OpenAI, and later gained Gemini as an option. The spec claimed that switching providers was "change the base URL", but testing showed otherwise:

- **GPT-6** rejects `max_tokens`, rejects a custom temperature unless reasoning is off, and only allows function tools in Chat Completions with `reasoning_effort="none"`.
- **Gemini 3** must get its tool calls' thought signatures back unchanged, wants temperature left at its default, and reports thinking tokens only in the total.
- **The OpenAI Responses API** has a different request and response shape, and needs its encrypted reasoning items sent back with the function calls they preceded.

## Decision

- **One interface.** `LLMAdapter.chat(messages, tools, *, name, tool_choice, trace_metadata) -> LLMResponse`, with messages and tools always in Chat Completions format. Each adapter converts to its own API:
  - `ChatCompletionsAdapter` for OpenAI, Groq, and any OpenAI-compatible server
  - `ResponsesAdapter`
  - `GeminiAdapter`
- **Shared by all adapters** (`_OpenAISDKAdapter`): one client with SDK retries off, the retry-once policy, and the mapping of SDK exceptions to four provider-agnostic errors.
- **Provider-specific data passes through opaquely.** Reasoning items and thought signatures travel on the assistant message under `_provider_items`, and only the adapter that produced them reads them.
- **The agent receives its adapters** (`run_agent(..., llm=, small_llm=)`). A setting picks the provider (`LLM_PROVIDER`, `LLM_API`), with per-provider defaults.

## Consequences

- **Adding Gemini needed zero changes** in the agent, tools or routes.
- **Tests inject a scripted fake LLM,** with no patching.
- **Enforced:** import-linter ensures that only `app.llm` imports the OpenAI SDK.
- **The cost:** every provider must speak something close to Chat Completions. A provider with a truly different API needs a full adapter, including its own error mapping.
- **Switching providers mid-turn isn't supported.** `_provider_items` are specific to one provider. Airtap handles this by tagging items with the model that produced them; ShopSense doesn't need it yet.
