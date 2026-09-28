# 2. OpenAI gpt-6-luna on Chat Completions as the default model

**Status:** Accepted (2026-09-28)

## Context

- **Llama 3.3 70B.** The spec targeted it on Groq's free tier, but the model was shut down, so its replacement, `openai/gpt-oss-120b`, was used instead.
- **Groq's free tier couldn't sustain the loop.** The limit is 8,000 tokens per minute for all its chat models. A ReAct loop resends a growing context every step, so runs hit 429s every other call, and one request exceeded the limit on its own (413).
- **Gemini** worked, but on the day it was tested it was often overloaded (503), and a turn cost 7–14× more.

## Decision

- **The default is OpenAI `gpt-6-luna` on Chat Completions,** with `reasoning_effort="none"`. The user's paid key is Tier 5.
- **Reasoning with tools is opt-in** through the Responses API (`LLM_API=responses`).
- **Side jobs** (suggestions, the request check) use `LLM_SMALL_MODEL`. On OpenAI that is also `gpt-6-luna`, the cheapest current-generation model.

## Consequences

- **Speed and cost.** Turns take about 20–40s and cost about $0.001, with no rate-limit waits.
- **The model doesn't "think" on the default path.** The prompt spells out a research workflow (search, then targeted searches, then recommend, then stop), because without it the model stopped after one search.
- **The Responses API is available for reasoning.** It adds reasoning summaries to the traces, but a turn takes about 2× the time and more tokens.
- **Groq and Gemini remain one setting away,** but aren't tested day to day.
