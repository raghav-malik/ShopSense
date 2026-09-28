# Architecture decision records

Short records of decisions that shaped ShopSense. Each one gives the context, the decision, and its consequences, including the costs. Add one when you make a decision someone new to the code would otherwise question. Number them in order and don't rewrite old ones; if a decision is replaced, write a new record and mark the old one "Superseded by".

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-provider-adapters.md) | One adapter interface over several LLM providers and APIs | Accepted |
| [0002](0002-openai-default.md) | OpenAI `gpt-6-luna` on Chat Completions as the default model | Accepted |
| [0003](0003-text-only-history-replay.md) | Replay past turns as plain text | Accepted |
| [0004](0004-tracing-design.md) | One Langfuse trace per turn, one generation per LLM attempt | Accepted |
| [0005](0005-untrusted-web-content.md) | Treat web content as untrusted; changes need the user's say-so | Accepted |
| [0006](0006-hand-written-agent-loop.md) | A hand-written ReAct loop instead of an agent framework | Accepted |
