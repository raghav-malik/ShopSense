# 6. A hand-written ReAct loop instead of an agent framework

**Status:** Accepted (2026-09-28)

## Context

Agent frameworks provide a loop, tool dispatch, memory and tracing hooks. But the parts that make ShopSense work are specific to it:

- the step limit and turn budgets, with a final answer when one is reached
- tool errors fed back so the model can correct itself
- the request check before any change
- concurrent web tools with ordered session tools
- the repeat guard
- a Langfuse generation per attempt, with step metadata
- text-only history replay

## Decision

Write the ReAct loop by hand in `app/agent/core.py`, about 160 lines. Tools are plain async functions with Pydantic input models, dispatched by `app/tools/registry.py`.

## Consequences

- **Readable.** Every guardrail and every trace call is in one file, in the order it runs.
- **Easy to test.** Tests pass a scripted fake LLM and check exactly what the loop did.
- **No framework upgrades to track.**
- **The cost:** features a framework would provide are ours to build and maintain, such as streaming responses, a resumable step store, or multi-agent handoffs.
- **When to revisit:** if the agent needs several of those at once.
