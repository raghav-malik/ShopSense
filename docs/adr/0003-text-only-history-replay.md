# 3. Replay past turns as plain text

**Status:** Accepted (2026-09-28)

## Context

The spec saved tool results as `role="tool"` messages, but not the assistant messages holding the matching `tool_calls`. On the next turn, the history replayed tool results with no preceding tool call.

- **Groq accepts that.** OpenAI's API rejects it with a 400, because a tool message must answer a preceding tool call.
- **A window can start mid-turn.** Taking the newest 50 messages can begin between a tool call and its result.

## Decision

- **Past turns go back to the model as user and assistant text only.** Anything before the first user message in the window is dropped.
- **Tool rows are still stored,** for `/history` and for debugging, but are never replayed.
- **Within the current turn,** the full tool-call history is kept, so the model sees every result it asked for.

## Consequences

- **Every provider accepts the history,** and a window that starts mid-turn is safe.
- **Requests stay smaller.** Final answers already carry the names, prices and links that matter, and the raw tool JSON (search snippets, page data) isn't resent every turn.
- **The cost:** in a later turn, the model can't see the exact tool results behind an earlier answer. If a follow-up needs detail that wasn't in the answer, the agent searches again.
