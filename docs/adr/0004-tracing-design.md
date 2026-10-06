# 4. One Langfuse trace per turn, one generation per LLM attempt

**Status:** Accepted (2026-09-28)

## Context

The spec's tracing used the Langfuse v2 SDK API, which no longer imports. It also used `@observe` on the LLM call, which records every argument, including the adapter object and all the tool schemas. Retries were invisible and tools were generic spans.

What a trace needs to show: each LLM call's real start time and exact input, every failed attempt (including the one before a retry), and the whole turn as one unit to evaluate. And tracing must never be able to break a request.

## Decision

- **One trace per chat turn:** an `@observe(as_type="agent")` root named `run-agent`.
  - Its input is the user's message and its output is the answer.
  - `session_id` and the HTTP `request_id` are on every observation.
- **One generation per LLM attempt,** through `GenerationTrace`:
  - It opens before the call, so the start time and input are exact.
  - It completes after the call with `success()` (output, token usage by bucket) or `error()` (level ERROR, with the provider's error body).
  - Every Langfuse call is guarded, so a tracing bug is logged and never replaces the real result.
- **Tools are typed observations:** `retriever` for search and extract, `tool` for the rest.
  - The model's stated reason for the call goes in metadata.
  - A tool error is marked WARNING.
- **Masking and export.**
  - Masking redacts emails and API keys before export, including the configured keys by value.
  - The export timeout is 20s; the SDK's 5s default dropped data.
- **Side calls get their own traces.** Follow-up suggestions are a separate trace in the same session.

## Consequences

- **Each turn is one unit** for evaluation: one input, one output, and the whole agent graph underneath. Langfuse recommends this layout.
- **Failures are visible.** A 429 before a retry shows as its own ERROR generation.
- **The cost:** unlike a one-trace-per-call layout, a long turn appears in Langfuse only once it finishes. OpenTelemetry exports spans when they end, so neither layout shows a call while it's running.
