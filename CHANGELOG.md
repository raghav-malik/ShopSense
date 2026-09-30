# Changelog

Notable changes to ShopSense. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org).

## [Unreleased]

## [1.0.0] - 2026-09-30

### Added
- `set_budget` tool: "under 5k" or "my budget is 3000" sets the session's budget, which the system prompt enforces and the sidebar shows. The session budget was dead code until now (SR-13). A web page can't set it: changes pass the request check.
- Langfuse traces carry the model, provider and API as tags (visible and filterable in the trace list), the app version, and metadata (models, reasoning effort, turn limits, request id) on the trace and every observation in it, as Airtap's traces show them. `/health` reports the version.
- README with the architecture, setup from clone to running, screenshots and an example conversation, the tech stack and why, the Airtap-inspired patterns, how to add a tool, and how to swap LLM providers.
- `AGENTS.md`: conventions, module layering and workflow for contributors and AI coding agents.
- Architecture decision records in `docs/adr/`.
- Module layering enforced by import-linter, in the commit hooks and in CI; only `app.llm` may import a provider SDK.
- Conventional Commits, checked by a commit-msg hook.
- `scripts/smoke_test.py`: one live agent turn in a throwaway database.

### Changed
- Every public module, class and function has a docstring, enforced by ruff.
- `get_preferences` is renamed `manage_preferences`, since it also saves preferences (SR-38).
- A session's `updated_at` moves with every message (SR-80).

### Fixed
- **Adding to the cart in a follow-up** ("add the cheapest", "ok add that one") now adds the product already shown, without searching again. Before, the agent re-verified the price, often couldn't, and refused. The cart tool no longer claims a price is required, and an unpriced item shows as "price unknown". Live: 6 of 6 follow-up adds succeeded, up from 1 of 3 in the traffic run.
- **The request check reads choices as cart requests** ("I'll take the second one", "add that one"). It scores 31/31 on `evals/request_check.py`, which now includes budget phrasings.

### Removed
- The `__main__` debug blocks and their `print` calls in the adapter, agent core, queries and search modules.

## [0.3.0] - 2026-09-29: security and reliability

### Added
- **SSRF protection** in `extract_product_info`: public addresses only, a connection pinned to the checked IP, redirects re-checked on every hop, and ports 80 and 443 only (#1).
- **Untrusted web content:**
  - results labelled `web_content_notice`
  - invisible characters stripped (tag characters, variation selectors, zero-width and bidi)
  - markdown images removed from answers (#3)
- **Request check:** cart and preference changes run only when the user's own message asks for them. It uses a separate small-model call that never sees tool results (#3).
- **Live evals:** `evals/prompt_injection.py` and `evals/request_check.py` (#3).
- **Per-job models:** `LLM_SMALL_MODEL` for suggestions and the request check (#2).
- **Concurrent web tools:** search, extract and compare run in parallel (at most 4), while cart and preference calls stay in order (#2).
- **Turn budgets:** `MAX_TURN_TOKENS` and `MAX_TURN_COST_USD`, with a per-model price table (cached input priced at the cached rate) and `estimated_cost_usd` in chat responses and the UI (#6).
- **Repeat guard:** identical tool calls in one turn run once (#6).
- **Health probes:** `/livez` and `/readyz` (#7).
- **Request ids and JSON logs:** every request gets an id (`X-Request-ID`), which appears on its log lines and on every Langfuse observation of its trace. Logs are JSON lines, or text with `LOG_FORMAT=text` (#7).
- **`POST /sessions/{id}/suggestions`** (#4).

### Changed
- Follow-up suggestions are made after the answer is shown, not before, so replies arrive about 1.5s sooner (#4).
- The final answer at a limit is traced as `answer-at-limit` (was `answer-at-step-limit`), with the reason (#6).
- Unhandled errors become the standard 500 in the request middleware, so the response and log line carry the request id (#7).

### Removed
- `suggestions` from the chat response; use the suggestions endpoint (#4).

### Security
- API keys are `SecretStr`: they no longer appear when settings are printed, logged or serialized. Traces also redact the configured keys by value, whatever their format (#5).

## [0.2.0] - 2026-09-28: engineering practices

### Added
- `pyproject.toml` with a uv lockfile; `requirements.txt` is generated from it.
- ruff with an explicit rule set; mypy `--strict` with the Pydantic plugin; a coverage floor of 85%.
- Offline tests for the retry and error mapping, the tracing guards, trace masking and product-page extraction.
- pre-commit hooks and GitHub Actions CI: format, lint, types, tests with coverage, a requirements drift check, pip-audit, and a gitleaks scan of all history.
- A ruleset on `main`: pull requests only, CI must pass, no bypass.

### Changed
- The LLM adapter is injected into the agent (`run_agent(..., llm=...)`).
- Chat messages and database rows are typed (TypedDicts).

## [0.1.0] - 2026-09-28: initial build

### Added
- **Config:** typed settings from `.env`, with per-provider defaults and validation at startup.
- **LLM layer:**
  - adapters for OpenAI Chat Completions (default: `gpt-6-luna`), the OpenAI Responses API, Groq and Gemini
  - retry-once for transient failures, and provider-agnostic errors
- **Tools:** `search_products` (ddgs, region `in-en`), `extract_product_info`, `compare_products`, `manage_cart` and `get_preferences`, with validated inputs and errors the model can act on.
- **SQLite storage** (aiosqlite, WAL) for sessions, messages, the cart and preferences.
- **The ReAct agent,** with a step limit that answers from research, text-only history replay, and follow-up suggestions.
- **Langfuse tracing:**
  - one trace per turn and one generation per LLM attempt, opened before the call and completed after (adapted from Airtap)
  - typed tool observations
  - PII and key masking
- **FastAPI:** session, chat, cart and history endpoints; one error format; a Langfuse auth check at startup.
- **Streamlit UI:** product cards, suggestion chips, a cart sidebar, a debug link to each trace, and the session kept in the URL.
- **Tests:** hermetic, with network tests marked.

[Unreleased]: https://github.com/raghav-malik/ShopSense/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/raghav-malik/ShopSense/compare/b10c1ba...v1.0.0
[0.3.0]: https://github.com/raghav-malik/ShopSense/compare/245c773...b10c1ba
[0.2.0]: https://github.com/raghav-malik/ShopSense/compare/5293387...245c773
[0.1.0]: https://github.com/raghav-malik/ShopSense/commits/5293387
