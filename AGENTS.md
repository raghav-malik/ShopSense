# AGENTS.md

How to work in this repository, for human contributors and AI coding agents alike. [README.md](README.md) explains what ShopSense is and how to run it. This file is the working rules.

## Commands

```bash
uv sync                                  # install (exact versions from uv.lock)
uv run pytest -m "not network"           # offline tests, about 4s. Run before every commit
uv run ruff check . && uv run ruff format .
uv run mypy                              # strict: app, frontend, tests, evals, scripts
uv run lint-imports                      # module layering (below)
uv run python -m scripts.smoke_test      # one live agent turn (real provider, a fraction of a cent)
```

Commit hooks run all of this. Install them once with `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`.

## Module layering

Each package may import only from packages *below* it. `lint-imports` enforces this in the hooks and in CI.

```
app.main                                   app startup, routes, middleware, probes
app.routes                                 HTTP endpoints and the error format
app.agent                                  the ReAct loop, prompt, guardrails, request check, suggestions
app.tools                                  tools and the tool registry
app.llm                                    LLM adapters, types, errors
app.db | app.tracing | app.request_context independent of each other
app.config                                 settings
```

- **Only `app.llm` imports a provider SDK (`openai`).** Everything else goes through `LLMAdapter`, so a provider change never touches the agent, the tools or the routes.
- **The UI (`frontend/`) talks to the backend only over HTTP.** It never imports `app`.
- **If you need an upward import, the design is wrong.** Pass the dependency in instead: the agent takes its models as arguments (`run_agent(..., llm=..., small_llm=...)`).

## Code conventions

- **Types.** mypy `--strict` must pass.
  - Use `JSONObject` (`app.llm.types`) for free-form JSON.
  - Use TypedDicts for messages (`ChatMessage`) and database rows (`MessageRow`, `CartItemRow`).
  - Use Pydantic models at every boundary: settings, API, tool inputs.
- **Docstrings.** Every public module, class and function needs one; ruff checks this. Say what it does and why, not how.
- **No `print`** in `app/` or `frontend/`. Use `logging.getLogger("shopsense....")`. Logs are JSON lines and carry the request id automatically. Only `evals/` and `scripts/` print, because their output is the point.
- **Errors:**
  - **Tools never raise.** They return `{"error", "error_type", "hint"}`, where the hint tells the agent what to do next.
  - **LLM failures** are the four classes in `app/llm/errors.py`, never raw SDK exceptions.
  - **API errors** use `api_error(status, code, message)` and the one error shape. Internals go to the log, never to the response.
- **Comments** explain *why*: a constraint, a measured number, or the issue it fixes (the `SR-xx` ids refer to the project's spec review).

## Tests

- **Offline and hermetic.** Tests use dummy keys, tracing off, and a temporary SQLite file each (`tests/conftest.py`). Tests that need the internet are marked `@pytest.mark.network`; CI doesn't run them.
- **No real LLM calls.** Use the scripted `FakeLLM` in `tests/test_agent.py` and pass it as `llm=` and `small_llm=`. A safety net fails any test whose agent tries to create a real adapter.
- **Stub at the boundary.** Replace `search._ddgs_text`, `extract._getaddrinfo`, or the HTTP transport (`httpx.MockTransport`), not the code under test.
- **Coverage must stay at or above 85%** (currently about 92%). New behaviour gets a test that fails without the change: check by reverting the change once.
- **Live checks:** `scripts/smoke_test.py` for one turn. Run the evals when changing the prompt, the model, or anything that shapes tool output:
  - `evals/prompt_injection.py`
  - `evals/request_check.py`

## Tracing (Langfuse)

- **One trace per chat turn:** the `run-agent` root, with `session_id` and `request_id` on every observation. Follow-up suggestions are a separate `suggest-follow-ups` trace in the same session.
- **One generation per LLM attempt,** through `GenerationTrace` (`app/tracing/generation.py`): opened before the call, completed after. Tracing must never break a request, so every Langfuse call in it is guarded.
- **Generation names are stable.** Dashboards and evaluators filter on them, so don't rename them casually: `generate-agent-response`, `answer-at-limit`, `generate-suggestions`, `check-user-request`.
- **Add `trace_metadata`** with `step` and `operation` on every LLM call.

## Security invariants

Don't weaken these without an explicit decision, and record one in `docs/adr/`:

- **Web text is untrusted.**
  - Anything from a web page or search result goes through `clean_text()`, and successful results carry `WEB_CONTENT_NOTICE`.
  - Answers never contain images; `_without_images` in `app/agent/core.py` enforces this.
- **Changes need the user's say-so.** A tool call that changes stored data must be in `_CHANGES` in `app/agent/request_check.py`. The check sees only the user's message and the previous reply, never tool results.
- **Only public addresses are fetched.** URLs are fetched only through `extract._fetch_html`. It resolves the host, blocks non-public addresses, connects to the checked IP, and re-checks every redirect.
- **Secrets stay secret.**
  - Keys are `SecretStr`. Call `.get_secret_value()` only where the key is sent.
  - Never log settings objects' secret values, never commit `.env`, and never paste keys anywhere.
- **Turns are bounded.** Every turn stays within `MAX_AGENT_STEPS`, `MAX_TURN_TOKENS` and `MAX_TURN_COST_USD`.

## Workflow

- **`main` is protected.** Work on a branch and open a pull request. Both CI jobs (`checks`, `secrets`) must pass, and there's no bypass.
- **Conventional Commits.** Commit messages and PR titles follow them: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`, `perf:`, `build:`. A commit-msg hook checks this.
- **Keep docs in step with changes:**
  - [CHANGELOG.md](CHANGELOG.md) for user-visible changes
  - [README.md](README.md) for setup, configuration or API changes
  - a short ADR in [docs/adr/](docs/adr/) for design decisions
- **Dependencies** go in `pyproject.toml`, then run `uv lock`. The commit hook regenerates `requirements.txt`.
