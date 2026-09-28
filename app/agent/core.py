"""The ReAct agent loop — the brain of the entire system.

Trace shape for one user message (one trace per message, one Langfuse session
per ShopSense session):

    run-agent                    agent       input: user message · output: reply
    ├── generate-agent-response  generation  model, tokens, cost, reasoning
    ├── search_products          retriever   input: tool args · metadata: the LLM's reasoning
    ├── generate-agent-response  generation
    ├── extract_product_info     retriever
    ├── generate-agent-response  generation  (final answer)
    └── generate-suggestions     generation

Observation names are referenced by Langfuse evaluators and dashboards; keep them stable.
"""

import asyncio
import json

from langfuse import observe, propagate_attributes

# Imported first: creates the Langfuse client before any traced call runs.
from app.tracing.langfuse_setup import get_langfuse

from app.agent.prompts import build_system_prompt
from app.agent.schemas import AgentResponse
from app.agent.suggestions import generate_suggestions
from app.config import settings
from app.db import queries
from app.db.models import Message
from app.llm.adapter import get_llm_adapter
from app.llm.errors import LLMError
from app.tools.registry import TOOL_MAP, execute_tool, get_tool_schemas

langfuse = get_langfuse()

# Lookups that don't change state are retrievers; everything else (cart and
# preference writes, the comparison transform) is a plain tool.
TOOL_OBSERVATION_TYPES = {
    "search_products": "retriever",
    "extract_product_info": "retriever",
}


@observe(name="run-agent", as_type="agent", capture_input=False, capture_output=False)
async def run_agent(session_id: str, user_message: str) -> AgentResponse:
    """
    Main agent loop. Implements the ReAct pattern:
    1. Load context (history, preferences, cart)
    2. Build messages array with system prompt
    3. Loop: LLM call → tool dispatch → observe → repeat until done
    4. Return final response with metadata
    """
    # Trace input is the user's message only, not every function argument.
    langfuse.update_current_span(input=user_message)

    # Everything inside (generations, tools, suggestions) inherits the session.
    with propagate_attributes(session_id=session_id, trace_name="run-agent"):
        result = await _run_agent(session_id, user_message)

    langfuse.update_current_span(
        output=result.response,
        metadata={
            "step_count": result.step_count,
            "tool_calls_made": result.tool_calls_made,
            "total_tokens": result.total_tokens,
            "suggestions": result.suggestions,
        },
    )
    return result


async def _run_agent(session_id: str, user_message: str) -> AgentResponse:
    # 1. Load session context
    session = await queries.get_session(session_id)
    if not session:
        langfuse.update_current_span(level="WARNING", status_message="session not found")
        return AgentResponse(response="Session not found. Please create a new session.", step_count=0)

    history = await queries.get_messages(session_id)
    preferences = await queries.get_all_preferences()
    cart = await queries.get_cart(session_id)

    # 2. Build messages array
    system_prompt = build_system_prompt(preferences, cart, session.budget)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_replay_history(history))

    # Add the new user message
    messages.append({"role": "user", "content": user_message})

    # Save the user message to DB
    await queries.save_message(Message(
        session_id=session_id, role="user", content=user_message,
    ))

    # 3. ReAct loop
    llm = get_llm_adapter()
    tools_schema = get_tool_schemas()
    tools_called: list[str] = []
    products_found: list[dict] = []
    total_tokens = 0
    step_count = 0

    for step in range(settings.max_agent_steps):
        step_count = step + 1

        # LLM call — traced as a generation inside the adapter
        response = await llm.chat(messages, tools_schema, name="generate-agent-response")
        total_tokens += response.usage.get("total_tokens", 0)

        # Check if the LLM wants to respond (no tool calls)
        if response.finish_reason == "stop" or not response.tool_calls:
            final_text = response.content or "I couldn't find a good answer. Could you rephrase?"

            # Save assistant response
            await queries.save_message(Message(
                session_id=session_id, role="assistant", content=final_text,
                token_count=response.usage.get("total_tokens", 0),
            ))

            # The reply goes into the context the suggestions are generated from;
            # otherwise they'd follow up on the turn *before* this answer.
            messages.append({"role": "assistant", "content": final_text})

            # Generate follow-on suggestions (best-effort, from Airtap pattern)
            suggestions = await generate_suggestions(messages)

            return AgentResponse(
                response=final_text,
                tool_calls_made=tools_called,
                products_found=products_found,
                suggestions=suggestions,
                trace_url=await _current_trace_url(),
                step_count=step_count,
                total_tokens=total_tokens,
            )

        # Tool dispatch — process each tool call
        # Add the assistant message with tool_calls
        assistant_msg = {"role": "assistant", "content": response.content, "tool_calls": []}
        for tc in response.tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
        messages.append(assistant_msg)

        for tc in response.tool_calls:
            tool_name = tc.function.name
            tools_called.append(tool_name)

            # Execute the tool
            result = await _execute_tool_traced(tool_name, tc.function.arguments, session_id, step)

            # Track products found
            if tool_name == "search_products":
                try:
                    parsed = json.loads(result)
                    for p in parsed.get("results", []):
                        products_found.append(p)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tc.id,
            })

            # Save tool result to DB
            await queries.save_message(Message(
                session_id=session_id, role="tool", content=result,
                tool_name=tool_name, tool_call_id=tc.id,
            ))

    # Guardrail: max steps reached. Rather than discarding the research, make one
    # last call with tools disabled so the model answers from what it gathered.
    langfuse.update_current_span(level="WARNING", status_message=f"max_agent_steps ({settings.max_agent_steps}) reached")
    final_text, final_tokens = await _answer_from_research(llm, messages, tools_schema, products_found)
    total_tokens += final_tokens
    await queries.save_message(Message(
        session_id=session_id, role="assistant", content=final_text,
    ))
    messages.append({"role": "assistant", "content": final_text})
    return AgentResponse(
        response=final_text, tool_calls_made=tools_called,
        products_found=products_found, suggestions=await generate_suggestions(messages),
        trace_url=await _current_trace_url(),
        step_count=step_count, total_tokens=total_tokens,
    )


STEP_LIMIT_NOTE = (
    "You've reached the research limit for this turn and can't call more tools. "
    "Answer now using only what the tool results above already show: recommend "
    "the best options you found (with prices and links where you have them), and "
    "say briefly what you couldn't confirm."
)


async def _answer_from_research(llm, messages: list[dict], tools_schema: list[dict], products_found: list[dict]) -> tuple[str, int]:
    """Final answer after hitting max_agent_steps. Falls back to listing the
    search results if even this call fails, so the user always gets something."""
    try:
        response = await llm.chat(
            [*messages, {"role": "system", "content": STEP_LIMIT_NOTE}],
            # Tools stay in the request (the history references them) but can't be called.
            tools_schema, tool_choice="none", name="answer-at-step-limit",
        )
        if response.content:
            return response.content, response.usage.get("total_tokens", 0)
    except LLMError:
        pass
    return _results_fallback(products_found), 0


def _results_fallback(products_found: list[dict]) -> str:
    lines = [
        f"- [{p.get('title') or p['url']}]({p['url']})"
        for p in products_found[:5] if p.get("url")
    ]
    if not lines:
        return "I couldn't finish researching this one. Could you tell me more about what you're looking for, such as a budget, brand, or must-have feature?"
    return (
        "I ran out of research steps before I could make a confident recommendation. "
        "Here are the most relevant results I found:\n\n" + "\n".join(lines)
        + "\n\nTell me which one interests you, or narrow it down (budget, brand, features) and I'll dig deeper."
    )


def _replay_history(history: list[dict]) -> list[dict]:
    """Past turns as plain user/assistant text.

    The DB stores tool results but not the assistant messages that requested
    them, so replaying tool rows would send tool results with no matching
    tool_calls — invalid in the OpenAI message format. Final answers already
    carry the products, prices and links, and dropping raw tool JSON keeps each
    request small (cheaper, faster, and inside tight limits like Groq's free tier).

    A history window can also start mid-turn (it's the newest N rows), so drop
    anything before the first user message.
    """
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


async def _execute_tool_traced(name: str, arguments: str, session_id: str, step: int) -> str:
    """Execute a tool, traced as a Langfuse tool/retriever observation.

    The LLM's `reasoning` argument never reaches the tool (the registry strips
    it); it's recorded here as metadata, so the trace shows why each tool was called.
    """
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = None
    reasoning = args.pop("reasoning", None) if isinstance(args, dict) else None

    with langfuse.start_as_current_observation(
        as_type=TOOL_OBSERVATION_TYPES.get(name, "tool"),
        # Unknown names come from the LLM; don't let them mint new observation names.
        name=name if name in TOOL_MAP else "unknown-tool",
        input=args if args is not None else arguments,
        metadata={"reasoning": reasoning, "step": step, "requested_tool": name},
    ) as observation:
        result = await execute_tool(name, arguments, session_id)
        try:
            output = json.loads(result)
        except json.JSONDecodeError:
            output = result
        error = output.get("error") if isinstance(output, dict) else None
        observation.update(
            output=output,
            # Tool errors go back to the LLM to self-correct, so they're warnings,
            # not failures — but they should stand out in the trace.
            level="WARNING" if error else None,
            status_message=str(error) if error else None,
        )
    return result


async def _current_trace_url() -> str | None:
    """Link to this trace in Langfuse. The first call does a blocking API lookup
    of the project id (then it's cached), so it runs off the event loop."""
    trace_id = langfuse.get_current_trace_id()
    if not trace_id:
        return None
    try:
        return await asyncio.to_thread(langfuse.get_trace_url, trace_id=trace_id)
    except Exception:
        return None


if __name__ == "__main__":
    # Smoke test against a throwaway DB (your real shopsense.db is untouched).
    # Sends a real trace to Langfuse. From the project root:  python -m app.agent.core
    import sys
    import tempfile
    from pathlib import Path

    from app.db.database import close_db, init_db
    from app.tracing.langfuse_setup import shutdown_langfuse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def _smoke_test() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings.db_path = str(Path(tmp) / "agent_smoke.db")
            await init_db()
            try:
                await queries.create_session("test-session-id")
                result = await run_agent("test-session-id", "find me wireless earbuds under 3000")
                print(result.model_dump_json(indent=2))
            finally:
                await close_db()

    try:
        asyncio.run(_smoke_test())
    finally:
        shutdown_langfuse()  # send the trace before the script exits
