import re

from langfuse import propagate_attributes

from app.db.models import MessageRow
from app.llm.adapter import LLMAdapter, get_small_llm_adapter
from app.llm.types import ChatMessage
from app.tracing.langfuse_setup import get_langfuse

SUGGESTIONS_SYSTEM_PROMPT = """You are a follow-up suggestion generator for a shopping assistant.
Given the conversation so far, generate 2-3 concise follow-up suggestions the user is likely to send next.

Rules:
- Each suggestion: 3-8 words, from the USER's point of view
- Base suggestions strictly on the conversation context
- Avoid restating completed actions
- Prefer clarifying, expanding, or next-step intents
- No emojis, no quotation marks
- Output as a plain text list, one suggestion per line

Good examples: "Compare these two", "Show me cheaper options", "Add the Sony ones to cart", "Show me waterproof ones"
Bad examples: "Would you like to see more?", "How can I help?", "Let me know if you need anything" """

# "- ", "* ", "• ", "1. ", "2) " at the start of a line
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


async def suggest_follow_ups(session_id: str, history: list[MessageRow], llm: LLMAdapter | None = None) -> list[str]:
    """Suggestions for the conversation so far, asked for by the UI after it
    shows the agent's answer (as Airtap makes them when a task completes), so
    the answer never waits for them. Traced as its own trace in the session."""
    messages: list[ChatMessage] = [
        {"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")
    ]
    langfuse = get_langfuse()
    with (
        propagate_attributes(session_id=session_id, trace_name="suggest-follow-ups"),
        langfuse.start_as_current_observation(as_type="span", name="suggest-follow-ups") as span,
    ):
        span.update(input=messages[-1]["content"] if messages else None)
        suggestions = await generate_suggestions(messages, llm or get_small_llm_adapter())
        span.update(output=suggestions)
    return suggestions


async def generate_suggestions(conversation_messages: list[ChatMessage], llm: LLMAdapter) -> list[str]:
    """
    Generate 2-3 follow-on suggestions based on conversation context.
    Uses the agent's LLM with a minimal prompt. Fails silently — suggestions
    are a UX enhancement, never a blocker.

    Traced as its own `generate-suggestions` generation inside the agent's trace
    (a failure still shows there, with level ERROR).
    """
    try:
        messages: list[ChatMessage] = [
            {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": _format_conversation_for_suggestions(conversation_messages)},
        ]

        response = await llm.chat(  # No tools needed
            messages,
            name="generate-suggestions",
            trace_metadata={"operation": "suggestions"},
        )
        if not response.content:
            return []

        # Parse: one suggestion per line, list markers and stray quotes removed
        suggestions = [_LIST_MARKER.sub("", line).strip().strip("\"'") for line in response.content.strip().split("\n")]
        return [s for s in suggestions if s][:3]  # Cap at 3

    except Exception:  # noqa: BLE001 - suggestions are best-effort, never fail the main response
        return []


def _format_conversation_for_suggestions(messages: list[ChatMessage]) -> str:
    """Format the last few messages for the suggestion prompt."""
    # Only send the last 6 messages to keep it cheap
    recent = messages[-6:] if len(messages) > 6 else messages
    lines = []
    for m in recent:
        role, content = m["role"], m["content"]
        if role in ("user", "assistant") and content:
            lines.append(f"{role.capitalize()}: {content[:300]}")
    return "\n".join(lines)
