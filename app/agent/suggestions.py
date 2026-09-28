import re

from app.llm.adapter import get_llm_adapter

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


async def generate_suggestions(conversation_messages: list[dict]) -> list[str]:
    """
    Generate 2-3 follow-on suggestions based on conversation context.
    Uses the same LLM but with a minimal prompt. Fails silently — suggestions
    are a UX enhancement, never a blocker.

    Traced as its own `generate-suggestions` generation inside the agent's trace
    (a failure still shows there, with level ERROR).
    """
    try:
        llm = get_llm_adapter()
        messages = [
            {"role": "system", "content": SUGGESTIONS_SYSTEM_PROMPT},
            {"role": "user", "content": _format_conversation_for_suggestions(conversation_messages)},
        ]

        response = await llm.chat(  # No tools needed
            messages, name="generate-suggestions", trace_metadata={"operation": "suggestions"},
        )
        if not response.content:
            return []

        # Parse: one suggestion per line, list markers and stray quotes removed
        suggestions = [
            _LIST_MARKER.sub("", line).strip().strip("\"'")
            for line in response.content.strip().split("\n")
        ]
        return [s for s in suggestions if s][:3]  # Cap at 3

    except Exception:
        return []  # Suggestions are best-effort, never fail the main response


def _format_conversation_for_suggestions(messages: list[dict]) -> str:
    """Format the last few messages for the suggestion prompt."""
    # Only send the last 6 messages to keep it cheap
    recent = messages[-6:] if len(messages) > 6 else messages
    lines = []
    for m in recent:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            lines.append(f"{role.capitalize()}: {content[:300]}")
    return "\n".join(lines)
