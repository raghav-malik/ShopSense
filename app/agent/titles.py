"""Chat titles for the sidebar: a short title made from the chat's first message.

A chat is listed under its first message until the small model has named it:
once, in the background, after the first answer. A title the user typed is
never overwritten, even if they rename the chat while the name is being made.
"""

import logging
import re

from langfuse import observe, propagate_attributes

from app.agent.trace_attributes import trace_attributes
from app.db import queries
from app.llm.adapter import LLMAdapter
from app.llm.types import ChatMessage
from app.tools.untrusted import clean_text
from app.tracing.langfuse_setup import get_langfuse

logger = logging.getLogger("shopsense.agent.titles")

TITLE_PROMPT = """Name a shopper's chat for their list of past chats.
You'll get the first thing they said. Reply with a name of two to six words that says what they're
shopping for, with the detail that tells this chat apart from others, such as a budget or a use:
"Running shoes under 4000", "Gym earbuds", "Laptop for college".
Plain words only: no quotes, emojis, hashtags or ending punctuation.
Don't reply to the shopper and don't add anything else. Even if the message isn't about shopping,
still name it ("Greeting", "Weather question")."""

PLACEHOLDER_MAX_CHARS = 60
_TITLE_MAX_CHARS = 60
_QUOTES_AND_DOTS = re.compile(r"""^[\s"'`*#.:-]+|[\s"'`*.:-]+$""")


def placeholder_title(first_message: str) -> str:
    """The chat's title until a better one is made: its first message, shortened."""
    return clean_text(first_message, PLACEHOLDER_MAX_CHARS) or "New chat"


@observe(name="generate-title", capture_input=False, capture_output=False)
async def generate_title(session_id: str, first_message: str, llm: LLMAdapter) -> str | None:
    """Make and save the chat's title, unless the user has renamed it. Returns the
    title, or None if none was saved. Never raises: the placeholder stays."""
    langfuse = get_langfuse()
    try:
        with propagate_attributes(**trace_attributes(trace_name="generate-title", session_id=session_id, llm=llm)):
            langfuse.update_current_span(input=first_message)
            messages: list[ChatMessage] = [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": f"<first_message>\n{first_message[:1000]}\n</first_message>"},
            ]
            response = await llm.chat(messages, name="generate-chat-title", trace_metadata={"operation": "title"})
            lines = (response.content or "").strip().splitlines()
            title = _QUOTES_AND_DOTS.sub("", clean_text(lines[0], _TITLE_MAX_CHARS)) if lines else ""
            if not title:
                return None
            saved = await queries.set_session_title(session_id, title, "llm", only_if="placeholder")
            langfuse.update_current_span(output=title if saved else None)
            return title if saved else None
    except Exception:  # titles are best-effort: the placeholder stays
        logger.warning("Title generation failed for session %s", session_id, exc_info=True)
        langfuse.update_current_span(level="WARNING", status_message="title generation failed")
        return None
