"""Everything ShopSense remembers, as markdown files the user can read and edit.

Each kind of memory is one markdown file with an "updated" time, shown in
Settings, edited as text and saved whole:

- user.md: what the user wrote about themselves. Stored as-is; only the user
  writes it (the agent and the extractor only read it).
- memory.md: facts learned from what the user said, one "- " line each under a
  heading per category. Saving applies the edit: a removed line is forgotten,
  a new line is stored as stated by the user, "(inferred)" marks a fact they
  didn't state outright.
- preferences.md: saved preferences, one "- key: value" line each.
- YYYY-MM-DD.md: the summaries of that day's chats, one section per chat. The
  day is when the chat happened, in the user's time zone (TIMEZONE), not when
  it was summarized. Editing a summary rewrites it; deleting a chat's section
  forgets that summary for good.

memory.md, preferences.md and the day files are views of database rows, made
fresh on every read, so they always match what the agent actually uses.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.clock import local_date, to_local
from app.config import settings
from app.db import queries
from app.db.models import USER_MD_TEMPLATE, ChatEpisodeRow, Memory, MemoryCategory, MemoryRow
from app.tools.untrusted import clean_text

FileKind = Literal["user", "long_term", "preferences", "short_term"]

USER_FILE = "user.md"
LONG_TERM_FILE = "memory.md"
PREFERENCES_FILE = "preferences.md"
_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$")

# Headings in memory.md, one per category, in this order.
CATEGORY_HEADINGS: dict[MemoryCategory, str] = {
    "brand_preference": "Brands you like",
    "brand_dislike": "Brands you avoid",
    "budget_range": "Budget",
    "retailer_preference": "Stores",
    "size_info": "Sizes",
    "category_interest": "Interests",
    "shopping_style": "Shopping style",
    "product_feedback": "Product feedback",
    "general": "Other",
}
_HEADING_TO_CATEGORY: dict[str, MemoryCategory] = {
    **{heading.lower(): category for category, heading in CATEGORY_HEADINGS.items()},
    **{category: category for category in CATEGORY_HEADINGS},
    **{category.replace("_", " "): category for category in CATEGORY_HEADINGS},
}
_INFERRED = " (inferred)"
_INFERRED_CONFIDENCE = 0.7
_INFERRED_BELOW = 0.9  # the same line the system prompt draws

USER_MD_MAX_CHARS = 4000
MAX_FACTS = 200
_FACT_MAX_CHARS = 200
_SUMMARY_MAX_CHARS = 500
# How far back the day files go (the prompt itself only uses the latest 3 chats).
_MAX_EPISODES = 500

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^[-*+]\s+(.*)$")
_PREFERENCE = re.compile(r"^[-*+]\s+([^:]+?)\s*:\s*(.*)$")
_CHAT_MARKER = re.compile(r"<!--\s*chat\s+([0-9a-f]{8})\s*-->", re.IGNORECASE)
_META_LINE = re.compile(r"^_?outcome:", re.IGNORECASE)


class MemoryFileError(ValueError):
    """A file that doesn't exist, or content that can't be saved; the message says which."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class MemoryFile:
    """One memory file as the user sees it."""

    name: str
    kind: FileKind
    content: str
    updated_at: str | None  # UTC ISO 8601; None for an empty file
    day: date | None = None  # short-term files: the day the chats happened (local)


# ---- reading ----


async def list_files() -> list[MemoryFile]:
    """user.md, memory.md, preferences.md, then one file per day with chats, newest day first."""
    return [
        await _user_file(),
        await _long_term_file(),
        await _preferences_file(),
        *[_day_file(day, episodes) for day, episodes in await _episodes_by_day()],
    ]


async def get_file(name: str) -> MemoryFile:
    """One file by name; MemoryFileError("unknown_file") if there's no such file."""
    if name == USER_FILE:
        return await _user_file()
    if name == LONG_TERM_FILE:
        return await _long_term_file()
    if name == PREFERENCES_FILE:
        return await _preferences_file()
    day = _parse_day(name)
    for file_day, episodes in await _episodes_by_day():
        if file_day == day:
            return _day_file(day, episodes)
    raise MemoryFileError("unknown_file", f"No memory file named {name!r}")


async def _user_file() -> MemoryFile:
    profile = await queries.get_user_profile()
    return MemoryFile(USER_FILE, "user", profile["content"], profile["updated_at"])


async def _long_term_file() -> MemoryFile:
    memories = await queries.get_all_memories(limit=MAX_FACTS)
    updated = max((m["updated_at"] for m in memories), default=None)
    return MemoryFile(LONG_TERM_FILE, "long_term", render_long_term(memories), updated)


async def _preferences_file() -> MemoryFile:
    preferences = await queries.get_all_preferences()
    return MemoryFile(
        PREFERENCES_FILE, "preferences", render_preferences(preferences), await queries.preferences_updated_at()
    )


def _day_file(day: date, episodes: list[ChatEpisodeRow]) -> MemoryFile:
    updated = max((e["created_at"] for e in episodes), default=None)
    return MemoryFile(f"{day.isoformat()}.md", "short_term", render_day(day, episodes), updated, day)


async def _episodes_by_day() -> list[tuple[date, list[ChatEpisodeRow]]]:
    """Chat summaries grouped by the local day the chat happened, newest day first,
    oldest chat first within a day."""
    by_day: dict[date, list[ChatEpisodeRow]] = defaultdict(list)
    for episode in await queries.get_recent_episodes(limit=_MAX_EPISODES):
        by_day[local_date(episode["last_active_at"])].append(episode)
    return [
        (day, sorted(episodes, key=lambda e: e["last_active_at"]))
        for day, episodes in sorted(by_day.items(), reverse=True)
    ]


def _parse_day(name: str) -> date:
    match = _DAY_FILE.match(name)
    if not match:
        raise MemoryFileError("unknown_file", f"No memory file named {name!r}")
    try:
        return date.fromisoformat(match.group(1))
    except ValueError as e:
        raise MemoryFileError("unknown_file", f"No memory file named {name!r}") from e


# ---- rendering ----


def render_long_term(memories: list[MemoryRow]) -> str:
    """memory.md: a heading per category that has facts, one "- " line per fact."""
    by_category: dict[str, list[MemoryRow]] = defaultdict(list)
    for memory in memories:
        by_category[memory["category"]].append(memory)
    sections = []
    for category, heading in CATEGORY_HEADINGS.items():
        facts = sorted(by_category.get(category, []), key=lambda m: m["created_at"])
        if facts:
            lines = [f"- {m['content']}{_INFERRED if m['confidence'] < _INFERRED_BELOW else ''}" for m in facts]
            sections.append(f"## {heading}\n" + "\n".join(lines))
    return "\n\n".join(sections) + ("\n" if sections else "")


def render_preferences(preferences: dict[str, object]) -> str:
    """preferences.md: one "- key: value" line per preference (lists and numbers as JSON)."""
    lines = [f"- {key}: {value if isinstance(value, str) else json.dumps(value)}" for key, value in preferences.items()]
    return "\n".join(lines) + ("\n" if lines else "")


def render_day(day: date, episodes: list[ChatEpisodeRow]) -> str:
    """YYYY-MM-DD.md: the day's chats in order, each with its local time, title and summary.

    The `<!-- chat 1a2b3c4d -->` line ties a section to its chat, so an edit or a
    deleted section can be applied; the outcome line is for reading only.
    """
    sections = [f"# {day:%A}, {day.day} {day:%B %Y}"]
    for e in episodes:
        when = to_local(e["last_active_at"])
        title = e["chat_title"] or "Chat"
        details = [f"Outcome: {e['outcome'] or 'unknown'}"]
        if searched := json.loads(e["products_searched"] or "[]"):
            details.append(f"Looked for: {', '.join(searched)}")
        if carted := json.loads(e["products_carted"] or "[]"):
            details.append(f"Carted: {', '.join(carted)}")
        sections.append(
            f"## {when:%I:%M %p}".replace(" 0", " ", 1)
            + f" · {title}\n<!-- chat {e['session_id'][:8]} -->\n{e['summary']}\n_{' · '.join(details)}_"
        )
    return "\n\n".join(sections) + "\n"


# ---- writing ----


async def save_file(name: str, content: str) -> MemoryFile:
    """Apply the user's edited file, then return it as it now reads."""
    if name == USER_FILE:
        await queries.set_user_profile(_clean_user_md(content))
    elif name == LONG_TERM_FILE:
        await _apply_long_term(content)
    elif name == PREFERENCES_FILE:
        await _apply_preferences(content)
    else:
        day = _parse_day(name)
        await _apply_day(day, content, dict(await _episodes_by_day()).get(day))
    return await get_file(name)


async def clear_file(name: str) -> None:
    """Clear a file: user.md goes back to the template; the others are forgotten."""
    if name == USER_FILE:
        await queries.set_user_profile(USER_MD_TEMPLATE)
    elif name == LONG_TERM_FILE:
        for memory in await queries.get_all_memories(limit=10_000):
            await queries.delete_memory(memory["id"])
    elif name == PREFERENCES_FILE:
        for key in await queries.get_all_preferences():
            await queries.delete_preference(key)
    else:
        day = _parse_day(name)
        await _apply_day(day, "", dict(await _episodes_by_day()).get(day))  # raises for a day without chats


def _clean_user_md(content: str) -> str:
    content = content.replace("\x00", "").replace("\r\n", "\n").strip()
    if len(content) > USER_MD_MAX_CHARS:
        raise MemoryFileError("too_long", f"user.md can be at most {USER_MD_MAX_CHARS} characters")
    return content + "\n" if content else USER_MD_TEMPLATE


@dataclass(frozen=True)
class _Fact:
    category: MemoryCategory
    content: str
    inferred: bool


def parse_long_term(content: str) -> list[_Fact]:
    """The facts in memory.md: every "- " line, under the category of the heading
    above it ("Other" before any heading or under one that isn't a category)."""
    facts: dict[tuple[str, str], _Fact] = {}
    category: MemoryCategory = "general"
    for raw in content.splitlines():
        line = raw.strip()
        if heading := _HEADING.match(line):
            category = _HEADING_TO_CATEGORY.get(heading.group(1).strip().lower(), "general")
        elif bullet := _BULLET.match(line):
            text = bullet.group(1).strip()
            inferred = text.lower().endswith(_INFERRED.strip())
            if inferred:
                text = text[: -len(_INFERRED.strip())].strip()
            text = clean_text(text, _FACT_MAX_CHARS)
            if text:
                facts.setdefault((category, text.lower()), _Fact(category, text, inferred))
    return list(facts.values())


async def _apply_long_term(content: str) -> None:
    facts = parse_long_term(content)
    if len(facts) > MAX_FACTS:
        raise MemoryFileError("too_long", f"memory.md can hold at most {MAX_FACTS} facts")
    existing = await queries.get_all_memories(limit=10_000)
    by_key = {(m["category"], m["content"].lower()): m for m in existing}
    by_text = {m["content"].lower(): m for m in existing}
    kept: set[str] = set()
    for fact in facts:
        match = by_key.get((fact.category, fact.content.lower()))
        if match is None and (moved := by_text.get(fact.content.lower())) and moved["id"] not in kept:
            match = moved  # the same fact under another heading: the user moved it
        if match is None:
            confidence = _INFERRED_CONFIDENCE if fact.inferred else 1.0
            await queries.save_memory(Memory(category=fact.category, content=fact.content, confidence=confidence))
            continue
        kept.add(match["id"])
        was_inferred = match["confidence"] < _INFERRED_BELOW
        confidence = match["confidence"] if fact.inferred == was_inferred else (0.7 if fact.inferred else 1.0)
        if (match["category"], match["content"], match["confidence"]) != (fact.category, fact.content, confidence):
            await queries.update_memory(match["id"], fact.category, fact.content, confidence)
    for memory in existing:
        if memory["id"] not in kept and (memory["category"], memory["content"].lower()) not in {
            (f.category, f.content.lower()) for f in facts
        }:
            await queries.delete_memory(memory["id"])


def parse_preferences(content: str) -> dict[str, object]:
    """The "- key: value" lines of preferences.md. A value that reads as JSON
    (a list, a number, true/false) is stored as that; anything else as text."""
    preferences: dict[str, object] = {}
    for raw in content.splitlines():
        if match := _PREFERENCE.match(raw.strip()):
            key, value = clean_text(match.group(1), 60), match.group(2).strip()
            if not key:
                continue
            try:
                preferences[key] = json.loads(value)
            except json.JSONDecodeError:
                preferences[key] = clean_text(value, _FACT_MAX_CHARS)
    return preferences


async def _apply_preferences(content: str) -> None:
    wanted = parse_preferences(content)
    current = await queries.get_all_preferences()
    for key, value in wanted.items():
        if current.get(key, object()) != value:
            await queries.set_preference(key, value)
    for key in current.keys() - wanted.keys():
        await queries.delete_preference(key)


async def _apply_day(day: date, content: str, episodes: list[ChatEpisodeRow] | None) -> None:
    """Rewrite the summaries kept in the day file; forget the chats whose section is gone."""
    if not episodes:
        raise MemoryFileError("unknown_file", f"No memory file named {day.isoformat()}.md")
    sections: dict[str, str] = {}
    marker: str | None = None
    lines: list[str] = []
    for raw in [*content.splitlines(), "## end"]:
        if raw.strip().startswith("## ") or (raw.strip().startswith("# ") and marker is None):
            if marker is not None:
                sections[marker] = clean_text(" ".join(lines), _SUMMARY_MAX_CHARS)
            marker, lines = None, []
        elif found := _CHAT_MARKER.search(raw):
            marker = found.group(1).lower()
        elif marker is not None and raw.strip() and not _META_LINE.match(raw.strip()):
            lines.append(raw.strip())
    for episode in episodes:
        summary = sections.get(episode["session_id"][:8])
        if not summary:
            await queries.delete_episode(episode["id"])
        elif summary != episode["summary"]:
            await queries.update_episode_summary(episode["id"], summary)


def timezone_name() -> str:
    """The time zone the day files and times are in."""
    return settings.timezone
