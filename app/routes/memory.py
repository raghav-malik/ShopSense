"""What ShopSense remembers about the user, and the controls to forget it.

Preferences (saved on request), memories (facts learned from what the user
said) and past-session summaries all go into every new chat's system prompt,
so the user can see each of them and delete it, or clear everything.
"""

from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from app.agent import memory_files
from app.db import queries
from app.db.models import EpisodeOutcome, MemoryCategory
from app.routes.errors import ErrorResponse, api_error

router = APIRouter(prefix="/memory", tags=["memory"])


# ---- memory as markdown files (the Settings view) ----


class MemoryFileOut(BaseModel):
    """One memory file. `updated_at` is UTC ISO 8601 (None when empty); `day` is the
    local day a short-term file's chats happened on."""

    name: str
    kind: memory_files.FileKind
    content: str
    updated_at: str | None
    day: str | None

    @classmethod
    def from_file(cls, file: memory_files.MemoryFile) -> "MemoryFileOut":
        """The file as the API returns it."""
        return cls(
            name=file.name,
            kind=file.kind,
            content=file.content,
            updated_at=file.updated_at,
            day=file.day.isoformat() if file.day else None,
        )


class MemoryFiles(BaseModel):
    """user.md, memory.md, preferences.md and the day files (newest first), and the time zone they use."""

    timezone: str
    files: list[MemoryFileOut]


class SaveFileRequest(BaseModel):
    """A memory file's new content."""

    content: str = Field(max_length=20_000)


_FILE_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Content that can't be saved (too long)"},
    404: {"model": ErrorResponse, "description": "No memory file with that name"},
}


def _file_error(e: memory_files.MemoryFileError) -> Exception:
    return api_error(404 if e.code == "unknown_file" else 400, e.code, str(e))


@router.get("/files")
async def get_memory_files() -> MemoryFiles:
    """Everything remembered, as markdown files."""
    files = await memory_files.list_files()
    return MemoryFiles(timezone=memory_files.timezone_name(), files=[MemoryFileOut.from_file(f) for f in files])


@router.put("/files/{name}", responses=_FILE_ERRORS)
async def save_memory_file(name: str, request: SaveFileRequest) -> MemoryFileOut:
    """Save an edited file and return it as it now reads (memory.md and the day
    files are re-rendered from what was stored)."""
    try:
        return MemoryFileOut.from_file(await memory_files.save_file(name, request.content))
    except memory_files.MemoryFileError as e:
        raise _file_error(e) from e


@router.delete("/files/{name}", status_code=204, responses=_FILE_ERRORS)
async def clear_memory_file(name: str) -> Response:
    """Clear a file: user.md goes back to its template; the others are forgotten."""
    try:
        await memory_files.clear_file(name)
    except memory_files.MemoryFileError as e:
        raise _file_error(e) from e
    return Response(status_code=204)


# ---- individual items ----

_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Nothing to delete with that id"}
}
# More than the prompt uses (15 memories, 3 summaries), so nothing stored is hidden
# from the user, but bounded.
_MAX_MEMORIES = 200
_MAX_EPISODES = 50


class MemoryOut(BaseModel):
    """A fact learned about the user."""

    id: str
    category: MemoryCategory
    content: str
    confidence: float
    updated_at: str


class EpisodeOut(BaseModel):
    """A past session's summary."""

    id: str
    session_id: str
    summary: str
    outcome: EpisodeOutcome | None
    created_at: str


class MemoryOverview(BaseModel):
    """Everything remembered about the user, as the agent sees it."""

    preferences: dict[str, Any]
    memories: list[MemoryOut]
    episodes: list[EpisodeOut]


@router.get("")
async def get_memory() -> MemoryOverview:
    """Preferences, learned facts (strongest first) and past-session summaries (newest first)."""
    memories = await queries.get_all_memories(limit=_MAX_MEMORIES)
    episodes = await queries.get_recent_episodes(limit=_MAX_EPISODES)
    return MemoryOverview(
        preferences=await queries.get_all_preferences(),
        memories=[MemoryOut.model_validate(m) for m in memories],
        episodes=[EpisodeOut.model_validate(e) for e in episodes],
    )


@router.delete("", status_code=204)
async def forget_everything() -> Response:
    """Forget every preference, learned fact and past-session summary. Chats stay."""
    await queries.forget_everything()
    return Response(status_code=204)


@router.delete("/memories/{memory_id}", status_code=204, responses=_NOT_FOUND)
async def forget_memory(memory_id: str) -> Response:
    """Forget one learned fact."""
    if not await queries.delete_memory(memory_id):
        raise api_error(404, "memory_not_found", f"Memory {memory_id} not found")
    return Response(status_code=204)


@router.delete("/episodes/{episode_id}", status_code=204, responses=_NOT_FOUND)
async def forget_episode(episode_id: str) -> Response:
    """Forget one past session's summary; that session won't be summarized again."""
    if not await queries.delete_episode(episode_id):
        raise api_error(404, "episode_not_found", f"Episode {episode_id} not found")
    return Response(status_code=204)


@router.delete("/preferences/{key}", status_code=204, responses=_NOT_FOUND)
async def forget_preference(key: str) -> Response:
    """Forget one saved preference."""
    if not await queries.delete_preference(key):
        raise api_error(404, "preference_not_found", f"Preference {key!r} not found")
    return Response(status_code=204)
