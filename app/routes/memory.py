"""What ShopSense remembers about the user, and the controls to forget it.

Preferences (saved on request), memories (facts learned from what the user
said) and past-session summaries all go into every new chat's system prompt,
so the user can see each of them and delete it, or clear everything.
"""

from typing import Any

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.db import queries
from app.db.models import EpisodeOutcome, MemoryCategory
from app.routes.errors import ErrorResponse, api_error

router = APIRouter(prefix="/memory", tags=["memory"])

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
