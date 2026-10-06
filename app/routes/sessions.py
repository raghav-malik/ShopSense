"""Session endpoints: start a chat, list past chats, rename and delete them."""

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query, Response
from pydantic import BaseModel, Field

from app.agent.memory import summarize_pending_sessions
from app.db import queries
from app.db.models import ChatListRow
from app.routes.errors import ErrorResponse, api_error

router = APIRouter(prefix="/sessions", tags=["sessions"])

_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse, "description": "Session not found"}}


class CreateSessionResponse(BaseModel):
    """A newly created session."""

    session_id: str
    created_at: str


@router.post("", status_code=201)
async def create_session(background_tasks: BackgroundTasks) -> CreateSessionResponse:
    """Create a new shopping session. Earlier sessions without a summary are
    summarized in the background, after the response is sent."""
    session = await queries.create_session()
    background_tasks.add_task(summarize_pending_sessions, session.id)
    return CreateSessionResponse(session_id=session.id, created_at=session.created_at)


class ChatSummary(BaseModel):
    """A chat in the sidebar list. Times are UTC ISO 8601; `updated_at` is the last activity."""

    id: str
    title: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: ChatListRow) -> "ChatSummary":
        """The list row, with "New chat" for a chat that has no title yet."""
        return cls(
            id=row["id"], title=row["title"] or "New chat", created_at=row["created_at"], updated_at=row["updated_at"]
        )


class ChatList(BaseModel):
    """One page of chats, most recently active first, and how many match in total."""

    items: list[ChatSummary]
    total: int


@router.get("")
async def list_chats(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    q: Annotated[str, Query(max_length=200, description="Case-insensitive search in chat titles")] = "",
) -> ChatList:
    """The user's chats (those with at least one message), most recently active first."""
    rows, total = await queries.list_chats(limit=limit, offset=offset, query=q)
    return ChatList(items=[ChatSummary.from_row(r) for r in rows], total=total)


class RenameRequest(BaseModel):
    """A new title for a chat."""

    title: str = Field(min_length=1, max_length=100)


@router.patch("/{session_id}", responses=_NOT_FOUND)
async def rename_chat(session_id: str, request: RenameRequest) -> ChatSummary:
    """Rename a chat. A title the user chose is never replaced by a generated one."""
    title = " ".join(request.title.split())
    if not title:
        raise api_error(400, "empty_title", "Title cannot be empty")
    session = await queries.get_session(session_id)
    if session is None or not await queries.set_session_title(session_id, title, "user"):
        raise api_error(404, "session_not_found", f"Session {session_id} not found")
    return ChatSummary(id=session.id, title=title, created_at=session.created_at, updated_at=session.updated_at)


@router.delete("/{session_id}", status_code=204, responses=_NOT_FOUND)
async def delete_chat(session_id: str) -> Response:
    """Remove a chat from the list. What was learned from it stays in memory,
    where it can be seen and deleted separately."""
    if not await queries.delete_session(session_id):
        raise api_error(404, "session_not_found", f"Session {session_id} not found")
    return Response(status_code=204)
