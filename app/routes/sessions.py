"""Session endpoints."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from app.agent.memory import summarize_pending_sessions
from app.db import queries

router = APIRouter(prefix="/sessions", tags=["sessions"])


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
