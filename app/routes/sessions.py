"""Session endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.db import queries

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    """A newly created session."""

    session_id: str
    created_at: str


@router.post("", status_code=201)
async def create_session() -> CreateSessionResponse:
    """Create a new shopping session."""
    session = await queries.create_session()
    return CreateSessionResponse(session_id=session.id, created_at=session.created_at)
