"""Live smoke test: one real agent turn against the configured LLM provider.

Uses a throwaway database (your shopsense.db is untouched), costs a fraction of
a cent on gpt-6-luna, and sends a real trace to Langfuse. For checking a new
key, model or provider end to end; the test suite never calls a real provider.

    uv run python -m scripts.smoke_test ["your message"]
"""

import asyncio
import io
import sys
import tempfile
from pathlib import Path

from app.agent.core import run_agent
from app.config import settings
from app.db import queries
from app.db.database import close_db, init_db
from app.tracing.langfuse_setup import shutdown_langfuse


async def smoke_test(message: str) -> None:
    """Run one agent turn in a temporary database and print the response."""
    with tempfile.TemporaryDirectory() as tmp:
        settings.db_path = str(Path(tmp) / "smoke.db")
        await init_db()
        try:
            session = await queries.create_session()
            result = await run_agent(session.id, message)
            print(result.model_dump_json(indent=2))
        finally:
            await close_db()


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(smoke_test(sys.argv[1] if len(sys.argv) > 1 else "find me wireless earbuds under 3000"))
    finally:
        shutdown_langfuse()  # send the trace before the script exits
