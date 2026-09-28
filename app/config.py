from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root, not the CWD, so uvicorn, streamlit and
# pytest all find it no matter where they're launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Groq / LLM
    groq_api_key: str = Field(..., description="Groq API key from console.groq.com")
    llm_model: str = Field(default="openai/gpt-oss-120b", description="Model ID on Groq")
    llm_base_url: str = Field(default="https://api.groq.com/openai/v1", description="OpenAI-compatible base URL")
    llm_max_tokens: int = Field(default=4096, description="Max tokens per LLM response")
    llm_temperature: float = Field(default=0.3, description="Lower = more deterministic tool selection")

    # Langfuse
    langfuse_public_key: str = Field(..., description="Langfuse public key")
    langfuse_secret_key: str = Field(..., description="Langfuse secret key")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        # Current SDK name is LANGFUSE_BASE_URL; LANGFUSE_HOST is the legacy one.
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
        description="Langfuse URL (EU: cloud.langfuse.com, US: us.cloud.langfuse.com)",
    )
    langfuse_tracing_environment: str = Field(
        default="development",
        description="Keeps local test traces out of production dashboards and evals",
    )
    langfuse_release: str | None = Field(default=None, description="Git SHA or release tag attached to traces")

    # Database
    db_path: str = Field(default="shopsense.db", description="SQLite database file path")

    # Agent
    max_agent_steps: int = Field(default=10, description="Max ReAct iterations per turn (guardrail)")
    max_search_results: int = Field(default=5, description="Default number of search results per query")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        # Tolerate unrelated keys in .env (e.g. LANGFUSE_DEBUG) instead of failing startup.
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()
