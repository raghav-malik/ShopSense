from pathlib import Path
from typing import Literal, Self

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root, not the CWD, so uvicorn, streamlit and
# pytest all find it no matter where they're launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["groq", "openai", "gemini"]
LLMApi = Literal["chat_completions", "responses"]

# Per-provider defaults: (base_url, model, reasoning_effort).
# OpenAI's GPT-6 models reject function tools in Chat Completions unless
# reasoning_effort is "none", and the agent always sends tools.
# Gemini: Google's OpenAI-compatible endpoint; gemini-3.8-flash is the current
# stable Flash model (2.0 is shut down, 2.5 closed to new projects). Thinking
# can't be turned off on Gemini 3, so the model's own default level applies.
PROVIDER_DEFAULTS: dict[str, tuple[str, str, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", None),
    "openai": ("https://api.openai.com/v1", "gpt-6-luna", "none"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash", None),
}
# The Responses API allows tools *with* reasoning. "medium" is OpenAI's default
# and, in testing on gpt-6-luna, the lowest level that reliably produced
# reasoning summaries ("low" often skipped reasoning on simple steps).
RESPONSES_DEFAULT_REASONING_EFFORT = "medium"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # LLM — every provider speaks the OpenAI Chat Completions API; OpenAI also the Responses API
    llm_provider: LLMProvider = Field(
        default="openai", description="Which LLM provider to call: 'openai' (default), 'groq' or 'gemini'"
    )
    llm_api: LLMApi = Field(
        default="chat_completions",
        description="'chat_completions' (default) or 'responses' (OpenAI only: reasoning together with tools)",
    )
    groq_api_key: str | None = Field(
        default=None, description="Groq API key from console.groq.com (required when LLM_PROVIDER=groq)"
    )
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key from platform.openai.com (required when LLM_PROVIDER=openai)"
    )
    # AliasChoices: accepted names in the environment. The Pydantic mypy plugin
    # can't check aliased fields in __init__, but Settings is only ever filled
    # from the environment, so that check doesn't apply here.
    gemini_api_key: str | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="Gemini API key from aistudio.google.com (required when LLM_PROVIDER=gemini)",
    )
    llm_model: str | None = Field(
        default=None,
        description="Model ID; defaults per provider (gpt-6-luna on OpenAI, gpt-oss-120b on Groq, gemini-3.8-flash on Gemini)",
    )
    llm_base_url: str | None = Field(default=None, description="OpenAI-compatible base URL; defaults per provider")
    llm_reasoning_effort: str | None = Field(
        default=None, description="reasoning effort sent to the model; defaults per provider and API"
    )
    llm_max_tokens: int = Field(
        default=4096, description="Max tokens per LLM response (max_completion_tokens / max_output_tokens)"
    )
    llm_temperature: float = Field(default=0.3, description="Lower = more deterministic tool selection")
    llm_timeout: float = Field(default=60.0, description="Seconds to wait for one LLM response before retrying once")

    # Langfuse
    langfuse_public_key: str = Field(..., description="Langfuse public key")
    langfuse_secret_key: str = Field(..., description="Langfuse secret key")
    langfuse_base_url: str = Field(  # type: ignore[pydantic-alias]  # see gemini_api_key
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
    langfuse_timeout: int = Field(
        default=20,
        # The SDK default (5s) covers the whole export including retries, so one
        # slow response drops the batch. Export runs on a background thread, so a
        # longer timeout never delays a request.
        description="Seconds allowed to export a batch of trace data to Langfuse",
    )

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

    @model_validator(mode="after")
    def apply_provider_defaults(self) -> Self:
        base_url, model, reasoning_effort = PROVIDER_DEFAULTS[self.llm_provider]
        if self.llm_api == "responses":
            reasoning_effort = RESPONSES_DEFAULT_REASONING_EFFORT
        self.llm_base_url = self.llm_base_url or base_url
        self.llm_model = self.llm_model or model
        self.llm_reasoning_effort = self.llm_reasoning_effort or reasoning_effort

        # Fail at startup rather than on the agent's first call.
        if not self.llm_api_key:
            raise ValueError(f"{self.llm_provider.upper()}_API_KEY is required when LLM_PROVIDER={self.llm_provider}")
        if self.llm_api == "responses" and self.llm_provider != "openai":
            raise ValueError("LLM_API=responses is only implemented for LLM_PROVIDER=openai")
        if self.llm_api == "chat_completions" and self.llm_provider == "openai" and self.llm_reasoning_effort != "none":
            raise ValueError(
                "OpenAI Chat Completions only allows function tools with LLM_REASONING_EFFORT=none "
                f"(got {self.llm_reasoning_effort!r}); the agent always sends tools. "
                "Use LLM_API=responses for reasoning with tools."
            )
        return self

    @property
    def llm_api_key(self) -> str | None:
        return {
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
        }[self.llm_provider]


# Singleton — import this everywhere
settings = Settings()
