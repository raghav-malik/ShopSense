"""Provider-agnostic LLM errors.

Adapters translate their SDK's exceptions into these, so the agent and the
API layer never import provider SDKs. Each class carries the stable error
`code` the API returns.
"""


class LLMError(Exception):
    """The LLM call failed in a way retrying won't fix (e.g. a rejected request)."""

    code = "llm_error"

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMRateLimitError(LLMError):
    """Rate limited, and the wait asked for is too long to hold the request open."""

    code = "llm_rate_limited"


class LLMTimeoutError(LLMError):
    """The provider didn't answer in time, even after a retry."""

    code = "llm_timeout"


class LLMUnavailableError(LLMError):
    """Can't use the model: it doesn't exist, the key is rejected, or the provider is down."""

    code = "llm_unavailable"
