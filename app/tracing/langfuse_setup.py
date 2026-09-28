"""Langfuse client for the whole process.

Built eagerly at import, from `settings`: pydantic-settings does not export .env
values to os.environ, and `@observe` / `get_client()` fall back to an unconfigured
client (a silent no-op) if no client exists when the first traced call runs.
Import this module before anything traced.
"""

import re
from collections.abc import Mapping

from langfuse import Langfuse
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanIdentifier, OtelSpanPatch

from app.config import settings

# Users type contact details into chat, and scraped product pages can carry
# them too. Redact before anything leaves the process.
_REDACTIONS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk)-lf-[\w-]{8,}\b"), "[LANGFUSE_KEY]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "[GROQ_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[GOOGLE_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[API_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer [TOKEN]"),
]


def _configured_secrets() -> list[str]:
    """The keys this process holds, whatever their format: the patterns above
    only know today's key formats (Google's newer Gemini keys don't start with
    'AIza'). Longest first, so a key that contains another is replaced whole."""
    values = [settings.openai_api_key, settings.groq_api_key, settings.gemini_api_key, settings.langfuse_secret_key]
    return sorted({s.get_secret_value() for s in values if s is not None and len(s) >= 8}, key=len, reverse=True)


def _redact(text: str) -> str:
    for secret in _configured_secrets():
        text = text.replace(secret, "[SECRET]")
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    """Redact emails and API keys from every span before it leaves the process."""
    patches: dict[OtelSpanIdentifier, OtelSpanPatch] = {}
    for identifier, span in params.spans.items():
        replacements: dict[str, str] = {}
        # OpenTelemetry declares AttributeValue with a chained assignment
        # (`AnyValue = AttributeValue = str | ...`), which type checkers can't read
        # as an alias; `object` is the honest type here. Values can be text,
        # numbers, booleans or sequences, and only text can hold PII.
        attributes: Mapping[str, object] = span.attributes
        for key, value in attributes.items():
            if isinstance(value, str) and (masked := _redact(value)) != value:
                replacements[key] = masked
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


_langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key.get_secret_value(),
    base_url=settings.langfuse_base_url,
    environment=settings.langfuse_tracing_environment,
    release=settings.langfuse_release,
    timeout=settings.langfuse_timeout,
    mask_otel_spans=mask_otel_spans,
)


def get_langfuse() -> Langfuse:
    """Get the Langfuse client singleton."""
    return _langfuse


def flush_langfuse() -> None:
    """Send buffered traces now. For scripts and tests that keep running."""
    _langfuse.flush()


def shutdown_langfuse() -> None:
    """Flush and stop background exporters. Call once at app shutdown."""
    _langfuse.shutdown()
