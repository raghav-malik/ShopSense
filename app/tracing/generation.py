"""Generation lifecycle for LLM calls (pattern from Airtap's omniTracing).

    with GenerationTrace("generate-agent-response", model=..., input=...) as trace:
        try:
            response = await client.call(...)
        except Exception as e:
            trace.error(e)
            raise
        trace.success(lambda: {"output": ..., "usage_details": ...})

The generation is opened *before* the call (true start time, exact input) and
completed *after* it succeeds or fails. Every Langfuse operation, including
building the success payload, is guarded: a tracing bug is logged and never
replaces the real LLM result or error.
"""

import logging
from contextlib import AbstractContextManager
from typing import Any, Callable

from app.tracing.langfuse_setup import get_langfuse

logger = logging.getLogger("shopsense.tracing")


class GenerationTrace:
    def __init__(
        self,
        name: str,
        *,
        model: str,
        input: Any,
        model_parameters: dict | None = None,
        metadata: dict | None = None,
    ):
        self._start_kwargs = {
            "as_type": "generation",
            "name": name,
            "model": model,
            "input": input,
            "model_parameters": model_parameters or {},
            "metadata": metadata or {},
        }
        self._metadata = dict(metadata or {})
        self._context: AbstractContextManager | None = None
        self._generation = None

    def __enter__(self) -> "GenerationTrace":
        try:
            self._context = get_langfuse().start_as_current_observation(**self._start_kwargs)
            self._generation = self._context.__enter__()
        except Exception:
            logger.exception("Couldn't start Langfuse generation %r; continuing untraced", self._start_kwargs["name"])
            self._context = self._generation = None
        return self

    def success(self, build_update: Callable[[], dict]) -> None:
        """`build_update` returns kwargs for `generation.update()`. It's a callable
        so that building the output (parsing an unexpected response shape, say)
        happens inside the guard too."""
        self._update(build_update, "success")

    def error(self, exc: BaseException) -> None:
        self._update(lambda: {
            "output": _error_output(exc),
            "level": "ERROR",
            "status_message": f"{type(exc).__name__}: {exc}",
        }, "error")

    def _update(self, build_update: Callable[[], dict], label: str) -> None:
        if self._generation is None:
            return
        try:
            update = build_update()
            # Re-send the start metadata with any additions so no key is lost.
            update["metadata"] = {**self._metadata, **(update.get("metadata") or {})}
            self._generation.update(**update)
        except Exception:
            logger.exception("Couldn't record Langfuse generation %s for %r", label, self._start_kwargs["name"])

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._context is not None:
            try:
                self._context.__exit__(exc_type, exc, tb)
            except Exception:
                logger.exception("Couldn't end Langfuse generation %r", self._start_kwargs["name"])
        return False  # never swallow the caller's exception


def _error_output(exc: BaseException) -> dict:
    """What failed, in the output field where it's easy to read in Langfuse. For
    provider errors that includes the provider's own error body (Airtap's
    _vendorResponse, but only on failure, to keep successful traces small)."""
    output: dict[str, Any] = {"error": str(exc), "error_type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        output["status_code"] = status_code
    body = getattr(exc, "body", None)
    if body:
        output["provider_error"] = body
    return output
