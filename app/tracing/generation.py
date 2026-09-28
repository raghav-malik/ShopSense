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
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

from langfuse import LangfuseGeneration

from app.tracing.langfuse_setup import get_langfuse

logger = logging.getLogger("shopsense.tracing")


class GenerationTrace:
    """One Langfuse generation for one LLM call: opened before the call, completed after (see the module docstring)."""

    def __init__(
        self,
        name: str,
        *,
        model: str,
        input: object,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._input = input
        self._model_parameters = model_parameters or {}
        self._metadata = dict(metadata or {})
        self._context: AbstractContextManager[LangfuseGeneration] | None = None
        self._generation: LangfuseGeneration | None = None

    def __enter__(self) -> "GenerationTrace":
        try:
            self._context = get_langfuse().start_as_current_observation(
                as_type="generation",
                name=self._name,
                model=self._model,
                input=self._input,
                model_parameters=self._model_parameters,
                metadata=self._metadata,
            )
            self._generation = self._context.__enter__()
        except Exception:
            logger.exception("Couldn't start Langfuse generation %r; continuing untraced", self._name)
            self._context = self._generation = None
        return self

    def success(self, build_update: Callable[[], dict[str, Any]]) -> None:
        """`build_update` returns kwargs for `generation.update()`. It's a callable
        so that building the output (parsing an unexpected response shape, say)
        happens inside the guard too."""
        self._update(build_update, "success")

    def error(self, exc: BaseException) -> None:
        """Record the call's failure: level ERROR, with the provider's error body when there is one."""
        self._update(
            lambda: {
                "output": _error_output(exc),
                "level": "ERROR",
                "status_message": f"{type(exc).__name__}: {exc}",
            },
            "error",
        )

    def _update(self, build_update: Callable[[], dict[str, Any]], label: str) -> None:
        if self._generation is None:
            return
        try:
            update = build_update()
            # Re-send the start metadata with any additions so no key is lost.
            update["metadata"] = {**self._metadata, **(update.get("metadata") or {})}
            self._generation.update(**update)
        except Exception:
            logger.exception("Couldn't record Langfuse generation %s for %r", label, self._name)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        # Returns None: never swallows the caller's exception.
        if self._context is not None:
            try:
                self._context.__exit__(exc_type, exc, tb)
            except Exception:
                logger.exception("Couldn't end Langfuse generation %r", self._name)


def _error_output(exc: BaseException) -> dict[str, Any]:
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
