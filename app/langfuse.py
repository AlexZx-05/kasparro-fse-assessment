"""Optional Langfuse tracing helpers using the current OpenTelemetry SDK API."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
from typing import Any, Iterator, Literal

logger = logging.getLogger(__name__)
ObservationType = Literal["span", "generation"]


class _NullObservation:
    def update(self, **_: Any) -> None:
        return None


def _get_client() -> Any | None:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").casefold() in {
        "0",
        "false",
        "no",
    }:
        return None
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as error:
        logger.warning(
            "Langfuse client unavailable; tracing disabled (%s)",
            type(error).__name__,
        )
        return None


@contextmanager
def tracked_observation(
    *,
    as_type: ObservationType,
    name: str,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
) -> Iterator[Any]:
    """Record an observation without allowing telemetry failure to break inference."""
    client = _get_client()
    if client is None:
        yield _NullObservation()
        return

    try:
        context = client.start_as_current_observation(
            as_type=as_type,
            name=name,
            metadata=metadata or {},
            input=input_data,
        )
        observation = context.__enter__()
    except Exception as error:
        logger.warning(
            "Could not start Langfuse observation %s (%s)",
            name,
            type(error).__name__,
        )
        yield _NullObservation()
        return

    body_error: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield observation
    except BaseException as error:
        body_error = sys.exc_info()
        try:
            observation.update(
                level="ERROR",
                status_message=type(error).__name__,
            )
        except Exception as telemetry_error:
            logger.debug(
                "Could not record Langfuse observation error (%s)",
                type(telemetry_error).__name__,
            )
        raise
    finally:
        try:
            context.__exit__(*body_error)
        except Exception as error:
            logger.warning(
                "Could not close Langfuse observation %s (%s)",
                name,
                type(error).__name__,
            )
        try:
            client.flush()
        except Exception as error:
            logger.warning("Could not flush Langfuse observations (%s)", type(error).__name__)


def tracked_trace(
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
) -> Any:
    """Open a root answer-level span; nested observations inherit its trace."""
    return tracked_observation(
        as_type="span",
        name=name,
        metadata=metadata,
        input_data=input_data,
    )


def tracked_generation(
    *,
    name: str,
    model: str,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
) -> Any:
    """Open a provider generation, optionally nested under an answer span."""
    return tracked_observation(
        as_type="generation",
        name=name,
        metadata={"model": model, **(metadata or {})},
        input_data=input_data,
    )


def tracked_tool(
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
    input_data: Any = None,
) -> Any:
    """Open a tool span without recording raw answer or catalogue contents."""
    return tracked_observation(
        as_type="span",
        name=name,
        metadata=metadata,
        input_data=input_data,
    )
