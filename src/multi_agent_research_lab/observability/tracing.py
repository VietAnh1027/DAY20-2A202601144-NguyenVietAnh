"""Tracing hooks for observability."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context for performance monitoring and trace logging."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    logger.debug(f"[Trace Start] {name} | attributes: {attributes}")
    try:
        yield span
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration
        logger.debug(f"[Trace End] {name} | duration: {duration:.4f}s")
