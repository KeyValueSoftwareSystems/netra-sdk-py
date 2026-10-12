import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

from opentelemetry import context as context_api
from opentelemetry.trace import Span

from netra.config import get_active_config
from netra.processors.root_span_processor import RootSpanProcessor

CONTENT_TRACING_OVERRIDE_KEY = "override_enable_content_tracing"


def should_capture_content() -> bool:
    """Whether prompt / completion / transcript content may be recorded on spans.

    Every Netra instrumentor gates content attributes (prompts, completions,
    tool arguments and results, system prompts, transcripts, TTS text) on this.
    Metadata such as model, usage, cost, ids and status is never gated.

    Reads ``trace_content`` from the config registered by ``Netra.init()``.
    Before init, falls back to ``TRACELOOP_TRACE_CONTENT`` (which ``Config``
    also sets), defaulting to enabled. As with Traceloop's instrumentors, the
    ``override_enable_content_tracing`` context value force-enables capture.
    """
    cfg = get_active_config()
    if cfg is not None:
        enabled = bool(cfg.trace_content)
    else:
        enabled = (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true"
    if enabled:
        return True
    return bool(context_api.get_value(CONTENT_TRACING_OVERRIDE_KEY))


def set_content_attribute(span: Span, key: str, value: Any) -> None:
    """Set a content attribute (prompt, completion, transcript, ...) only when capture is enabled."""
    if should_capture_content():
        span.set_attribute(key, value)


def _safe_set_attribute(span: Span, key: str, value: Any, max_length: Optional[int] = None) -> bool:
    """Safely set a span attribute with optional truncation and null checks.

    Args:
        span: The OpenTelemetry span on which to set the attribute.
        key: The attribute key.
        value: The attribute value. If None, the attribute is not set.
        max_length: If provided, the string representation of value is truncated
            to this length before being set.

    Returns:
        True if the attribute was successfully set, False otherwise.
    """
    if not span.is_recording() or value is None:
        return False

    try:
        str_value = str(value)
        if max_length and len(str_value) > max_length:
            str_value = str_value[:max_length]
    except Exception:
        logger.warning("Failed to convert value to string for attribute '%s'", key, exc_info=True)
        return False

    try:
        span.set_attribute(key, str_value)
    except Exception:
        logger.warning("Failed to set span attribute '%s'", key, exc_info=True)
        return False
    return True


TIMESTAMP_ATTRIBUTE_SUFFIX = ".timestamp"


def record_span_timing(
    span: Span,
    attribute: str,
    event_time: Optional[float] = None,
    use_root_span: bool = False,
    reference_time: Optional[float] = None,
    record_event_timestamp: bool = False,
) -> bool:
    """Compute elapsed time for an event and set it as a span attribute.

    Elapsed time is measured from:
      - ``reference_time`` (seconds since epoch) if provided explicitly.
      - ``use_root_span=False`` (default): the start time of the given span.
      - ``use_root_span=True``: the start time of the root span of the given span.

    Args:
        span: The OpenTelemetry span on which to record the timing attribute.
        attribute: The attribute key under which the elapsed time is stored.
        event_time: The event timestamp in seconds since epoch. Defaults to
            ``time.time()`` if not provided.
        use_root_span: If True, elapsed time is measured from the root span's
            start time instead of the given span's start time. Ignored when
            ``reference_time`` is provided.
        reference_time: Optional explicit reference timestamp in seconds since
            epoch. When provided, elapsed is computed as
            ``event_time - reference_time``, bypassing span start-time lookup.
        record_event_timestamp: If True and the timing attribute is successfully
            set, also stores the event timestamp as a UTC ISO 8601 string
            under ``{attribute}.timestamp``.

    Returns:
        True if the timing attribute was successfully set, False if the elapsed
        time could not be computed (e.g. missing start time or root span).
    """
    t = event_time if event_time is not None else time.time()

    if reference_time is not None:
        success = _safe_set_attribute(span, attribute, t - reference_time)
    else:
        start_time = None

        if not use_root_span:
            start_time = getattr(span, "start_time", None)
        else:
            root_span = RootSpanProcessor.get_root_span(span)
            if not root_span:
                return False
            start_time = getattr(root_span, "start_time", None)

        if not start_time:
            return False

        elapsed = t - start_time / 1e9  # Convert nanoseconds to seconds
        success = _safe_set_attribute(span, attribute, elapsed)

    if success and record_event_timestamp:
        utc_timestamp = datetime.fromtimestamp(t, tz=timezone.utc).isoformat()
        _safe_set_attribute(span, f"{attribute}{TIMESTAMP_ATTRIBUTE_SUFFIX}", utc_timestamp)

    return success
