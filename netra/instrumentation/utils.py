import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

from opentelemetry.trace import Span

from netra.processors.root_span_processor import RootSpanProcessor


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


def record_span_timing(
    span: Span,
    attribute: str,
    event_time: Optional[float] = None,
    use_root_span: bool = False,
) -> bool:
    """Compute elapsed time for an event and set it as a span attribute.

    Elapsed time is measured from:
      - ``use_root_span=False`` (default): the start time of the given span.
      - ``use_root_span=True``: the start time of the root span of the given span.

    Args:
        span: The OpenTelemetry span on which to record the timing attribute.
        attribute: The attribute key under which the elapsed time is stored.
        event_time: The event timestamp in seconds since epoch. Defaults to
            ``time.time()`` if not provided.
        use_root_span: If True, elapsed time is measured from the root span's
            start time instead of the given span's start time.

    Returns:
        True if the timing attribute was successfully set, False if the elapsed
        time could not be computed (e.g. missing start time or root span).
    """
    t = event_time if event_time is not None else time.time()
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
    return _safe_set_attribute(span, attribute, elapsed)
