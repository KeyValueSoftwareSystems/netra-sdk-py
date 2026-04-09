import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

from opentelemetry.trace import Span

from netra.processors.root_span_processor import RootSpanProcessor


def _safe_set_attribute(span: Span, key: str, value: Any, max_length: Optional[int] = None) -> bool:
    """Safely set span attribute with optional truncation and null checks."""
    if not span.is_recording() or value is None:
        return False

    str_value = str(value)
    if max_length and len(str_value) > max_length:
        str_value = str_value[:max_length]

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
    """
    Compute elapsed time for an event.

    Elapsed is measured from:
      use_root_span=False (default): start_time of the given span
      use_root_span=True           : start_time of the root span of the given span

    Returns: True if the attribute was set, False if it could not be computed.

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
