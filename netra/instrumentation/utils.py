import time
from typing import Optional

from opentelemetry.trace import Span

from netra.processors.root_span_processor import RootSpanProcessor


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
    span.set_attribute(attribute, elapsed)
    return True
