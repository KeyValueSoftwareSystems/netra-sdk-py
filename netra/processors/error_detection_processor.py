import logging
from typing import Any, Optional, Union

import httpx
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Context, Span, Status, StatusCode

from netra import Netra

logger = logging.getLogger(__name__)


class ErrorDetectionProcessor(SpanProcessor):  # type: ignore[misc]
    """
    OpenTelemetry span processor that monitors for error attributes in spans and creates custom events.
    """

    def __init__(self) -> None:
        pass

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span starts."""
        span_id = self._get_span_id(span)
        if not span_id:
            return

        # Wrap span methods to capture data
        self._wrap_span_methods(span, span_id)

    def on_end(self, span: Span) -> None:
        """Called when a span ends."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending data."""
        return True

    def shutdown(self) -> bool:
        """Shutdown the processor."""
        return True

    def _get_span_id(self, span: Span) -> Optional[str]:
        """Get a unique identifier for the span."""
        try:
            span_context = span.get_span_context()
            return f"{span_context.trace_id:032x}-{span_context.span_id:016x}"
        except Exception:
            return None

    def _status_code_processing(self, status_code: int) -> None:
        if httpx.codes.is_error(status_code):
            event_attributes = {"has_error": True, "status_code": status_code}
            Netra.set_custom_event(event_name="error_detected", attributes=event_attributes)

    def _wrap_span_methods(self, span: Span, span_id: str) -> Any:
        """Wrap span methods to capture attributes and events."""
        # Wrap set_attribute
        original_set_attribute = span.set_attribute

        def wrapped_set_attribute(key: str, value: Any) -> Any:
            # Status code processing
            if key == "http.status_code":
                self._status_code_processing(value)

            return original_set_attribute(key, value)

        # Wrap set_status
        original_set_status = span.set_status

        def wrapped_set_status(status: Union[Status, StatusCode]) -> Any:
            # Check if status code is ERROR
            if isinstance(status, Status):
                status_code = status.status_code
            elif isinstance(status, StatusCode):
                status_code = status
            if status_code == StatusCode.ERROR:
                event_attributes = {
                    "has_error": True,
                }
                Netra.set_custom_event(event_name="error_detected", attributes=event_attributes)

            return original_set_status(status)

        span.set_attribute = wrapped_set_attribute
        span.set_status = wrapped_set_status
