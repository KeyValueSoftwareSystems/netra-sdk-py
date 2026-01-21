import logging
import threading
from typing import Dict, Optional, Set

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import SpanContext

logger = logging.getLogger(__name__)


class LlmTraceIdentifierSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """
    Identifies and marks traces containing LLM calls for auto-evaluation.

    This processor monitors spans for LLM-related attributes and automatically
    marks the root span when an LLM call is detected within the trace.

    Thread-safe implementation using locks to protect shared state.

    Attributes:
        DEFAULT_REQUEST_MODEL_KEY: Default attribute key for request model.
        DEFAULT_RESPONSE_MODEL_KEY: Default attribute key for response model.
        DEFAULT_ROOT_MARKER_KEY: Default attribute key for marking root spans.
    """

    # Default attribute keys
    DEFAULT_REQUEST_MODEL_KEY = "gen_ai.request.model"
    DEFAULT_RESPONSE_MODEL_KEY = "gen_ai.response.model"
    DEFAULT_ROOT_MARKER_KEY = "netra.trace.llm.call"

    def __init__(
        self,
        request_model_attribute_key: str = DEFAULT_REQUEST_MODEL_KEY,
        response_model_attribute_key: str = DEFAULT_RESPONSE_MODEL_KEY,
        root_marker_attribute_key: str = DEFAULT_ROOT_MARKER_KEY,
    ) -> None:
        """
        Initialize the LLM trace identifier span processor.

        Args:
            request_model_attribute_key: Attribute key to identify request model in spans.
            response_model_attribute_key: Attribute key to identify response model in spans.
            root_marker_attribute_key: Attribute key to mark root spans containing LLM calls.

        Raises:
            ValueError: If any attribute key is empty.
        """
        self._request_model_key = request_model_attribute_key
        self._response_model_key = response_model_attribute_key
        self._root_marker_key = root_marker_attribute_key

        # Thread synchronization
        self._lock = threading.Lock()

        # Trace state tracking
        self._root_spans_by_trace: Dict[int, Span] = {}
        self._root_span_ids_by_trace: Dict[int, int] = {}
        self._marked_traces: Set[int] = set()

    def on_start(self, span: Span, parent_context: Optional[object] = None) -> None:
        """
        Handle span start events.

        Registers "local root spans" for trace tracking. A local root is either:
        - A true root span (no parent), OR
        - The first span seen for a trace in this process (handles remote parent case)

        Args:
            span: The span that has started.
            parent_context: Optional parent context (unused).
        """
        try:
            span_context = self._get_span_context(span)
            if span_context is None:
                return

            trace_id = span_context.trace_id
            span_id = span_context.span_id

            with self._lock:
                # Register the first span we see for this trace as the local root.
                # Using setdefault ensures we only register the first span, handling
                # both true roots and distributed tracing (remote parent) cases.
                if trace_id not in self._root_spans_by_trace:
                    self._root_spans_by_trace[trace_id] = span
                    self._root_span_ids_by_trace[trace_id] = span_id

        except Exception as e:
            logger.warning("Error processing span start: %s", e, exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Handle span end events.

        Checks if the span contains LLM call attributes and marks the root span
        if found. Cleans up trace state when root spans complete.

        Args:
            span: The span that has ended.
        """
        try:
            span_context = self._get_span_context(span)
            if span_context is None:
                return

            trace_id = span_context.trace_id

            if self._is_root_span_ending(trace_id, span_context.span_id):
                self._cleanup_trace(trace_id)
                return

            if self._is_trace_marked(trace_id):
                return

            if not self._is_llm_span(span):
                return

            self._mark_root_span(trace_id)

        except Exception as e:
            logger.warning("Error processing span end: %s", e, exc_info=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any pending data.

        This processor doesn't buffer data, so this is a no-op that always succeeds.

        Args:
            timeout_millis: Maximum time to wait for flush in milliseconds (unused).

        Returns:
            Always returns True.
        """
        return True

    def shutdown(self) -> None:
        """
        Shutdown the processor and release all resources.

        Clears all internal state and releases locks.
        """
        with self._lock:
            self._root_spans_by_trace.clear()
            self._root_span_ids_by_trace.clear()
            self._marked_traces.clear()

    @staticmethod
    def _is_root_span(span: Span) -> bool:
        """
        Determine if a span is a true root span (no valid parent).

        Args:
            span: The span to check.

        Returns:
            True if the span has no valid parent, False otherwise.
        """
        parent = getattr(span, "parent", None)
        if parent is None:
            return True

        # is_valid can be either a property (bool) or a method (callable)
        is_valid = getattr(parent, "is_valid", False)
        if callable(is_valid):
            return not is_valid()
        return not is_valid

    @staticmethod
    def _get_span_context(span: ReadableSpan) -> Optional[SpanContext]:
        """
        Safely extract span context from a span.

        Args:
            span: The span to extract context from.

        Returns:
            The span context, or None if unavailable.
        """
        try:
            return span.get_span_context()
        except Exception as e:
            logger.debug("Failed to get span context: %s", e)
            return None

    def _is_root_span_ending(self, trace_id: int, span_id: int) -> bool:
        """
        Check if the ending span is the root span for its trace.

        Args:
            trace_id: The trace ID to check.
            span_id: The span ID to check.

        Returns:
            True if this is the root span ending, False otherwise.
        """
        with self._lock:
            root_span_id = self._root_span_ids_by_trace.get(trace_id)
            return root_span_id is not None and span_id == root_span_id

    def _is_trace_marked(self, trace_id: int) -> bool:
        """
        Check if a trace has already been marked.

        Args:
            trace_id: The trace ID to check.

        Returns:
            True if the trace is marked, False otherwise.
        """
        with self._lock:
            return trace_id in self._marked_traces

    def _is_llm_span(self, span: ReadableSpan) -> bool:
        """
        Determine if a span represents an LLM call.

        Args:
            span: The span to check.

        Returns:
            True if the span has LLM-related attributes, False otherwise.
        """
        attributes = getattr(span, "attributes", None)
        if attributes is None:
            return False

        has_request_model = self._request_model_key in attributes
        has_response_model = self._response_model_key in attributes

        return has_request_model or has_response_model

    def _mark_root_span(self, trace_id: int) -> None:
        """
        Mark the root span of a trace as containing an LLM call.

        Args:
            trace_id: The trace ID whose root span should be marked.
        """
        with self._lock:
            root_span = self._root_spans_by_trace.get(trace_id)
            self._marked_traces.add(trace_id)

            if root_span is None:
                return

            is_recording = getattr(root_span, "is_recording", lambda: False)()
            if not is_recording:
                logger.debug("Root span not recording for trace_id=%s", trace_id)
                return

            try:
                root_span.set_attribute(self._root_marker_key, True)
            except Exception as e:
                logger.warning(
                    "Failed to mark root span for trace_id=%s: %s",
                    trace_id,
                    e,
                    exc_info=True,
                )

    def _cleanup_trace(self, trace_id: int) -> None:
        """
        Clean up state for a completed trace.

        Args:
            trace_id: The trace ID to clean up.
        """
        with self._lock:
            self._root_spans_by_trace.pop(trace_id, None)
            self._root_span_ids_by_trace.pop(trace_id, None)
            self._marked_traces.discard(trace_id)
