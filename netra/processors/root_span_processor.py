import logging
import threading
from typing import Dict, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class RootSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """
    A SpanProcessor that tracks root spans using an in-memory dictionary
    keyed by trace_id.

    This implementation identifies root spans at span start and stores them
    in a global mapping. The mapping is cleaned up when the corresponding
    root span ends.

    The dict is a class-level variable so that static lookup helpers can
    resolve root spans without requiring a processor instance.  All mutations
    are protected by the class-level ``_lock``.
    """

    _root_spans: Dict[int, Span] = {}
    _lock: threading.Lock = threading.Lock()

    @staticmethod
    def get_root_span_by_trace_id(trace_id: int) -> Optional[Span]:
        """
        Retrieve the root span associated with a given trace ID.

        Args:
            trace_id: The trace identifier.

        Returns:
            The root Span if present, otherwise None.
        """
        try:
            with RootSpanProcessor._lock:
                return RootSpanProcessor._root_spans.get(trace_id)
        except Exception:
            logger.debug("Failed to get root span", exc_info=True)
            return None

    @staticmethod
    def is_root_span_for_trace(trace_id: int, span_id: int) -> bool:
        """
        Check whether the given span_id is the root span for the given trace_id.

        Args:
            trace_id: The trace identifier.
            span_id: The span identifier to test.

        Returns:
            True if the span is the recorded root span for this trace, False otherwise.
        """
        try:
            with RootSpanProcessor._lock:
                root = RootSpanProcessor._root_spans.get(trace_id)
                if root is None:
                    return False
                root_ctx = root.get_span_context()
                return root_ctx is not None and root_ctx.span_id == span_id
        except Exception:
            logger.debug("RootSpanProcessor: Failed to check root span", exc_info=True)
            return False

    @staticmethod
    def replace_root_span(span: Span) -> bool:
        """Record *span* as its trace's root span, replacing any existing entry.

        ``on_start`` files the first parentless span it sees under a ``setdefault``,
        which is correct for every instrumentation that lets a span's parent stand
        as created.  An instrumentation that *re-roots* a trace after the fact —
        by clearing a span's parent once the span has already started — leaves that
        mapping naming a span which is no longer the root, so the LLM-call marker
        and ``SessionManager``'s root-span helpers would write to the wrong span.
        This is the seam for those instrumentations to correct it.

        Only the mapping is updated; the caller owns the reparenting itself.  The
        entry's lifecycle then follows the new root: ``on_end`` clears the mapping
        when the *recorded* root ends, so the span that used to be the root ending
        first no longer takes the entry with it.

        The replacement is unconditional, so the caller — which is the only party
        that knows whether the span it is holding really is the trace's new root —
        owns the decision to call at all.  Calling it for each of several spans
        that were re-rooted in the same trace leaves the mapping naming whichever
        called last, and every root-span write for that trace then lands there.

        Args:
            span: The span that is now the root of its trace.

        Returns:
            True if the mapping was updated, False when *span* has no valid span
            context to key it by.
        """
        try:
            span_ctx = span.get_span_context()
            if span_ctx is None or not span_ctx.is_valid:
                return False

            with RootSpanProcessor._lock:
                RootSpanProcessor._root_spans[span_ctx.trace_id] = span
            return True
        except Exception:
            logger.debug("RootSpanProcessor: failed to replace root span", exc_info=True)
            return False

    @staticmethod
    def get_root_span(span: Span) -> Optional[Span]:
        """
        Resolve the root span for a given span.

        Args:
            span: The span whose root span is to be determined.

        Returns:
            The root Span if available, otherwise None.
        """
        try:
            span_ctx = span.get_span_context()
            if not span_ctx or not span_ctx.is_valid:
                return None

            return RootSpanProcessor.get_root_span_by_trace_id(span_ctx.trace_id)
        except Exception:
            logger.debug("RootSpanProcessor: Failed to resolve root span", exc_info=True)
            return None

    def _is_root_span(
        self,
        parent_context: Optional[context_api.Context],
    ) -> bool:
        """
        Determine whether a span is a root span.

        A span is considered root if:
            - There is no parent span, or
            - The parent span context is invalid.

        Args:
            parent_context: The parent context passed to on_start.

        Returns:
            True if the span is a root span, False otherwise.
        """
        parent_span = trace.get_current_span(parent_context)
        if parent_span is None:
            return True

        parent_span_ctx = parent_span.get_span_context()
        return parent_span_ctx is None or not parent_span_ctx.is_valid

    def on_start(
        self,
        span: Span,
        parent_context: Optional[context_api.Context],
    ) -> None:
        """
        Hook executed when a span starts.

        If the span is identified as a root span, it is stored in the
        internal mapping using its trace_id.

        Args:
            span: The span being started.
            parent_context: The parent context of the span.
        """
        try:
            span_ctx = span.get_span_context()
            if span_ctx is None or not span_ctx.is_valid:
                return

            if not self._is_root_span(parent_context):
                return

            with self._lock:
                self._root_spans.setdefault(span_ctx.trace_id, span)

        except Exception:
            logger.debug("RootSpanProcessor: error in on_start", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Hook executed when a span ends.

        If the ending span is the root span for its trace, it is removed
        from the internal mapping to prevent stale entries.

        Args:
            span: The span being ended.
        """
        try:
            span_ctx = span.get_span_context()
            if span_ctx is None or not span_ctx.is_valid:
                return

            with self._lock:
                root = self._root_spans.get(span_ctx.trace_id)
                if root is not None:
                    root_ctx = root.get_span_context()
                    if root_ctx is not None and root_ctx.span_id == span_ctx.span_id:
                        self._root_spans.pop(span_ctx.trace_id, None)

        except Exception:
            logger.debug("RootSpanProcessor: error in on_end", exc_info=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any pending data.

        This processor holds no buffered export data, so this is a no-op
        that always returns True.

        Args:
            timeout_millis: Maximum time to wait in milliseconds (unused).

        Returns:
            Always True.
        """
        return True

    def shutdown(self) -> None:
        """
        Shutdown the processor and release all tracked root spans.

        Clears the class-level mapping so that stale entries do not leak
        across test runs or provider resets.
        """
        with self._lock:
            self._root_spans.clear()
