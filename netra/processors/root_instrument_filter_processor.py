import logging
import threading
from typing import Optional, Set, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import INVALID_SPAN_ID

logger = logging.getLogger(__name__)

# Attribute written on blocked spans so that the FilteringSpanExporter drops them.
_LOCAL_BLOCKED_ATTR = "netra.local_blocked"

# Scope-name prefixes that identify auto-instrumentation libraries.
_INSTRUMENTATION_PREFIXES = ("opentelemetry.instrumentation.", "netra.instrumentation.")


class RootInstrumentFilterProcessor(SpanProcessor):  # type: ignore[misc]
    """Blocks root spans (and their entire subtree) from instrumentations not in
    the allowed *root_instruments* set.

    The set stores the **instrumentation name values** (e.g. ``"openai"``,
    ``"adk"``, ``"google_genai"``) that are permitted to create root-level spans.
    Any root span whose instrumentation name is *not* in this set is marked with
    ``netra.local_blocked = True`` and its ``span_id`` is recorded.  Child spans
    whose parent ``span_id`` appears in the blocked registry inherit the block.

    Args:
        allowed_root_instrument_names: Set of instrumentation-name strings
            (matching ``InstrumentSet`` member values) that are allowed to
            produce root spans.
    """

    def __init__(self, allowed_root_instrument_names: Set[str]) -> None:
        """
        Initialize the processor with a set of allowed root instrument names.

        Args:
            allowed_root_instrument_names: Set of instrumentation-name strings
                (matching ``InstrumentSet`` member values) that are allowed to
                produce root spans.
        """
        self._allowed: frozenset[str] = frozenset(allowed_root_instrument_names)
        # span_id -> True for every span that belongs to a blocked root tree.
        self._blocked_span_ids: dict[int, bool] = {}
        self._lock = threading.Lock()

    def on_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context] = None,
    ) -> None:
        """
        Called when a span is started.

        Args:
            span: The span that is being started.
            parent_context: The parent context of the span.
        """
        try:
            self._process_span_start(span, parent_context)
        except Exception:
            logger.debug("RootInstrumentFilterProcessor.on_start failed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span is ended.

        Args:
            span: The span that is being ended.
        """
        try:
            span_id = self._get_span_id(span)
            if span_id is not None:
                with self._lock:
                    self._blocked_span_ids.pop(span_id, None)
        except Exception:
            pass

    def shutdown(self) -> None:
        """
        Called when the processor is shut down.
        """
        with self._lock:
            self._blocked_span_ids.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Called when the processor is forced to flush.

        Args:
            timeout_millis: The timeout in milliseconds.

        Returns:
            True if the flush was successful, False otherwise.
        """
        return True

    def _process_span_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context],
    ) -> None:
        """
        Processes the start of a span.

        Args:
            span: The span that is being started.
            parent_context: The parent context of the span.
        """
        parent_span_id = self._resolve_parent_span_id(parent_context)

        if parent_span_id is not None and parent_span_id != INVALID_SPAN_ID:
            # This is a child span – inherit blocked status from parent.
            with self._lock:
                if parent_span_id in self._blocked_span_ids:
                    own_id = self._get_span_id(span)
                    if own_id is not None:
                        self._blocked_span_ids[own_id] = True
                    self._mark_blocked(span)
            return

        # Root span – only apply the allow-list to auto-instrumentation spans.
        # Spans created directly through netra (decorators / Netra.start_span)
        # use arbitrary tracer names and must never be blocked.
        if not self._is_from_instrumentation_library(span):
            return

        instr_name = self._extract_instrumentation_name(span)
        if instr_name is not None and instr_name not in self._allowed:
            own_id = self._get_span_id(span)
            if own_id is not None:
                with self._lock:
                    self._blocked_span_ids[own_id] = True
            self._mark_blocked(span)

    @staticmethod
    def _resolve_parent_span_id(
        parent_context: Optional[otel_context.Context],
    ) -> Optional[int]:
        """
        Return the parent span's ``span_id`` from the supplied context, or ``None``.

        Args:
            parent_context: The parent context of the span.

        Returns:
            The parent span's ``span_id`` or ``None``.
        """
        if parent_context is None:
            return None
        parent_span = trace.get_current_span(parent_context)
        if parent_span is None:
            return None
        sc = parent_span.get_span_context()
        if sc is None:
            return None
        return cast(Optional[int], sc.span_id)

    @staticmethod
    def _get_span_id(span: object) -> Optional[int]:
        """
        Get the span ID from the span.

        Args:
            span: The span to get the ID from.

        Returns:
            The span ID or None.
        """
        ctx = getattr(span, "context", None) or getattr(span, "get_span_context", lambda: None)()
        if ctx is None:
            return None
        return cast(Optional[int], getattr(ctx, "span_id", None))

    @staticmethod
    def _mark_blocked(span: Span) -> None:
        """
        Mark the span as blocked.

        Args:
            span: The span to mark as blocked.
        """
        try:
            span.set_attribute(_LOCAL_BLOCKED_ATTR, True)
        except Exception:
            pass

    @staticmethod
    def _is_from_instrumentation_library(span: Span) -> bool:
        """Return ``True`` if the span originates from a known auto-instrumentation library.

        Spans created by netra decorators or ``Netra.start_span`` use arbitrary
        tracer names that do not match the instrumentation naming convention and
        will return ``False``.

        Args:
            span: The span to check.

        Returns:
            ``True`` when the span's instrumentation scope starts with a known
            instrumentation prefix, ``False`` otherwise.
        """
        scope = getattr(span, "instrumentation_scope", None)
        if scope is None:
            return False
        name = getattr(scope, "name", None)
        if not isinstance(name, str) or not name:
            return False
        return name.startswith(_INSTRUMENTATION_PREFIXES)

    @staticmethod
    def _extract_instrumentation_name(span: Span) -> Optional[str]:
        """
        Extract the short instrumentation name from the span's scope.

        Mirrors the logic in ``InstrumentationSpanProcessor._extract_instrumentation_name``.

        Args:
            span: The span to extract the instrumentation name from.

        Returns:
            The instrumentation name or None.
        """
        scope = getattr(span, "instrumentation_scope", None)
        if scope is None:
            return None
        name = getattr(scope, "name", None)
        if not isinstance(name, str) or not name:
            return None
        for prefix in _INSTRUMENTATION_PREFIXES:
            if name.startswith(prefix):
                base = name.rsplit(".", 1)[-1].strip()
                return base if base else name
        return name
