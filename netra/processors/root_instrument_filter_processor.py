import logging
import threading
import time
from collections import OrderedDict
from typing import Optional, Set, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import INVALID_SPAN_ID

logger = logging.getLogger(__name__)

_LOCAL_BLOCKED_ATTR = "netra.local_blocked"
_INSTRUMENTATION_PREFIXES = ("opentelemetry.instrumentation.", "netra.instrumentation.")

_MAX_BLOCKED_TRACES = 4096
_BLOCKED_TRACE_TTL_SECONDS = 600.0


class RootInstrumentFilterProcessor(SpanProcessor):  # type: ignore[misc]
    """Block root spans (and their entire subtree) whose instrumentation is
    not in the allowed *root_instruments* set.

    When an auto-instrumentation root span (e.g. FastAPI, Flask) is not
    permitted, this processor:

    1. Marks it with ``netra.local_blocked = True``.
    2. Records its **trace_id** in a bounded, TTL-evicted registry.
    3. Marks every subsequent child span that shares the same trace ID.

    Tracking by trace ID (rather than individual span IDs) guarantees that
    the block propagates correctly even in async frameworks where a parent
    span may end before all of its children have started.

    Spans created directly through netra decorators or ``Netra.start_span``
    are never filtered — only spans from recognised auto-instrumentation
    libraries (scope prefix ``opentelemetry.instrumentation.*`` or
    ``netra.instrumentation.*``) are subject to the allow-list.

    Args:
        allowed_root_instrument_names: Instrumentation-name strings
            (e.g. ``"openai"``, ``"fastapi"``) that may produce root spans.
    """

    def __init__(self, allowed_root_instrument_names: Set[str]) -> None:
        self._allowed: frozenset[str] = frozenset(allowed_root_instrument_names)
        self._blocked_trace_ids: OrderedDict[int, float] = OrderedDict()
        self._lock = threading.Lock()

    def on_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context] = None,
    ) -> None:
        """Evaluate whether *span* belongs to a blocked trace and mark it
        accordingly.

        Args:
            span: The span that is being started.
            parent_context: The parent context of the span.
        """
        try:
            self._process_span_start(span, parent_context)
        except Exception:
            logger.debug("RootInstrumentFilterProcessor.on_start failed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """Prune expired entries from the blocked-trace registry.

        The registry is **not** cleared on a per-span basis — entries
        survive until they expire via TTL so that late-starting children
        still see their trace as blocked.

        Args:
            span: The span that is being ended.
        """
        try:
            self._evict_stale_traces()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Release all resources held by the processor."""
        with self._lock:
            self._blocked_trace_ids.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op — this processor does not buffer data.

        Args:
            timeout_millis: Ignored.

        Returns:
            Always ``True``.
        """
        return True

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _process_span_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context],
    ) -> None:
        """Decide whether *span* should be blocked.

        A span is classified as a **child** (non-root) when either the
        *parent_context* carries a valid span or the span object itself
        records a valid ``parent_span_id``.  The two-pronged check handles
        instrumentations (e.g. HTTPX, OpenAI) that set the parent link on
        the span directly without propagating through the active context.

        Args:
            span: The span that is being started.
            parent_context: The parent context of the span.
        """
        parent_span_id = self._resolve_parent_span_id(parent_context)
        if parent_span_id is None or parent_span_id == INVALID_SPAN_ID:
            parent_span_id = self._get_parent_span_id_from_span(span)

        is_child = parent_span_id is not None and parent_span_id != INVALID_SPAN_ID

        if is_child:
            self._maybe_block_child(span)
        else:
            self._maybe_block_root(span)

    def _maybe_block_child(self, span: Span) -> None:
        """Block *span* if its trace is in the blocked registry.

        Args:
            span: A child (non-root) span that is being started.
        """
        trace_id = self._get_trace_id(span)
        if trace_id is None:
            return
        with self._lock:
            if trace_id in self._blocked_trace_ids:
                self._mark_blocked(span)

    def _maybe_block_root(self, span: Span) -> None:
        """Block *span* if it is an auto-instrumentation root whose name
        is not in the allowed set.

        Args:
            span: A root span that is being started.
        """
        if not self._is_from_instrumentation_library(span):
            return

        instr_name = self._extract_instrumentation_name(span)
        if instr_name is None or instr_name in self._allowed:
            return

        trace_id = self._get_trace_id(span)
        if trace_id is not None:
            with self._lock:
                self._blocked_trace_ids[trace_id] = time.monotonic()
                self._blocked_trace_ids.move_to_end(trace_id)
                self._evict_overflow()
        self._mark_blocked(span)

    @staticmethod
    def _resolve_parent_span_id(
        parent_context: Optional[otel_context.Context],
    ) -> Optional[int]:
        """Return the parent span's ``span_id`` from *parent_context*.

        Args:
            parent_context: The context passed to ``on_start``.

        Returns:
            The parent ``span_id``, or ``None`` if unavailable.
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
    def _get_parent_span_id_from_span(span: Span) -> Optional[int]:
        """Extract the parent ``span_id`` from the span's internal state.

        The OTel SDK ``Span`` stores the parent ``SpanContext`` directly,
        which is authoritative even when the active-context-based
        *parent_context* has no current span.

        Args:
            span: The span to inspect.

        Returns:
            The parent ``span_id``, or ``None`` if unavailable.
        """
        parent = getattr(span, "parent", None)
        if parent is None:
            return None
        parent_id = getattr(parent, "span_id", None)
        if parent_id is None or parent_id == INVALID_SPAN_ID:
            return None
        return cast(Optional[int], parent_id)

    @staticmethod
    def _get_trace_id(span: object) -> Optional[int]:
        """Return the ``trace_id`` carried by *span*.

        Args:
            span: Any span-like object with a ``context`` or
                ``get_span_context`` accessor.

        Returns:
            The integer trace ID, or ``None``.
        """
        ctx = getattr(span, "context", None) or getattr(span, "get_span_context", lambda: None)()
        if ctx is None:
            return None
        return cast(Optional[int], getattr(ctx, "trace_id", None))

    @staticmethod
    def _is_from_instrumentation_library(span: Span) -> bool:
        """Return ``True`` when *span* originates from a known
        auto-instrumentation library.

        Spans created by netra decorators or ``Netra.start_span`` use
        arbitrary tracer names that do not match the instrumentation
        naming convention and will return ``False``.

        Args:
            span: The span to check.

        Returns:
            Whether the span's scope starts with a recognised prefix.
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
        """Extract the short instrumentation name from *span*'s scope.

        For a scope named ``netra.instrumentation.fastapi`` this returns
        ``"fastapi"``.

        Args:
            span: The span to inspect.

        Returns:
            The short name, or ``None`` if extraction fails.
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

    @staticmethod
    def _mark_blocked(span: Span) -> None:
        """Set the ``netra.local_blocked`` attribute on *span*.

        Args:
            span: The span to mark.
        """
        try:
            span.set_attribute(_LOCAL_BLOCKED_ATTR, True)
        except Exception:
            pass

    def _evict_stale_traces(self) -> None:
        """Remove entries older than ``_BLOCKED_TRACE_TTL_SECONDS``."""
        cutoff = time.monotonic() - _BLOCKED_TRACE_TTL_SECONDS
        with self._lock:
            while self._blocked_trace_ids:
                _, ts = next(iter(self._blocked_trace_ids.items()))
                if ts > cutoff:
                    break
                self._blocked_trace_ids.popitem(last=False)

    def _evict_overflow(self) -> None:
        """Trim the registry to ``_MAX_BLOCKED_TRACES``.

        Must be called while holding ``self._lock``.
        """
        while len(self._blocked_trace_ids) > _MAX_BLOCKED_TRACES:
            self._blocked_trace_ids.popitem(last=False)
