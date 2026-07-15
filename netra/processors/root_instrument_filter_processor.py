import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Set, cast

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import INVALID_SPAN_ID, SpanContext

logger = logging.getLogger(__name__)

_INSTRUMENTATION_PREFIXES = ("opentelemetry.instrumentation.", "netra.instrumentation.")

_MAX_ROOT_CANDIDATES = 4096
_ROOT_CANDIDATE_TTL_SECONDS = 600.0

# Durable per-span marker set on a span the moment it is classified as a
# root-block candidate.  Unlike the registry, this marker travels *with* the
# span into its export batch, so the exporter can still recognise a blocked root
# even if the registry entry was evicted (TTL/overflow) or cleared (shutdown)
# between ``on_start`` and export.  The exporter strips it before export so it
# never leaks into surviving (kept) spans.
ROOT_BLOCK_CANDIDATE_ATTR = "netra.root_block_candidate"

# Process-global registry of "root-block candidate" spans: spans emitted by an
# auto-instrumentation library that is *not* permitted to produce root-level
# spans.  Maps ``span_id -> (parent_span_context_or_None, monotonic_timestamp)``.
#
# The registry lets the ``FilteringSpanExporter`` resolve *cross-batch*
# ancestry — reparenting a child that exports in a later batch than its dropped
# ancestor.  It is a supplement, not the source of truth: candidacy is carried
# durably on the span via ``ROOT_BLOCK_CANDIDATE_ATTR`` (see above).  Entries
# survive until they expire via TTL so late-exporting children still find their
# dropped ancestor's parent.
_root_candidates_lock = threading.Lock()
ROOT_BLOCK_CANDIDATES: "OrderedDict[int, tuple[Optional[SpanContext], float]]" = OrderedDict()


def get_root_block_candidates() -> Dict[int, Optional[SpanContext]]:
    """Return a snapshot ``{span_id -> parent_span_context}`` of the current
    root-block candidate registry.

    The snapshot decouples the exporter's read from concurrent writes by
    ``RootInstrumentFilterProcessor.on_start``.

    Returns:
        A plain dict mapping each candidate span's ID to the ``SpanContext`` of
        its parent (``None`` for a candidate that is itself a trace root).
    """
    with _root_candidates_lock:
        return {span_id: parent_ctx for span_id, (parent_ctx, _ts) in ROOT_BLOCK_CANDIDATES.items()}


class RootInstrumentFilterProcessor(SpanProcessor):  # type: ignore[misc]
    """Record spans from auto-instrumentation libraries that are not permitted
    to produce root-level spans, so the exporter can drop-and-reparent them.

    Unlike a naive "block the whole trace" filter, this processor never
    discards a subtree.  When an auto-instrumentation span (e.g. FastAPI,
    Flask, ASGI) comes from a library outside the allowed *root_instruments*
    set, the processor marks it as a **root-block candidate** with a durable
    span attribute (``ROOT_BLOCK_CANDIDATE_ATTR``) and also records it, along
    with its parent ``SpanContext``, in a shared TTL-evicted registry
    (``ROOT_BLOCK_CANDIDATES``) used for cross-batch reparenting.

    The actual drop decision is made at export time by
    :class:`~netra.exporters.filtering_span_exporter.FilteringSpanExporter`:
    a candidate is dropped only when it is *root-connected* — i.e. it is a
    trace root, or every ancestor up to the trace root is also a dropped
    candidate.  Dropped candidates have their children reparented onto the
    dropped span's parent (``None`` for a true root, so the child becomes the
    new root).  This peel repeats recursively: if the promoted child is itself
    from a non-root instrumentation it is dropped too, until a survivor is
    reached (an allowed instrument, a netra decorator / manual span, or any
    non-instrumentation span).

    Spans created directly through netra decorators or ``Netra.start_span``
    are never candidates — only spans from recognised auto-instrumentation
    libraries (scope prefix ``opentelemetry.instrumentation.*`` or
    ``netra.instrumentation.*``) are subject to the allow-list.

    Args:
        allowed_root_instrument_names: Instrumentation-name strings
            (e.g. ``"openai"``, ``"fastapi"``) that may produce root spans.
    """

    def __init__(self, allowed_root_instrument_names: Set[str]) -> None:
        self._allowed: frozenset[str] = frozenset(allowed_root_instrument_names)

    def on_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context] = None,
    ) -> None:
        """Record *span* as a root-block candidate when it comes from a
        non-allowed auto-instrumentation library.

        Args:
            span: The span that is being started.
            parent_context: The parent context of the span.
        """
        try:
            self._process_span_start(span)
        except Exception:
            logger.debug("RootInstrumentFilterProcessor.on_start failed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """Prune expired entries from the candidate registry.

        Entries are **not** cleared per-span — they survive until TTL
        expiry so that children exporting after their (already-ended) blocked
        ancestor can still be reparented.

        Args:
            span: The span that is being ended.
        """
        try:
            self._evict_stale_candidates()
        except Exception:
            pass

    def shutdown(self) -> None:
        """No-op — the candidate registry is **not** cleared here.

        This processor is registered before the exporter's span processor, so
        clearing the shared registry on shutdown would empty it *before* the
        exporter's final flush runs — causing still-buffered blocked root spans
        to slip through as exported roots.  The registry is process-global and
        already bounded (``_MAX_ROOT_CANDIDATES``) and TTL-evicted, so leaving
        it intact is safe; clearing it here is not.
        """
        return None

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

    def _process_span_start(self, span: Span) -> None:
        """Record *span* as a candidate if it is an auto-instrumentation span
        whose instrument is not in the allowed root set.

        Every qualifying span is recorded regardless of whether it is a root
        or a child at start time.  A child only *becomes* a root after its
        ancestors are dropped, which is resolved at export; recording it now
        ensures the exporter can peel it recursively.

        Args:
            span: The span that is being started.
        """
        if not self._is_from_instrumentation_library(span):
            return

        instr_name = self._extract_instrumentation_name(span)
        if instr_name is None or instr_name in self._allowed:
            return

        span_id = self._get_own_span_id(span)
        if span_id is None or span_id == INVALID_SPAN_ID:
            return

        parent_ctx = self._get_parent_span_context(span)
        # Durable marker first: it must survive even if the registry entry is
        # later evicted/cleared before this span reaches the exporter.
        self._mark_candidate(span)
        self._record_candidate(span_id, parent_ctx)

    @staticmethod
    def _mark_candidate(span: Span) -> None:
        """Stamp *span* with the durable root-block-candidate marker.

        Args:
            span: The candidate span being started.
        """
        try:
            span.set_attribute(ROOT_BLOCK_CANDIDATE_ATTR, True)
        except Exception:
            pass

    @staticmethod
    def _record_candidate(span_id: int, parent_ctx: Optional[SpanContext]) -> None:
        """Register *span_id* as a root-block candidate with its parent context.

        Args:
            span_id: The candidate span's own span ID.
            parent_ctx: The candidate's parent ``SpanContext`` (``None`` when it
                is a trace root).
        """
        with _root_candidates_lock:
            ROOT_BLOCK_CANDIDATES[span_id] = (parent_ctx, time.monotonic())
            ROOT_BLOCK_CANDIDATES.move_to_end(span_id)
            while len(ROOT_BLOCK_CANDIDATES) > _MAX_ROOT_CANDIDATES:
                ROOT_BLOCK_CANDIDATES.popitem(last=False)

    @staticmethod
    def _get_own_span_id(span: Span) -> Optional[int]:
        """Return *span*'s own ``span_id``.

        Args:
            span: The span to inspect.

        Returns:
            The integer span ID, or ``None`` if unavailable.
        """
        ctx = getattr(span, "context", None) or getattr(span, "get_span_context", lambda: None)()
        if ctx is None:
            return None
        return cast(Optional[int], getattr(ctx, "span_id", None))

    @staticmethod
    def _get_parent_span_context(span: Span) -> Optional[SpanContext]:
        """Return the parent ``SpanContext`` recorded on *span*.

        The OTel SDK ``Span`` stores its parent ``SpanContext`` directly, which
        is authoritative for reparenting.  A ``None`` result means the span is a
        trace root (its children should be promoted to roots when it is
        dropped).

        Args:
            span: The span to inspect.

        Returns:
            The parent ``SpanContext``, or ``None`` for a root span.
        """
        parent = getattr(span, "parent", None)
        if parent is None:
            return None
        parent_id = getattr(parent, "span_id", None)
        if parent_id is None or parent_id == INVALID_SPAN_ID:
            return None
        return cast(Optional[SpanContext], parent)

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

    def _evict_stale_candidates(self) -> None:
        """Remove entries older than ``_ROOT_CANDIDATE_TTL_SECONDS``."""
        cutoff = time.monotonic() - _ROOT_CANDIDATE_TTL_SECONDS
        with _root_candidates_lock:
            while ROOT_BLOCK_CANDIDATES:
                _, (_, ts) = next(iter(ROOT_BLOCK_CANDIDATES.items()))
                if ts > cutoff:
                    break
                ROOT_BLOCK_CANDIDATES.popitem(last=False)
