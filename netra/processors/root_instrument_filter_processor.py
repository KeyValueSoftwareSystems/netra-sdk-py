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

# Durable per-span marker identifying a root-block candidate.  It is a plain
# *instance attribute* — deliberately NOT an OTel span attribute — so it does not
# consume the span's bounded attribute capacity, is never evicted by
# ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` when later instrumentation adds attributes,
# and is never serialised to the backend, so nothing needs to strip it before
# export.  Carrying candidacy on the span object itself is what lets the exporter
# recognise a blocked root even when the registry entry was evicted
# (TTL/overflow) or cleared (shutdown) before export.
#
# The marker is stamped twice, and the ``on_end`` stamp is the one that reaches
# the exporter: ``Span.end()`` calls ``on_end(self._readable_span())``, handing
# the processor — and therefore the export batch — a fresh ``ReadableSpan`` built
# from a fixed field list that no instance attribute is part of.  The ``on_start``
# stamp consequently never travels to the exporter on a real SDK span; it is kept
# only for span implementations that hand ``on_end`` the live object.  See
# ``_mark_candidate_on_export_copy``.
ROOT_BLOCK_CANDIDATE_FIELD = "_netra_root_block_candidate"

# One-shot guard for reporting a failure to stamp the marker.  A failure means
# root-instrument filtering silently drops nothing, so it must not be swallowed —
# but it is a structural condition, not a per-span one, so it is reported loudly
# once and at DEBUG thereafter rather than once per ending span.
_marker_failure_lock = threading.Lock()
_marker_failure_reported = False


def _report_marker_failure(span: ReadableSpan) -> None:
    """Report an inability to stamp the root-block candidacy marker on *span*.

    Must be called from inside an ``except`` block so the traceback is captured.

    The marker is the only way the exporter recognises a blocked root, so a
    failure here means ``root_instruments`` filtering is inert for this span —
    exactly the class of silent leak the marker exists to prevent.  Stamping is
    a plain ``setattr`` on a ``ReadableSpan`` and does not fail on any supported
    ``opentelemetry-sdk``; it would start failing if a future release gave
    ``ReadableSpan`` ``__slots__``.

    Args:
        span: The span whose marker could not be set.
    """
    global _marker_failure_reported
    with _marker_failure_lock:
        is_first_failure = not _marker_failure_reported
        _marker_failure_reported = True

    if is_first_failure:
        logger.warning(
            "Could not stamp the root-block candidacy marker on span %r; spans from "
            "instrumentations outside root_instruments will not be dropped",
            getattr(span, "name", "<unknown>"),
            exc_info=True,
        )
    else:
        logger.debug("Could not stamp the root-block candidacy marker", exc_info=True)


# Process-global registry of "root-block candidate" spans: spans emitted by an
# auto-instrumentation library that is *not* permitted to produce root-level
# spans.  Maps ``span_id -> (parent_span_context_or_None, ended_at)`` where
# ``ended_at`` is ``None`` while the span is still active and the monotonic end
# timestamp once it has ended.
#
# The registry lets the ``FilteringSpanExporter`` resolve *cross-batch*
# ancestry — reparenting a child that exports in a later batch than its dropped
# ancestor.  It is a supplement, not the source of truth: candidacy is carried
# durably on the span via ``ROOT_BLOCK_CANDIDATE_FIELD`` (see above).  The TTL
# clock only starts when a candidate *ends*, so a long-lived root is never
# evicted while still active, and ended entries survive long enough for
# late-exporting children to still find their dropped ancestor's parent.
_root_candidates_lock = threading.Lock()
ROOT_BLOCK_CANDIDATES: "OrderedDict[int, tuple[Optional[SpanContext], Optional[float]]]" = OrderedDict()


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
        return {span_id: parent_ctx for span_id, (parent_ctx, _ended_at) in ROOT_BLOCK_CANDIDATES.items()}


def root_block_candidates_contains(span_ids: Set[int]) -> bool:
    """Return whether any of *span_ids* is currently a recorded candidate.

    A cheap membership probe (set lookups under the lock, no copy) that lets the
    exporter skip the full :func:`get_root_block_candidates` snapshot on batches
    whose spans reference no cross-batch candidate ancestor.

    Args:
        span_ids: Span IDs to probe against the registry.

    Returns:
        ``True`` if at least one ID is a live candidate.
    """
    if not span_ids:
        return False
    with _root_candidates_lock:
        return any(span_id in ROOT_BLOCK_CANDIDATES for span_id in span_ids)


class RootInstrumentFilterProcessor(SpanProcessor):  # type: ignore[misc]
    """Record spans from auto-instrumentation libraries that are not permitted
    to produce root-level spans, so the exporter can drop-and-reparent them.

    When an auto-instrumentation span comes from a library outside the allowed
    *root_instruments* set, the processor marks it as a **root-block candidate**
    with a durable instance-level marker (``ROOT_BLOCK_CANDIDATE_FIELD``) and
    also records it, along with its parent ``SpanContext``, in a shared TTL-evicted
    registry (``ROOT_BLOCK_CANDIDATES``) used for cross-batch reparenting.

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
        """Mark the export-bound span copy, start the TTL clock for a just-ended
        candidate, then prune expired entries.

        Entries are **not** cleared per-span — once a candidate ends its entry
        survives for ``_ROOT_CANDIDATE_TTL_SECONDS`` so that children exporting
        after their (already-ended) blocked ancestor can still be reparented.
        The TTL is measured from the *end* time, so an active long-lived root is
        never evicted before it finishes.

        Args:
            span: The span that is being ended.  On a real SDK span this is the
                ``ReadableSpan`` that goes into the export batch, not the live
                span marked at ``on_start``.
        """
        try:
            self._mark_candidate_on_export_copy(span)
            self._mark_ended(span)
            self._evict_stale_candidates()
        except Exception:
            logger.debug("RootInstrumentFilterProcessor.on_end failed", exc_info=True)

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
        # Stamps the *live* span, which a real SDK span never carries to the
        # exporter — ``on_end`` re-stamps the export copy that it does read (see
        # ``_mark_candidate_on_export_copy``).  Kept for span implementations
        # that hand ``on_end`` the same object they handed ``on_start``.
        self._mark_candidate(span)
        self._record_candidate(span_id, parent_ctx)

    def _mark_candidate_on_export_copy(self, span: ReadableSpan) -> None:
        """Re-stamp the candidacy marker on the span object that reaches the exporter.

        ``Span.end()`` does not hand ``on_end`` the live span marked at
        ``on_start``: it calls ``on_end(self._readable_span())``, and
        ``_readable_span()`` builds a fresh ``ReadableSpan`` from a fixed list of
        fields that an instance attribute is not part of.  The ``on_start``
        marker is therefore always absent from the object the export batch
        carries, and candidacy has to be resolved again here — cheaply, from the
        same scope string — on the copy the exporter will actually read.

        Without this, ``FilteringSpanExporter`` sees no in-batch candidate and
        falls back to the cross-batch registry, which it only consults when some
        batch span's *parent* is a registry entry.  A blocked root with no
        children in the batch — any standalone redis / requests / sqlalchemy
        call — is then nobody's parent, matches nothing, and is exported instead
        of dropped.

        Args:
            span: The ending span, as it will be handed to the exporter.
        """
        if not self._is_from_instrumentation_library(span):
            return
        instr_name = self._extract_instrumentation_name(span)
        if instr_name is None or instr_name in self._allowed:
            return
        self._mark_candidate(span)

    @staticmethod
    def _mark_candidate(span: ReadableSpan) -> None:
        """Stamp *span* with the durable root-block-candidate marker.

        The marker is a plain instance attribute (see ``ROOT_BLOCK_CANDIDATE_FIELD``),
        not an OTel span attribute, so it cannot be evicted by the span attribute
        limit and is never exported.

        Args:
            span: The candidate span being started, or the ``ReadableSpan`` copy
                of it being ended.
        """
        try:
            setattr(span, ROOT_BLOCK_CANDIDATE_FIELD, True)
        except Exception:
            _report_marker_failure(span)

    @staticmethod
    def _record_candidate(span_id: int, parent_ctx: Optional[SpanContext]) -> None:
        """Register *span_id* as an **active** root-block candidate with its parent.

        The entry's TTL clock is left unstarted (``ended_at = None``) until the
        span ends, so a candidate that stays open longer than the TTL is never
        evicted while still active.

        Args:
            span_id: The candidate span's own span ID.
            parent_ctx: The candidate's parent ``SpanContext`` (``None`` when it
                is a trace root).
        """
        with _root_candidates_lock:
            ROOT_BLOCK_CANDIDATES[span_id] = (parent_ctx, None)
            ROOT_BLOCK_CANDIDATES.move_to_end(span_id)
            while len(ROOT_BLOCK_CANDIDATES) > _MAX_ROOT_CANDIDATES:
                ROOT_BLOCK_CANDIDATES.popitem(last=False)

    @staticmethod
    def _mark_ended(span: ReadableSpan) -> None:
        """Start the TTL clock for *span*'s candidate entry, if it has one.

        Args:
            span: The span that is ending.
        """
        ctx = getattr(span, "context", None)
        span_id = getattr(ctx, "span_id", None) if ctx is not None else None
        if span_id is None:
            return
        with _root_candidates_lock:
            entry = ROOT_BLOCK_CANDIDATES.get(span_id)
            if entry is not None:
                ROOT_BLOCK_CANDIDATES[span_id] = (entry[0], time.monotonic())

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
    def _is_from_instrumentation_library(span: ReadableSpan) -> bool:
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
    def _extract_instrumentation_name(span: ReadableSpan) -> Optional[str]:
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
        """Evict entries whose span ended more than ``_ROOT_CANDIDATE_TTL_SECONDS`` ago.

        Active candidates (``ended_at is None``) are never TTL-evicted. Scanning
        stops at the first entry that is still active or not yet stale; a
        long-lived active entry may therefore defer reclamation of stale entries
        behind it, but memory stays bounded by ``_MAX_ROOT_CANDIDATES``.
        """
        cutoff = time.monotonic() - _ROOT_CANDIDATE_TTL_SECONDS
        with _root_candidates_lock:
            while ROOT_BLOCK_CANDIDATES:
                _, (_, ended_at) = next(iter(ROOT_BLOCK_CANDIDATES.items()))
                if ended_at is None or ended_at > cutoff:
                    break
                ROOT_BLOCK_CANDIDATES.popitem(last=False)
