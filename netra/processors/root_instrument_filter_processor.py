import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, Set, cast

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import INVALID_SPAN_ID, SpanContext

from netra.instrumentation.instruments import THIRD_PARTY_INSTRUMENTATION_SCOPES

logger = logging.getLogger(__name__)

_INSTRUMENTATION_PREFIXES = ("opentelemetry.instrumentation.", "netra.instrumentation.")

_MAX_ROOT_CANDIDATES = 4096
_ROOT_CANDIDATE_TTL_SECONDS = 600.0

# Per-span marker set on a span the moment it is classified as a root-block
# candidate.  It is a plain *instance attribute* — deliberately NOT an OTel span
# attribute — so it costs none of the span's bounded attribute capacity, cannot be
# evicted by ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` when later instrumentation adds
# attributes, and is never serialised to the backend.
#
# It does NOT reach the exporter under the OTel SDK: ``Span.end()`` hands the
# processor chain a fresh ``ReadableSpan`` from ``Span._readable_span()``, which
# copies the span's fields and not its instance attributes.  ``ROOT_BLOCK_CANDIDATES``
# is therefore the authority on candidacy at export time; this marker only serves
# a caller that exports the live span (the reparenting unit tests), where it lets a
# candidate be recognised after its registry entry was evicted or cleared.
ROOT_BLOCK_CANDIDATE_FIELD = "_netra_root_block_candidate"

# Process-global registry of "root-block candidate" spans: spans emitted by an
# auto-instrumentation library that is *not* permitted to produce root-level
# spans.  Maps ``span_id -> (parent_span_context_or_None, ended_at)`` where
# ``ended_at`` is ``None`` while the span is still active and the monotonic end
# timestamp once it has ended.
#
# The registry lets the ``FilteringSpanExporter`` resolve *cross-batch*
# ancestry — reparenting a child that exports in a later batch than its dropped
# ancestor — and, since ``ROOT_BLOCK_CANDIDATE_FIELD`` does not reach the exporter
# under the SDK, it is also the sole authority on candidacy at export time.  The
# TTL clock only starts when a candidate *ends*, so a long-lived root is never
# evicted while still active, and ended entries survive long enough for
# late-exporting children to still find their dropped ancestor's parent.
_root_candidates_lock = threading.Lock()
ROOT_BLOCK_CANDIDATES: "OrderedDict[int, tuple[Optional[SpanContext], Optional[float]]]" = OrderedDict()

# Span IDs of candidates recorded by an explicit :func:`mark_as_root_block_candidate`
# call rather than by the per-instrumentation policy in ``on_start``.
#
# Overflow eviction pops by *insertion* order and does not care whether a span is
# still open, so a single deliberately marked span — one ``job_entrypoint`` per
# LiveKit job — is among the first evicted once incidental candidates flood the
# registry (every ``httpx``/``requests`` span is one, none of those libraries being
# in ``DEFAULT_INSTRUMENTS_FOR_ROOT``).  Losing that one entry does not merely skip
# a reparent: it puts the span back in the export as a second root of a trace that
# was re-rooted precisely to be rid of it.  Explicit marks are therefore evicted
# last — see ``_evict_one_for_overflow``.
#
# Bounded by the number of LiveKit jobs in flight in the process (one entry each,
# reclaimed by the TTL sweep once the job ends), and in any case never allowed to
# push the registry past ``_MAX_ROOT_CANDIDATES``.
PINNED_ROOT_BLOCK_CANDIDATES: Set[int] = set()


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
    libraries are subject to the allow-list: those whose scope carries the
    ``opentelemetry.instrumentation.*`` / ``netra.instrumentation.*`` prefix, plus
    the third-party scopes named in ``THIRD_PARTY_INSTRUMENTATION_SCOPES`` (e.g.
    ``livekit-agents``), which Netra enables but does not author and which
    therefore do not follow that naming convention.

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
        """Start the TTL clock for a just-ended candidate, then prune expired entries.

        Entries are **not** cleared per-span — once a candidate ends its entry
        survives for ``_ROOT_CANDIDATE_TTL_SECONDS`` so that children exporting
        after their (already-ended) blocked ancestor can still be reparented.
        The TTL is measured from the *end* time, so an active long-lived root is
        never evicted before it finishes.

        Args:
            span: The span that is being ended.
        """
        try:
            self._mark_ended(span)
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
        instr_name = self._resolve_instrument_name(span)
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

        The marker is a plain instance attribute (see ``ROOT_BLOCK_CANDIDATE_FIELD``),
        not an OTel span attribute, so it cannot be evicted by the span attribute
        limit and is never exported.

        Args:
            span: The candidate span being started.
        """
        try:
            setattr(span, ROOT_BLOCK_CANDIDATE_FIELD, True)
        except Exception:
            pass

    @staticmethod
    def _record_candidate(span_id: int, parent_ctx: Optional[SpanContext], *, pinned: bool = False) -> None:
        """Register *span_id* as an **active** root-block candidate with its parent.

        The entry's TTL clock is left unstarted (``ended_at = None``) until the
        span ends, so a candidate that stays open longer than the TTL is never
        evicted while still active.

        Args:
            span_id: The candidate span's own span ID.
            parent_ctx: The candidate's parent ``SpanContext`` (``None`` when it
                is a trace root).
            pinned: Whether this entry was recorded by an explicit
                :func:`mark_as_root_block_candidate` call, which makes it the last
                thing overflow eviction reclaims. See
                ``PINNED_ROOT_BLOCK_CANDIDATES``.
        """
        with _root_candidates_lock:
            ROOT_BLOCK_CANDIDATES[span_id] = (parent_ctx, None)
            ROOT_BLOCK_CANDIDATES.move_to_end(span_id)
            if pinned:
                PINNED_ROOT_BLOCK_CANDIDATES.add(span_id)
            while len(ROOT_BLOCK_CANDIDATES) > _MAX_ROOT_CANDIDATES:
                _evict_one_for_overflow()

    @staticmethod
    def _refresh_candidate_parent(span_id: int, parent_ctx: Optional[SpanContext]) -> None:
        """Rewrite the parent recorded for *span_id*, leaving its TTL state alone.

        Only touches an entry that already exists: a span whose instrumentation is
        allowed to emit roots has none, and must not gain one here.

        Args:
            span_id: The candidate span's own span ID.
            parent_ctx: The parent ``SpanContext`` the span now has (``None`` once
                it has been promoted to a trace root).
        """
        with _root_candidates_lock:
            entry = ROOT_BLOCK_CANDIDATES.get(span_id)
            if entry is not None:
                ROOT_BLOCK_CANDIDATES[span_id] = (parent_ctx, entry[1])

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
    def _get_own_span_id(span: ReadableSpan) -> Optional[int]:
        """Return *span*'s own ``span_id``.

        Typed against ``ReadableSpan`` — the wider of the two, and all this reads —
        so it also serves the ended-span snapshot ``on_end`` is handed.

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
    def _resolve_instrument_name(span: Span) -> Optional[str]:
        """Return the instrument name *span* is subject to, or ``None``.

        Answers both "did an auto-instrumentation library produce this span?" and
        "which instrument is it?" from the single scope string, deliberately in
        one function. As two separate predicates they could disagree about a
        scope, and a scope that the first accepted but the second could not name
        would slip past the allow-list unchecked.

        A scope named ``netra.instrumentation.fastapi`` resolves to ``fastapi``.
        A third-party scope registered in ``THIRD_PARTY_INSTRUMENTATION_SCOPES``
        resolves to its ``InstrumentSet`` value — ``livekit-agents`` to
        ``livekit`` — which is what brings those spans under ``root_instruments``
        control despite the non-conforming scope name.

        Args:
            span: The span to inspect.

        Returns:
            The short instrumentation name, or ``None`` when the scope belongs to
            no recognised instrumentation — a netra decorator, ``Netra.start_span``
            or any user tracer — in which case the span is never a candidate.
        """
        scope = getattr(span, "instrumentation_scope", None)
        if scope is None:
            return None
        name = getattr(scope, "name", None)
        if not isinstance(name, str) or not name:
            return None

        # Exact-match aliases first: a third-party scope carries no prefix, so
        # the two branches cannot both claim the same name.
        alias = THIRD_PARTY_INSTRUMENTATION_SCOPES.get(name)
        if alias is not None:
            return alias

        if name.startswith(_INSTRUMENTATION_PREFIXES):
            base = name.rsplit(".", 1)[-1].strip()
            return base if base else name

        return None

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
                span_id, (_, ended_at) = next(iter(ROOT_BLOCK_CANDIDATES.items()))
                if ended_at is None or ended_at > cutoff:
                    break
                ROOT_BLOCK_CANDIDATES.popitem(last=False)
                PINNED_ROOT_BLOCK_CANDIDATES.discard(span_id)


def _evict_one_for_overflow() -> None:
    """Drop exactly one entry to bring the registry back under its size cap.

    Prefers the oldest *unpinned* entry, so a flood of incidental candidates
    cannot displace the handful of spans an instrumentation deliberately named
    (see ``PINNED_ROOT_BLOCK_CANDIDATES``).  When every entry is pinned the oldest
    is taken anyway: the size cap is a hard memory bound, and a registry that
    could grow past it is a worse failure than a lost mark.

    The caller MUST hold ``_root_candidates_lock``.
    """
    oldest_unpinned = next(
        (span_id for span_id in ROOT_BLOCK_CANDIDATES if span_id not in PINNED_ROOT_BLOCK_CANDIDATES),
        None,
    )
    if oldest_unpinned is not None:
        del ROOT_BLOCK_CANDIDATES[oldest_unpinned]
        return

    span_id, _ = ROOT_BLOCK_CANDIDATES.popitem(last=False)
    PINNED_ROOT_BLOCK_CANDIDATES.discard(span_id)


def mark_as_root_block_candidate(span: Span) -> None:
    """Record *span* as a root-block candidate, whatever instrumentation produced it.

    :class:`RootInstrumentFilterProcessor` decides candidacy per *instrumentation*,
    which is the right granularity for "may this library emit root spans at all".
    It cannot express "this library may emit roots, except for this one span" — the
    case where a library emits a wrapper span that makes a poor trace root (it ends
    long before the subtree it opened) while the same library owns the span that
    *should* be the root. Naming that span here hands it to the exporter's existing
    drop-and-reparent path, instead of forcing the whole library out of the root
    allow-list and peeling its entire tree.

    Marking a span is on its own **not** enough to re-root a trace. The exporter
    reparents a dropped span's children onto its parent, and a child that exports
    after the registry entry was TTL-evicted has nothing left to reparent onto. A
    caller that needs one specific child promoted deterministically must clear that
    child's parent itself, and then call :func:`refresh_root_block_candidate_parent`
    on it so a registry entry recorded before the clearing does not go on describing
    a parent link the span no longer has.

    The entry is *pinned*: recorded here means recorded on purpose, about a span the
    caller has already reasoned about, so it outranks the incidental candidates that
    every disallowed instrumentation contributes. See
    ``PINNED_ROOT_BLOCK_CANDIDATES`` for what that protects against.

    Args:
        span: The span to record. Both marked and registered, exactly as
            ``RootInstrumentFilterProcessor.on_start`` does for a disallowed
            instrumentation — see ``ROOT_BLOCK_CANDIDATE_FIELD`` for which of the two
            the exporter actually reads.
    """
    span_id = RootInstrumentFilterProcessor._get_own_span_id(span)
    if span_id is None or span_id == INVALID_SPAN_ID:
        return
    RootInstrumentFilterProcessor._mark_candidate(span)
    RootInstrumentFilterProcessor._record_candidate(
        span_id, RootInstrumentFilterProcessor._get_parent_span_context(span), pinned=True
    )


def unmark_root_block_candidate(span: ReadableSpan) -> None:
    """Undo a :func:`mark_as_root_block_candidate`, so *span* exports normally again.

    For a caller that must mark a span *provisionally* — early enough that children
    exporting before the decision is made are still reparented past it — and can
    only tell whether the drop was warranted once the span ends.

    Clears the registry entry only.  ``ROOT_BLOCK_CANDIDATE_FIELD`` is left on the
    live span, which is harmless: the exporter never sees it (see that constant),
    and the caller here is handed a ``ReadableSpan`` snapshot rather than the span
    the field lives on.

    Args:
        span: The span that should no longer be dropped.
    """
    span_id = RootInstrumentFilterProcessor._get_own_span_id(span)
    if span_id is None or span_id == INVALID_SPAN_ID:
        return
    with _root_candidates_lock:
        ROOT_BLOCK_CANDIDATES.pop(span_id, None)
        PINNED_ROOT_BLOCK_CANDIDATES.discard(span_id)


def refresh_root_block_candidate_parent(span: Span) -> None:
    """Re-read *span*'s current parent into its registry entry, if it has one.

    ``RootInstrumentFilterProcessor.on_start`` snapshots a candidate's parent at
    span start, and it runs before any instrumentation's own processors — so an
    instrumentation that re-roots a span by clearing its parent leaves the registry
    describing a link that no longer exists.  Since the registry, not the span, is
    what the exporter reads at export time, that stale link is what the drop-and-
    reparent walk would act on.

    A no-op for a span with no entry, which is the common case: an instrumentation
    on the root allow-list never becomes a candidate.

    Args:
        span: The span whose parent link has just been rewritten.
    """
    span_id = RootInstrumentFilterProcessor._get_own_span_id(span)
    if span_id is None or span_id == INVALID_SPAN_ID:
        return
    RootInstrumentFilterProcessor._refresh_candidate_parent(
        span_id, RootInstrumentFilterProcessor._get_parent_span_context(span)
    )
