import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import INVALID_SPAN_ID, SpanContext

from netra.config import get_trial_block_duration_seconds
from netra.processors.root_instrument_filter_processor import (
    ROOT_BLOCK_CANDIDATE_FIELD,
    get_root_block_candidates,
    root_block_candidates_contains,
)

logger = logging.getLogger(__name__)

# Span attributes set by LocalFilteringSpanProcessor and read back here to
# decide per-span local blocking (kept in sync with that processor's keys).
LOCAL_BLOCKED_SPANS_ATTR = "netra.local_blocked_spans"
LOCAL_BLOCKED_FLAG_ATTR = "netra.local_blocked"

_trial_status_lock = threading.Lock()
_trial_blocked_at: Optional[float] = None
_blocked_trace_ids: Set[str] = set()


def set_trial_blocked(blocked: bool) -> None:
    """Set the trial blocked status with automatic expiration after 15 minutes.

    When called with blocked=True, starts a timer. All span exports will be blocked
    for 15 minutes. After 15 minutes, exports automatically resume even if this
    function is not called again.

    Args:
        blocked: True to start the 15-minute blocking period, False to reset
    """
    global _trial_blocked_at
    with _trial_status_lock:
        if blocked:
            if _trial_blocked_at is None:
                # Start a new 15-minute blocking period
                _trial_blocked_at = time.time()
                logger.warning(
                    "Trial/quota exhausted: blocking span export for %d seconds (15 minutes)",
                    get_trial_block_duration_seconds(),
                )
            else:
                elapsed = time.time() - _trial_blocked_at
                remaining = get_trial_block_duration_seconds() - elapsed
                logger.debug("Trial already blocked: %d seconds remaining", max(0, int(remaining)))
        else:
            if _trial_blocked_at is not None:
                logger.info("Trial blocking manually reset")
            _trial_blocked_at = None


def is_trial_blocked() -> bool:
    """Check if trial is currently blocked.

    Automatically returns False after 15 minutes have passed, even if
    set_trial_blocked(True) was never called again.

    Returns:
        True if currently within the 15-minute blocking period, False otherwise
    """
    global _trial_blocked_at

    with _trial_status_lock:
        if _trial_blocked_at is None:
            return False

        # Check if 15 minutes have passed since blocking started
        elapsed = time.time() - _trial_blocked_at
        if elapsed >= get_trial_block_duration_seconds():
            _trial_blocked_at = None
            logger.info("Trial blocking period (15 minutes) expired, resuming exports")
            return False

        return True


def add_blocked_trace_id(trace_id: str) -> None:
    """Add a trace ID to the blocked list.

    Trace IDs that started during the blocking period should be added to this list.
    All spans from these trace IDs will be filtered out, even after the 15-minute
    block expires. Only new trace IDs created after block expiration will be exported.

    Args:
        trace_id: The trace ID to block (format: hex string)
    """
    with _trial_status_lock:
        _blocked_trace_ids.add(trace_id)
        logger.debug("Added trace ID to blocked list: %s (total blocked: %d)", trace_id, len(_blocked_trace_ids))


def is_trace_id_blocked(trace_id: str) -> bool:
    """Check if a trace ID is in the blocked list.

    Args:
        trace_id: The trace ID to check

    Returns:
        True if this trace ID should be filtered, False otherwise
    """
    with _trial_status_lock:
        return trace_id in _blocked_trace_ids


def get_trace_id(span: ReadableSpan) -> str:
    """Extract trace ID from span.

    Args:
        span: The span to extract trace ID from

    Returns:
        Trace ID as hex string, or empty string if not found
    """
    try:
        context = getattr(span, "context", None)
        if context is None:
            return ""

        trace_id = getattr(context, "trace_id", None)
        if trace_id is None:
            return ""

        # trace_id is typically an integer, convert to hex string
        if isinstance(trace_id, int):
            return format(trace_id, "032x")
        else:
            return str(trace_id)
    except Exception as e:
        logger.debug("Error extracting trace ID from span: %s", e)
        return ""


class PatternMatcher:
    """Matches span names against a set of block patterns.

    Pattern grammar:
    - ``"Foo"``     exact match: ``name == "Foo"``.
    - ``"Foo.*"``   prefix match: ``name`` starts with ``"Foo."``.
    - ``"*.Query"`` suffix match: ``name`` ends with ``".Query"``.

    A pattern that both starts and ends with ``*`` is treated as an exact match;
    empty patterns are ignored. Patterns are classified once at construction so
    matching is a constant-time set lookup plus tuple ``startswith``/``endswith``.
    """

    def __init__(self, patterns: Sequence[str]) -> None:
        """Classify *patterns* into exact / prefix / suffix buckets once.

        Args:
            patterns: The block patterns to compile. Empty patterns are ignored.
        """
        exact: Set[str] = set()
        prefixes: List[str] = []
        suffixes: List[str] = []
        for pattern in patterns:
            if not pattern:
                continue
            if pattern.endswith("*") and not pattern.startswith("*"):
                prefixes.append(pattern[:-1])
            elif pattern.startswith("*") and not pattern.endswith("*"):
                suffixes.append(pattern[1:])
            else:
                exact.add(pattern)
        self._exact = exact
        # str.startswith / str.endswith accept a tuple of candidates.
        self._prefixes = tuple(prefixes)
        self._suffixes = tuple(suffixes)

    def matches(self, name: str) -> bool:
        """Check whether *name* matches any configured pattern.

        Args:
            name: The span name to test.

        Returns:
            ``True`` if *name* matches an exact, prefix, or suffix pattern.
        """
        return name in self._exact or name.startswith(self._prefixes) or name.endswith(self._suffixes)


def get_span_id(span: ReadableSpan) -> Optional[int]:
    """Return *span*'s own span ID.

    Args:
        span: The span to inspect.

    Returns:
        The integer span ID, or ``None`` if unavailable.
    """
    span_context = getattr(span, "context", None)
    if span_context is None:
        return None
    return cast(Optional[int], getattr(span_context, "span_id", None))


def normalize_parent(parent: Any) -> Optional[SpanContext]:
    """Normalize a span's parent link, collapsing an invalid parent to ``None``.

    Mirrors ``RootInstrumentFilterProcessor._get_parent_span_context`` so the
    in-batch overlay and the registry store parents identically. A ``None``
    result means the span is a trace root.

    Args:
        parent: The span's ``parent`` ``SpanContext`` (or ``None``).

    Returns:
        The parent ``SpanContext``, or ``None`` when the span is a trace root.
    """
    if parent is None:
        return None
    parent_id = getattr(parent, "span_id", None)
    if parent_id is None or parent_id == INVALID_SPAN_ID:
        return None
    return cast(Optional[SpanContext], parent)


def _read_attribute(span: ReadableSpan, key: str) -> Any:
    """Read attribute *key* off *span*, tolerating read-only proxies and misses.

    ``ReadableSpan.attributes`` may be a plain mapping or a read-only proxy.

    Args:
        span: The span to read the attribute from.
        key: The attribute key to look up.

    Returns:
        The attribute value, or ``None`` when it is absent or unreadable.
    """
    attrs = getattr(span, "attributes", None)
    if not attrs:
        return None
    try:
        if hasattr(attrs, "get"):
            return attrs.get(key)
        return attrs[key]
    except Exception:
        return None


def read_local_block_patterns(span: ReadableSpan) -> List[str]:
    """Read the per-span local-block patterns set by ``LocalFilteringSpanProcessor``.

    Args:
        span: The span to read patterns from.

    Returns:
        The non-empty string patterns, or an empty list if none are set.
    """
    value = _read_attribute(span, LOCAL_BLOCKED_SPANS_ATTR)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return [v for v in value if v]
    return []


def has_local_block_flag(span: ReadableSpan) -> bool:
    """Check whether the processor explicitly flagged *span* as locally blocked.

    Args:
        span: The span to check.

    Returns:
        ``True`` if the span carries a truthy local-block flag.
    """
    return bool(_read_attribute(span, LOCAL_BLOCKED_FLAG_ATTR))


def is_root_block_candidate(span: ReadableSpan) -> bool:
    """Check whether *span* carries the durable root-block-candidate marker.

    The marker is a plain instance attribute (not an OTel span attribute), so it
    cannot be evicted by the span attribute limit.

    Args:
        span: The span to check.

    Returns:
        ``True`` if the span was marked as a root-block candidate at start.
    """
    return bool(getattr(span, ROOT_BLOCK_CANDIDATE_FIELD, False))


def find_root_spans_blocked(
    spans: Sequence[ReadableSpan],
    parents_of_spans_blocked_by_name: Optional[Dict[int, Optional[SpanContext]]] = None,
) -> Tuple[Set[int], Dict[int, Optional[SpanContext]]]:
    """Find the root-block candidates in *spans* that must be dropped.

    A candidate is dropped when it is *root-connected*: it is a trace root (no
    local parent), or every ancestor between it and the trace root is itself
    dropped. Walking up the ancestor chain stops at the first surviving ancestor
    — an allowed instrument, a netra/manual span, or a non-instrumentation span —
    and never crosses a remote (cross-process) parent link.

    ``parents_of_spans_blocked_by_name`` are spans dropped for *other* reasons (a
    global or per-span local name block) that are removed from the exported tree
    just like candidates. Folding them in lets the walk "see through" them: a
    candidate whose only surviving ancestor was a name-blocked span becomes
    root-connected once that span is dropped, and so must be dropped too rather
    than promoted to a root it is not allowed to be.

    The candidate set is the registry snapshot overlaid with any span in the
    current batch that carries the durable candidacy marker. The overlay makes
    the decision robust: a blocked root still in the batch is dropped even if
    its registry entry was evicted (TTL/overflow) or cleared, because the marker
    travels with the span. Only ancestor chains reachable from this batch are
    evaluated, so the cost is proportional to the batch (plus its ancestry)
    rather than to the whole registry.

    Args:
        spans: The batch of spans being exported.
        parents_of_spans_blocked_by_name: ``{span_id -> parent_span_context}`` for
            spans dropped by name/local rules, treated as transparent dropped nodes.

    Returns:
        ``(root_connected_ids, parents_of_root_spans_blocked)`` where
        ``parents_of_root_spans_blocked`` maps each dropped span ID to its parent
        ``SpanContext`` (``None`` for a true root) for reparenting.
    """
    candidates = _collect_root_block_candidates(spans)
    # Only genuine root-block candidates trigger a chain walk; name/local drops
    # alone (with no candidate in play) are handled by the caller directly.
    if not candidates:
        return set(), {}

    if parents_of_spans_blocked_by_name:
        for span_id, parent_ctx in parents_of_spans_blocked_by_name.items():
            candidates.setdefault(span_id, parent_ctx)

    root_connected_ids = _find_root_connected_candidates(spans, candidates)
    parents_of_root_spans_blocked: Dict[int, Optional[SpanContext]] = {
        sid: candidates[sid] for sid in root_connected_ids
    }
    return root_connected_ids, parents_of_root_spans_blocked


def _collect_root_block_candidates(spans: Sequence[ReadableSpan]) -> Dict[int, Optional[SpanContext]]:
    """Merge in-batch candidacy markers with the cross-batch registry.

    The in-batch markers let a marked span be recognised even if its registry
    entry was evicted or cleared before export — but only where the exporter is
    handed the live span. The OTel SDK is not such a caller: ``Span.end()`` passes
    a fresh ``ReadableSpan`` built by ``_readable_span()``, which copies the span's
    fields and not its instance attributes, so ``ROOT_BLOCK_CANDIDATE_FIELD`` does
    not survive the hand-off. Under the SDK the registry is therefore the only
    source of candidacy, and it MUST be consulted for a batch span's own id as well
    as its parent's — a candidate root exporting in a batch that holds none of its
    children (its own batch, if it ends well before or after them, or every batch
    when ``disable_batch`` is set) would otherwise be exported as a root.

    Args:
        spans: The batch of spans being exported.

    Returns:
        A ``{span_id -> parent_span_context}`` map of every relevant candidate,
        with in-batch markers overriding the registry snapshot.
    """
    in_batch: Dict[int, Optional[SpanContext]] = {}
    referenced_ids: Set[int] = set()
    for span in spans:
        span_id = get_span_id(span)
        if span_id is not None and span_id != INVALID_SPAN_ID:
            referenced_ids.add(span_id)
            if is_root_block_candidate(span):
                in_batch[span_id] = normalize_parent(getattr(span, "parent", None))
        parent = getattr(span, "parent", None)
        parent_id = getattr(parent, "span_id", None) if parent is not None else None
        if parent_id is not None and parent_id != INVALID_SPAN_ID:
            referenced_ids.add(parent_id)

    # A registry entry is only worth walking if it belongs to a span in this batch
    # or is the parent of one (every in-batch candidate contributes its own parent
    # id here too, so multi-hop cross-batch chains are covered). Candidates already
    # resolvable in-batch never need the registry.
    if not root_block_candidates_contains(referenced_ids - set(in_batch)):
        return in_batch

    candidates = get_root_block_candidates()
    candidates.update(in_batch)
    return candidates


def _find_root_connected_candidates(
    spans: Sequence[ReadableSpan], candidates: Dict[int, Optional[SpanContext]]
) -> Set[int]:
    """Return the candidate span IDs that are root-connected (and so dropped).

    Traversal is seeded only from batch spans and their parents, so unrelated
    registry entries are never walked.

    Args:
        spans: The batch of spans being exported.
        candidates: The ``{span_id -> parent_span_context}`` candidate map from
            :func:`_collect_root_block_candidates`.

    Returns:
        The set of candidate span IDs that are root-connected.
    """
    memo: Dict[int, bool] = {}
    root_connected_ids: Set[int] = set()

    def resolves_to_root(start_id: int) -> bool:
        """Resolve whether *start_id* is a root-connected candidate, memoized.

        Walks the candidate ancestry chain iteratively (no recursion, so a
        deeply nested trace cannot blow the Python stack). Every node on a
        linear candidate chain shares the terminal verdict — the chain is
        dropped iff it walks all the way up to a true root — so the resolved
        result is written back to the whole walked path in one pass.

        Args:
            start_id: The span ID to resolve.

        Returns:
            ``True`` if *start_id* is a candidate whose ancestry reaches a root.
        """
        walked_ids: List[int] = []
        seen_ids: Set[int] = set()
        current_id = start_id
        while True:
            if current_id in memo:
                reaches_root = memo[current_id]
                break
            if current_id not in candidates:
                # Not a candidate -> a surviving ancestor. Stops the walk.
                reaches_root = False
                break
            if current_id in seen_ids:
                # Cycle guard: treat as non-dropped to avoid looping forever.
                reaches_root = False
                break

            # Fresh candidate node: it shares the chain's terminal verdict.
            walked_ids.append(current_id)
            seen_ids.add(current_id)

            parent_ctx = candidates[current_id]
            if parent_ctx is None:
                reaches_root = True  # candidate is a true trace root
                break
            if getattr(parent_ctx, "is_remote", False):
                reaches_root = False  # do not walk across a process boundary
                break
            parent_id = getattr(parent_ctx, "span_id", None)
            if parent_id is None or parent_id == INVALID_SPAN_ID:
                reaches_root = True
                break

            current_id = parent_id

        for span_id in walked_ids:
            memo[span_id] = reaches_root
            if reaches_root:
                root_connected_ids.add(span_id)
        return reaches_root

    for span in spans:
        span_id = get_span_id(span)
        if span_id is not None and span_id != INVALID_SPAN_ID:
            resolves_to_root(span_id)
        parent_ctx = getattr(span, "parent", None)
        parent_id = getattr(parent_ctx, "span_id", None) if parent_ctx is not None else None
        if parent_id is not None and parent_id != INVALID_SPAN_ID:
            resolves_to_root(parent_id)

    return root_connected_ids


def reparent_spans(spans: Sequence[ReadableSpan], dropped_span_parents: Dict[Any, Any]) -> None:
    """Reparent each span past any dropped ancestor onto its first survivor.

    Walks the chain of dropped parents (following ``dropped_span_parents``) until
    a surviving parent — or ``None`` (promote to root) — is reached.

    Args:
        spans: The surviving spans whose parents may need rewriting.
        dropped_span_parents: A ``{dropped_span_id -> parent}`` map of dropped
            spans to their own parents.
    """
    if not dropped_span_parents:
        return

    for span in spans:
        parent = getattr(span, "parent", None)
        if parent is None:
            continue

        new_parent = parent
        visited: Set[Any] = set()
        changed = False
        while new_parent is not None:
            parent_span_id = getattr(new_parent, "span_id", None)
            if parent_span_id not in dropped_span_parents or parent_span_id in visited:
                break
            visited.add(parent_span_id)
            new_parent = dropped_span_parents[parent_span_id]
            changed = True

        if changed:
            set_span_parent(span, new_parent)


def set_span_parent(span: ReadableSpan, parent: Any) -> None:
    """Set *span*'s parent, preferring the private ``_parent`` slot.

    Args:
        span: The span to reparent.
        parent: The new parent ``SpanContext`` (``None`` to promote to root).
    """
    if hasattr(span, "_parent"):
        try:
            span._parent = parent
            return
        except Exception:
            pass
    try:
        setattr(span, "parent", parent)
    except Exception:
        logger.debug("Failed to reparent span %s", getattr(span, "name", "<unknown>"), exc_info=True)
