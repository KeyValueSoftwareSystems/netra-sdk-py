import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import INVALID_SPAN_ID, SpanContext

from netra.config import Config
from netra.processors.root_instrument_filter_processor import ROOT_BLOCK_CANDIDATE_ATTR, get_root_block_candidates

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
                    Config.TRIAL_BLOCK_DURATION_SECONDS,
                )
            else:
                elapsed = time.time() - _trial_blocked_at
                remaining = Config.TRIAL_BLOCK_DURATION_SECONDS - elapsed
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
        if elapsed >= Config.TRIAL_BLOCK_DURATION_SECONDS:
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

    Args:
        span: The span to check.

    Returns:
        ``True`` if the span was marked as a root-block candidate at start.
    """
    return bool(_read_attribute(span, ROOT_BLOCK_CANDIDATE_ATTR))


def strip_root_block_candidate_marker(span: ReadableSpan) -> None:
    """Remove the internal candidacy marker so it never reaches the backend.

    Mutates the underlying ``_attributes`` mapping (``ReadableSpan.attributes``
    is a read-only proxy), falling back to a mutable ``attributes`` mapping.

    Args:
        span: The span to clean before export.
    """
    try:
        attrs = getattr(span, "_attributes", None)
        if attrs is None or not hasattr(attrs, "__delitem__"):
            attrs = getattr(span, "attributes", None)
        if attrs is not None and hasattr(attrs, "__delitem__") and ROOT_BLOCK_CANDIDATE_ATTR in attrs:
            del attrs[ROOT_BLOCK_CANDIDATE_ATTR]
    except Exception:
        logger.debug("Failed to strip root-block-candidate marker", exc_info=True)


def resolve_root_dropped(spans: Sequence[ReadableSpan]) -> Tuple[Set[int], Dict[int, Optional[SpanContext]]]:
    """Resolve the root-block candidates in *spans* that must be dropped.

    A candidate is dropped when it is *root-connected*: it is a trace root (no
    local parent), or its parent is itself a dropped candidate. The peel stops
    at the first surviving ancestor — an allowed instrument, a netra/manual
    span, or a non-instrumentation span — and never crosses a remote
    (cross-process) parent link.

    The candidate set is the registry snapshot overlaid with any span in the
    current batch that carries the durable candidacy marker. The overlay makes
    the decision robust: a blocked root still in the batch is dropped even if
    its registry entry was evicted (TTL/overflow) or cleared, because the marker
    travels with the span. Only ancestor chains reachable from this batch are
    evaluated, so the cost is proportional to the batch (plus its ancestry)
    rather than to the whole registry.

    Args:
        spans: The batch of spans being exported.

    Returns:
        ``(dropped_span_ids, dropped_parent_map)`` where ``dropped_parent_map``
        maps each dropped span ID to its parent ``SpanContext`` (``None`` for a
        true root) for reparenting.
    """
    candidates = _collect_candidates(spans)
    if not candidates:
        return set(), {}

    dropped = _peel_root_connected(spans, candidates)
    dropped_parent_map: Dict[int, Optional[SpanContext]] = {sid: candidates[sid] for sid in dropped}
    return dropped, dropped_parent_map


def _collect_candidates(spans: Sequence[ReadableSpan]) -> Dict[int, Optional[SpanContext]]:
    """Merge the registry snapshot with in-batch candidacy markers.

    The in-batch overlay lets a marked span be recognised even if its registry
    entry was evicted or cleared before export.

    Args:
        spans: The batch of spans being exported.

    Returns:
        A ``{span_id -> parent_span_context}`` map of every known candidate,
        with in-batch markers overriding the registry snapshot.
    """
    candidates = get_root_block_candidates()
    for span in spans:
        if not is_root_block_candidate(span):
            continue
        span_id = get_span_id(span)
        if span_id is None or span_id == INVALID_SPAN_ID:
            continue
        candidates[span_id] = normalize_parent(getattr(span, "parent", None))
    return candidates


def _peel_root_connected(spans: Sequence[ReadableSpan], candidates: Dict[int, Optional[SpanContext]]) -> Set[int]:
    """Return the candidate span IDs that are root-connected (and so dropped).

    Traversal is seeded only from batch spans and their parents, so unrelated
    registry entries are never walked.

    Args:
        spans: The batch of spans being exported.
        candidates: The ``{span_id -> parent_span_context}`` candidate map from
            :func:`_collect_candidates`.

    Returns:
        The set of candidate span IDs that are root-connected.
    """
    memo: Dict[int, bool] = {}
    dropped: Set[int] = set()

    def is_dropped(span_id: int, visiting: Set[int]) -> bool:
        """Resolve whether *span_id* is a root-connected candidate, memoized.

        Args:
            span_id: The span ID to resolve.
            visiting: The span IDs on the current recursion path (cycle guard).

        Returns:
            ``True`` if *span_id* is a candidate whose ancestry peels to a root.
        """
        if span_id in memo:
            return memo[span_id]
        if span_id not in candidates:
            # Not a candidate -> a surviving ancestor. Stops the peel.
            return False
        if span_id in visiting:
            # Cycle guard: treat as non-dropped to avoid infinite recursion.
            return False

        parent_ctx = candidates[span_id]
        if parent_ctx is None:
            result = True  # candidate is a true trace root
        elif getattr(parent_ctx, "is_remote", False):
            result = False  # do not peel across a process boundary
        else:
            parent_id = getattr(parent_ctx, "span_id", None)
            if parent_id is None or parent_id == INVALID_SPAN_ID:
                result = True
            else:
                visiting.add(span_id)
                result = is_dropped(parent_id, visiting)
                visiting.discard(span_id)

        memo[span_id] = result
        if result:
            dropped.add(span_id)
        return result

    for span in spans:
        span_id = get_span_id(span)
        if span_id is not None and span_id != INVALID_SPAN_ID:
            is_dropped(span_id, set())
        parent_ctx = getattr(span, "parent", None)
        parent_id = getattr(parent_ctx, "span_id", None) if parent_ctx is not None else None
        if parent_id is not None and parent_id != INVALID_SPAN_ID:
            is_dropped(parent_id, set())

    return dropped


def reparent_spans(spans: Sequence[ReadableSpan], blocked_parent_map: Dict[Any, Any]) -> None:
    """Reparent each span past any dropped ancestor onto its first survivor.

    Walks the chain of blocked parents (following ``blocked_parent_map``) until
    a surviving parent — or ``None`` (promote to root) — is reached.

    Args:
        spans: The surviving spans whose parents may need rewriting.
        blocked_parent_map: A ``{blocked_span_id -> parent}`` map of dropped
            spans to their own parents.
    """
    if not blocked_parent_map:
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
            if parent_span_id not in blocked_parent_map or parent_span_id in visited:
                break
            visited.add(parent_span_id)
            new_parent = blocked_parent_map[parent_span_id]
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
