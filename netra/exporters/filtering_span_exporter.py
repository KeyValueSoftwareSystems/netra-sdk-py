import logging
from typing import Any, Dict, List, Sequence, Set

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from netra.exporters.utils import (
    PatternMatcher,
    add_blocked_trace_id,
    find_root_spans_blocked,
    get_span_id,
    get_trace_id,
    has_local_block_flag,
    is_trace_id_blocked,
    is_trial_blocked,
    normalize_parent,
    read_local_block_patterns,
    reparent_spans,
)
from netra.processors.local_filtering_span_processor import BLOCKED_LOCAL_PARENT_MAP

logger = logging.getLogger(__name__)


class FilteringSpanExporter(SpanExporter):  # type: ignore[misc]
    """SpanExporter wrapper that drops spans by name and by root-instrument policy.

    A span is dropped when any of the following holds:
    - its trace ID was blocked while a trial/quota block was active;
    - its name matches a globally configured block pattern (see ``PatternMatcher``);
    - its name matches a per-span local block pattern set by
      ``LocalFilteringSpanProcessor``;
    - it is a root-connected span from an instrumentation not allowed to emit
      root spans (resolved by ``find_root_spans_blocked``).

    Children of a dropped span are reparented onto the dropped span's parent so
    subtrees are never silently discarded.
    """

    def __init__(self, exporter: SpanExporter, patterns: Sequence[str]) -> None:
        """
        Initialize the filtering span exporter.

        Args:
            exporter: The underlying span exporter to forward surviving spans to.
            patterns: Global name patterns to block.
        """
        self._exporter = exporter
        self._matcher = PatternMatcher(patterns)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Filter *spans*, reparent survivors, and forward them to the wrapped exporter.

        Args:
            spans: The batch of spans to export.

        Returns:
            The wrapped exporter's ``SpanExportResult``, or ``SUCCESS`` when the
            batch is fully filtered (nothing left to forward).
        """
        if is_trial_blocked():
            logger.debug("Trial/quota exhausted: blocking spans from export")
            self._record_blocked_trace_ids(spans)
            return SpanExportResult.SUCCESS

        # Find unconditional name/local drops first, then resolve which
        # root-block candidates are root-connected. Feeding the name/local drops
        # into that walk is what stops a disallowed candidate from being promoted
        # to a root when its only surviving ancestor is itself name-blocked.
        parents_of_spans_blocked_by_name = self._find_spans_blocked_by_name(spans)
        root_spans_blocked, parents_of_root_spans_blocked = find_root_spans_blocked(
            spans, parents_of_spans_blocked_by_name
        )

        all_blocked_span_ids: Set[Any] = set(parents_of_spans_blocked_by_name)
        all_blocked_span_ids.update(root_spans_blocked)

        surviving_spans = self._collect_survivors(spans, all_blocked_span_ids)

        reparent_map = self._build_reparent_map(parents_of_root_spans_blocked, parents_of_spans_blocked_by_name)
        if reparent_map:
            reparent_spans(surviving_spans, reparent_map)
        if not surviving_spans:
            return SpanExportResult.SUCCESS
        return self._exporter.export(surviving_spans)

    def _record_blocked_trace_ids(self, spans: Sequence[ReadableSpan]) -> None:
        """Remember the trace IDs seen during a block so their later spans are dropped too.

        Args:
            spans: The batch of spans being dropped during the block.
        """
        for span in spans:
            trace_id = get_trace_id(span)
            if trace_id:
                add_blocked_trace_id(trace_id)

    def _find_spans_blocked_by_name(self, spans: Sequence[ReadableSpan]) -> Dict[Any, Any]:
        """Find the spans dropped unconditionally by a global or local name rule.

        Trace-blocked spans are skipped entirely (they are dropped without
        reparenting). The result feeds both the root-connected candidate walk —
        as transparent dropped ancestors — and the reparent map.

        Args:
            spans: The batch of spans to classify.

        Returns:
            A ``{dropped_span_id -> normalized_parent}`` map for name/locally
            blocked spans in this batch.
        """
        parent_map: Dict[Any, Any] = {}
        for span in spans:
            trace_id = get_trace_id(span)
            if trace_id and is_trace_id_blocked(trace_id):
                continue

            name = getattr(span, "name", None)
            if name is None:
                continue

            if self._matcher.matches(name) or self._is_locally_blocked(span, name):
                span_id = get_span_id(span)
                if span_id is not None:
                    parent_map[span_id] = normalize_parent(getattr(span, "parent", None))

        return parent_map

    def _collect_survivors(self, spans: Sequence[ReadableSpan], all_blocked_span_ids: Set[Any]) -> List[ReadableSpan]:
        """Return the spans that survive filtering.

        Args:
            spans: The batch of spans to filter.
            all_blocked_span_ids: Span IDs dropped by name/local rules or by the
                root-instrument policy.

        Returns:
            The spans to forward to the wrapped exporter (before reparenting).
        """
        surviving_spans: List[ReadableSpan] = []
        for span in spans:
            trace_id = get_trace_id(span)
            if trace_id and is_trace_id_blocked(trace_id):
                continue

            span_id = get_span_id(span)
            if span_id is not None and span_id in all_blocked_span_ids:
                continue

            surviving_spans.append(span)

        return surviving_spans

    def _is_locally_blocked(self, span: ReadableSpan, name: str) -> bool:
        """Check whether *span* is blocked by its per-span local rules.

        Args:
            span: The span carrying the local-block attributes.
            name: The span name to match against local patterns.

        Returns:
            ``True`` if *name* matches a local pattern or the span carries the
            local-block flag; ``False`` on any read error.
        """
        try:
            local_patterns = read_local_block_patterns(span)
            if local_patterns and PatternMatcher(local_patterns).matches(name):
                return True
            return has_local_block_flag(span)
        except Exception:
            return False

    def _build_reparent_map(
        self, parents_of_root_spans_blocked: Dict[int, Any], parents_of_spans_blocked_by_name: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        """Merge the cross-batch registry with this batch's dropped-parent maps.

        Ordering matters: the process-global registry captured by processors is
        the base, overlaid by root-blocked parents, then this batch's name/local
        blocks — so in-batch parents win on conflict.

        Args:
            parents_of_root_spans_blocked: Root-blocked span IDs to their parents.
            parents_of_spans_blocked_by_name: This batch's name/locally blocked
                span IDs to their parents.

        Returns:
            The merged ``{dropped_span_id -> parent}`` map used for reparenting.
        """
        reparent_map: Dict[Any, Any] = {}
        try:
            if BLOCKED_LOCAL_PARENT_MAP:
                reparent_map.update(BLOCKED_LOCAL_PARENT_MAP)
        except Exception:
            pass
        reparent_map.update(parents_of_root_spans_blocked)
        reparent_map.update(parents_of_spans_blocked_by_name)
        return reparent_map

    def shutdown(self) -> None:
        """Shutdown the wrapped exporter, suppressing shutdown errors."""
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        """Force flush the wrapped exporter.

        Args:
            timeout_millis: The flush timeout in milliseconds.

        Returns:
            The wrapped exporter's flush result, or ``True`` if it raised.
        """
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            return True
