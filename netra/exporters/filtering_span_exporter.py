import logging
from typing import Any, Dict, List, Sequence, Set, Tuple

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from netra.exporters.utils import (
    PatternMatcher,
    add_blocked_trace_id,
    get_span_id,
    get_trace_id,
    has_local_block_flag,
    is_trace_id_blocked,
    is_trial_blocked,
    read_local_block_patterns,
    reparent_spans,
    resolve_root_dropped,
    strip_root_block_candidate_marker,
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
      root spans (resolved by ``resolve_root_dropped``).

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

        # Resolve which root-block candidates are root-connected (and must be
        # dropped) once per batch.
        root_dropped, root_dropped_parent_map = resolve_root_dropped(spans)

        kept, blocked_parent_map = self._partition(spans, root_dropped)

        merged_map = self._merge_parent_maps(root_dropped_parent_map, blocked_parent_map)
        if merged_map:
            reparent_spans(kept, merged_map)
        if not kept:
            return SpanExportResult.SUCCESS
        return self._exporter.export(kept)

    def _record_blocked_trace_ids(self, spans: Sequence[ReadableSpan]) -> None:
        """Remember the trace IDs seen during a block so their later spans are dropped too.

        Args:
            spans: The batch of spans being dropped during the block.
        """
        for span in spans:
            trace_id = get_trace_id(span)
            if trace_id:
                add_blocked_trace_id(trace_id)

    def _partition(
        self, spans: Sequence[ReadableSpan], root_dropped: Set[int]
    ) -> Tuple[List[ReadableSpan], Dict[Any, Any]]:
        """Split *spans* into survivors and a map of dropped span IDs to their parents.

        The dropped-parent map only records name/locally blocked spans; root-dropped
        spans are already covered by ``resolve_root_dropped``'s parent map.

        Args:
            spans: The batch of spans to classify.
            root_dropped: Span IDs resolved as root-connected and to be dropped.

        Returns:
            ``(kept, blocked_parent_map)`` — the surviving spans, and a
            ``{blocked_span_id -> parent}`` map for reparenting their children.
        """
        kept: List[ReadableSpan] = []
        blocked_parent_map: Dict[Any, Any] = {}

        for span in spans:
            trace_id = get_trace_id(span)
            if trace_id and is_trace_id_blocked(trace_id):
                continue

            span_id = get_span_id(span)
            root_blocked = span_id is not None and span_id in root_dropped

            # Strip the internal candidacy marker so it never leaks onto a
            # surviving (kept) span in the exported output.
            strip_root_block_candidate_marker(span)

            name = getattr(span, "name", None)
            if name is None:
                if not root_blocked:
                    kept.append(span)
                continue

            if not (self._matcher.matches(name) or self._is_locally_blocked(span, name) or root_blocked):
                kept.append(span)
                continue

            # Blocked: record its parent so children in this or a later batch can
            # be reparented past it.
            if span_id is not None:
                blocked_parent_map[span_id] = getattr(span, "parent", None)

        return kept, blocked_parent_map

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

    def _merge_parent_maps(
        self, root_dropped_parent_map: Dict[int, Any], blocked_parent_map: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        """Merge the cross-batch registry with this batch's dropped-parent maps.

        Ordering matters: the process-global registry captured by processors is
        the base, overlaid by root-dropped parents, then this batch's name/local
        blocks — so in-batch parents win on conflict.

        Args:
            root_dropped_parent_map: Dropped-root span IDs to their parents.
            blocked_parent_map: This batch's name/locally blocked span IDs to
                their parents.

        Returns:
            The merged ``{blocked_span_id -> parent}`` map used for reparenting.
        """
        merged_map: Dict[Any, Any] = {}
        try:
            if BLOCKED_LOCAL_PARENT_MAP:
                merged_map.update(BLOCKED_LOCAL_PARENT_MAP)
        except Exception:
            pass
        merged_map.update(root_dropped_parent_map)
        merged_map.update(blocked_parent_map)
        return merged_map

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
