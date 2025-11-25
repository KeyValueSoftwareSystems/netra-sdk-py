import logging
from typing import Any, Dict, List, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)

from netra.exporters.trial_status_manager import is_trial_blocked, is_trace_id_blocked, add_blocked_trace_id
from netra.processors.local_filtering_span_processor import (
    BLOCKED_LOCAL_PARENT_MAP,
)

logger = logging.getLogger(__name__)


class FilteringSpanExporter(SpanExporter):  # type: ignore[misc]
    """
    SpanExporter wrapper that filters out spans by name.

    Matching rules:
    - Exact match: pattern "Foo" blocks span.name == "Foo".
    - Prefix match: pattern ending with '*' (e.g., "CloudSpanner.*") blocks spans whose
      names start with the prefix before '*', e.g., "CloudSpanner.", "CloudSpanner.Query".
    - Suffix match: pattern starting with '*' (e.g., "*.Query") blocks spans whose
      names end with the suffix after '*', e.g., "DB.Query", "Search.Query".
    """

    def __init__(self, exporter: SpanExporter, patterns: Sequence[str]) -> None:
        """
        Initialize the filtering span exporter.

        Args:
            exporter: The span exporter to use.
            patterns: List of patterns to block.
        """
        self._exporter = exporter
        # Normalize once for efficient checks
        exact: List[str] = []
        prefixes: List[str] = []
        suffixes: List[str] = []
        for p in patterns:
            if not p:
                continue
            if p.endswith("*") and not p.startswith("*"):
                prefixes.append(p[:-1])
            elif p.startswith("*") and not p.endswith("*"):
                suffixes.append(p[1:])
            else:
                exact.append(p)
        self._exact = set(exact)
        self._prefixes = prefixes
        self._suffixes = suffixes

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to the exporter.

        Args:
            spans: List of spans to export.

        Returns:
            SpanExportResult.SUCCESS if the export was successful.
        """
        if is_trial_blocked():
            logger.debug("Trial/quota exhausted: blocking %d span(s) from export", len(spans))
            # Track trace IDs from spans being blocked during blocking period
            for span in spans:
                trace_id = self._get_trace_id(span)
                if trace_id:
                    add_blocked_trace_id(trace_id)
            return SpanExportResult.SUCCESS
        
        filtered: List[ReadableSpan] = []
        blocked_parent_map: Dict[Any, Any] = {}
        for span in spans:
            trace_id = self._get_trace_id(span)
            
            # Check if this span belongs to a trace ID that was blocked
            if trace_id and is_trace_id_blocked(trace_id):
                logger.debug("Filtering span from previously blocked trace ID: %s", trace_id)
                continue
            
            name = getattr(span, "name", None)
            if name is None:
                filtered.append(span)
                continue

            # Global blocking (configured patterns)
            globally_blocked = self._is_blocked(name)

            # Local per-span blocking via attribute set by LocalFilteringSpanProcessor
            locally_blocked = False
            try:
                local_patterns = self._get_local_patterns(span)
                if local_patterns:
                    locally_blocked = self._matches_any_pattern(name, local_patterns)
                # Fallback: if processor explicitly marked the span as locally blocked
                if not locally_blocked and self._has_local_block_flag(span):
                    locally_blocked = True
            except Exception:
                locally_blocked = False

            if not (globally_blocked or locally_blocked):
                filtered.append(span)
                continue

            # Collect mapping for reparenting children of the blocked span
            span_context = getattr(span, "context", None)
            span_id = getattr(span_context, "span_id", None) if span_context else None
            if span_id is not None:
                blocked_parent_map[span_id] = getattr(span, "parent", None)

        # Merge with registry of locally blocked spans captured by processor to handle
        # cases where children export before their blocked parent (SimpleSpanProcessor)
        merged_map: Dict[Any, Any] = {}
        try:
            if BLOCKED_LOCAL_PARENT_MAP:
                merged_map.update(BLOCKED_LOCAL_PARENT_MAP)
        except Exception:
            pass
        merged_map.update(blocked_parent_map)

        if merged_map:
            self._reparent_blocked_children(filtered, merged_map)
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._exporter.export(filtered)

    def _is_blocked(self, name: str) -> bool:
        """
        Check if a span name is blocked.

        Args:
            name: The span name to check.

        Returns:
            True if the span name is blocked, False otherwise.
        """
        if name in self._exact:
            return True
        for pref in self._prefixes:
            if name.startswith(pref):
                return True
        for suf in self._suffixes:
            if name.endswith(suf):
                return True
        return False

    def _get_trace_id(self, span: ReadableSpan) -> str:
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
                return format(trace_id, '032x')
            else:
                return str(trace_id)
        except Exception as e:
            logger.debug("Error extracting trace ID from span: %s", e)
            return ""

    def _get_local_patterns(self, span: ReadableSpan) -> List[str]:
        """
        Fetch local-block patterns from span attributes set by LocalFilteringSpanProcessor.

        Args:
            span: The span to fetch local-block patterns from.

        Returns:
            List of local-block patterns.
        """
        try:
            attrs = getattr(span, "attributes", None)
            if not attrs:
                return []
            value = None
            # Prefer Mapping.get if available
            try:
                if hasattr(attrs, "get"):
                    value = attrs.get("netra.local_blocked_spans")
                else:
                    value = attrs["netra.local_blocked_spans"]
            except Exception:
                value = None
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                return [v for v in value if v]
        except Exception:
            logger.debug("Failed reading local blocked patterns from span", exc_info=True)
        return []

    def _matches_any_pattern(self, name: str, patterns: Sequence[str]) -> bool:
        """
        Check if a span name matches any of the given patterns.

        Args:
            name: The span name to check.
            patterns: List of patterns to check against.

        Returns:
            True if the span name matches any of the given patterns, False otherwise.
        """
        for p in patterns:
            if not p:
                continue
            if p.endswith("*") and not p.startswith("*"):
                if name.startswith(p[:-1]):
                    return True
            elif p.startswith("*") and not p.endswith("*"):
                if name.endswith(p[1:]):
                    return True
            else:
                if name == p:
                    return True
        return False

    def _has_local_block_flag(self, span: ReadableSpan) -> bool:
        """
        Check if a span has a local-block flag.

        Args:
            span: The span to check.

        Returns:
            True if the span has a local-block flag, False otherwise.
        """
        try:
            attrs = getattr(span, "attributes", None)
            if not attrs:
                return False
            try:
                if hasattr(attrs, "get"):
                    value = attrs.get("netra.local_blocked")
                else:
                    value = attrs["netra.local_blocked"]
            except Exception:
                value = None
            return bool(value) is True
        except Exception:
            return False

    def _reparent_blocked_children(
        self,
        spans: Sequence[ReadableSpan],
        blocked_parent_map: Dict[Any, Any],
    ) -> None:
        """
        Reparent blocked children of a span.

        Args:
            spans: List of spans to reparent.
            blocked_parent_map: Dictionary mapping span IDs to their blocked parent spans.
        """
        if not blocked_parent_map:
            return

        for span in spans:
            parent_context = getattr(span, "parent", None)
            if parent_context is None:
                continue

            updated_parent = parent_context
            visited: set[Any] = set()
            changed = False

            while updated_parent is not None:
                parent_span_id = getattr(updated_parent, "span_id", None)
                if parent_span_id not in blocked_parent_map or parent_span_id in visited:
                    break
                visited.add(parent_span_id)
                updated_parent = blocked_parent_map[parent_span_id]
                changed = True

            if changed:
                self._set_span_parent(span, updated_parent)

    def _set_span_parent(self, span: ReadableSpan, parent: Any) -> None:
        """
        Set the parent of a span.

        Args:
            span: The span to set the parent of.
            parent: The parent to set.
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

    def shutdown(self) -> None:
        """
        Shutdown the exporter.
        """
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        """
        Force flush the exporter.

        Args:
            timeout_millis: The timeout in milliseconds.

        Returns:
            The result of the force flush operation.
        """
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            return True
