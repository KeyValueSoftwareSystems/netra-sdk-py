import logging
from typing import Iterator, Literal, Optional

from netra.client import parse_paginated_response
from netra.config import Config
from netra.usage.client import UsageHttpClient
from netra.usage.constants import LOG_PREFIX
from netra.usage.models import SessionUsageData, SpansPage, TenantUsageData, TracesPage, TraceSpan, TraceSummary
from netra.usage.utils import parse_trace_spans, parse_trace_summaries

logger = logging.getLogger(__name__)


class Usage:
    """Public entry-point exposed as Netra.usage."""

    __slots__ = ("_config", "_client")

    def __init__(self, cfg: Config) -> None:
        """Initialize the usage client.

        Args:
            cfg: Configuration object with usage settings.
        """
        self._config = cfg
        self._client = UsageHttpClient(cfg)

    def close(self) -> None:
        """Release resources held by the usage client."""
        self._client.close()

    def get_session_usage(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
    ) -> Optional[SessionUsageData]:
        """Get session usage data.

        Args:
            session_id: Session identifier.
            start_time: Start time for the usage data (in ISO 8601 UTC format).
            end_time: End time for the usage data (in ISO 8601 UTC format).

        Returns:
            SessionUsageData on success, or None on failure.
        """
        if not session_id:
            logger.error("%s: session_id is required to fetch session usage", LOG_PREFIX)
            return None
        if not start_time or not end_time:
            logger.error("%s: start_time and end_time are required to fetch session usage", LOG_PREFIX)
            return None

        result = self._client.get_session_usage(session_id, start_time=start_time, end_time=end_time)
        if result is None:
            return None

        session_id_value = result.get("session_id", "")
        if not session_id_value:
            return None

        return SessionUsageData(
            session_id=session_id_value,
            token_count=result.get("tokenCount", 0),
            request_count=result.get("requestsCount", 0),
            total_cost=result.get("totalCost", 0.0),
        )

    def get_tenant_usage(
        self,
        tenant_id: str,
        start_time: str,
        end_time: str,
    ) -> Optional[TenantUsageData]:
        """Get tenant usage data.

        Args:
            tenant_id: Tenant identifier.
            start_time: Start time for the usage data (in ISO 8601 UTC format).
            end_time: End time for the usage data (in ISO 8601 UTC format).

        Returns:
            TenantUsageData on success, or None on failure.
        """
        if not tenant_id:
            logger.error("%s: tenant_id is required to fetch tenant usage", LOG_PREFIX)
            return None
        if not start_time or not end_time:
            logger.error("%s: start_time and end_time are required to fetch tenant usage", LOG_PREFIX)
            return None

        result = self._client.get_tenant_usage(tenant_id, start_time=start_time, end_time=end_time)
        if result is None:
            return None

        tenant_id_value = result.get("tenant_id", "")
        if not tenant_id_value:
            return None

        return TenantUsageData(
            tenant_id=tenant_id_value,
            organisation_id=result.get("organisation_id"),
            token_count=result.get("tokenCount", 0),
            request_count=result.get("requestsCount", 0),
            session_count=result.get("sessionsCount", 0),
            total_cost=result.get("totalCost", 0.0),
        )

    def list_traces(
        self,
        start_time: str,
        end_time: str,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        sort_field: Optional[str] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> Optional[TracesPage]:
        """List all traces.

        Args:
            start_time: Start time for the traces (in ISO 8601 UTC format).
            end_time: End time for the traces (in ISO 8601 UTC format).
            trace_id: Search based on trace_id, if provided.
            session_id: Search based on session_id, if provided.
            user_id: Search based on user_id, if provided.
            tenant_id: Search based on tenant_id, if provided.
            limit: Maximum number of traces to return.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            sort_field: Field to sort by.
            sort_order: Order to sort by.

        Returns:
            TracesPage on success, or None on failure.
        """
        if not start_time or not end_time:
            logger.error("%s: start_time and end_time are required to list traces", LOG_PREFIX)
            return None

        result = self._client.list_traces(
            start_time=start_time,
            end_time=end_time,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
            cursor=cursor,
            direction=direction,
            sort_field=sort_field,
            sort_order=sort_order,
        )

        if result is None:
            return None

        items, has_next_page, next_cursor = parse_paginated_response(result, items_key="data")
        traces = parse_trace_summaries(items)
        return TracesPage(traces=traces, has_next_page=has_next_page, next_cursor=next_cursor)

    def list_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        limit: Optional[int] = None,
        span_name: Optional[str] = None,
    ) -> Optional[SpansPage]:
        """List all spans for a given trace.

        Args:
            trace_id: Trace identifier.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            limit: Maximum number of spans to return.
            span_name: Search with span name or span kind name for the spans.

        Returns:
            SpansPage on success, or None on failure.
        """
        if not trace_id:
            logger.error("%s: trace_id is required to list spans", LOG_PREFIX)
            return None

        result = self._client.list_spans_by_trace_id(
            trace_id=trace_id,
            cursor=cursor,
            direction=direction,
            limit=limit,
            span_name=span_name,
        )

        if result is None:
            return None

        items, has_next_page, next_cursor = parse_paginated_response(result, items_key="data")
        spans = parse_trace_spans(items)
        return SpansPage(spans=spans, has_next_page=has_next_page, next_cursor=next_cursor)

    def iter_traces(
        self,
        start_time: str,
        end_time: str,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        sort_field: Optional[str] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> Iterator[TraceSummary]:
        """Iterate over traces using cursor-based pagination.

        This is a convenience helper over list_traces that repeatedly
        fetches pages and yields individual TraceSummary items.

        Args:
            start_time: Start time for the traces (in ISO 8601 UTC format).
            end_time: End time for the traces (in ISO 8601 UTC format).
            trace_id: Search based on trace_id, if provided.
            session_id: Search based on session_id, if provided.
            user_id: Search based on user_id, if provided.
            tenant_id: Search based on tenant_id, if provided.
            limit: Maximum number of traces to return.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            sort_field: Field to sort by.
            sort_order: Order to sort by.

        Yields:
            TraceSummary items from all pages.
        """
        if not start_time or not end_time:
            logger.error("%s: start_time and end_time are required to iterate traces", LOG_PREFIX)
            return

        current_cursor = cursor
        while True:
            page = self.list_traces(
                start_time=start_time,
                end_time=end_time,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                limit=limit,
                cursor=current_cursor,
                direction=direction,
                sort_field=sort_field,
                sort_order=sort_order,
            )

            if page is None:
                break

            for trace in page.traces:
                yield trace

            if not page.has_next_page or not page.next_cursor:
                break

            current_cursor = page.next_cursor

    def iter_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        limit: Optional[int] = None,
        span_name: Optional[str] = None,
    ) -> Iterator[TraceSpan]:
        """Iterate over spans for a given trace using cursor-based pagination.

        This is a convenience helper over list_spans_by_trace_id that
        repeatedly fetches pages and yields individual TraceSpan items.

        Args:
            trace_id: Trace identifier.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            limit: Maximum number of spans to return.
            span_name: Search with span name or span kind name for the spans.

        Yields:
            TraceSpan items from all pages.
        """
        if not trace_id:
            logger.error("%s: trace_id is required to iterate spans", LOG_PREFIX)
            return

        current_cursor = cursor
        while True:
            page = self.list_spans_by_trace_id(
                trace_id=trace_id,
                cursor=current_cursor,
                direction=direction,
                limit=limit,
                span_name=span_name,
            )

            if page is None:
                break

            for span in page.spans:
                yield span

            if not page.has_next_page or not page.next_cursor:
                break

            current_cursor = page.next_cursor
