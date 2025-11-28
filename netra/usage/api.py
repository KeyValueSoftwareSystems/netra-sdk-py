import logging
from typing import Any, Iterator, Literal, Optional

from netra.config import Config
from netra.usage.client import UsageHttpClient
from netra.usage.models import SessionUsageData, SpansPage, TenantUsageData, TracesPage, TraceSpan, TraceSummary

logger = logging.getLogger(__name__)


class Usage:
    """Public entry-point exposed as Netra.usage"""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the usage client.

        Args:
            cfg: Configuration object with usage settings
        """
        self._config = cfg
        self._client = UsageHttpClient(cfg)

    def get_session_usage(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
    ) -> SessionUsageData | Any:
        """
        Get session usage data.

        Args:
            session_id: Session identifier
            start_time: Start time for the usage data (in ISO 8601 UTC format)
            end_time: End time for the usage data (in ISO 8601 UTC format)

        Returns:
            SessionUsageData: Session usage data
        """
        if not session_id:
            logger.error("netra.usage: session_id is required to fetch session usage")
            return None
        if not start_time or not end_time:
            logger.error("netra.usage: start_time and end_time are required to fetch session usage")
            return None
        result = self._client.get_session_usage(session_id, start_time=start_time, end_time=end_time)
        session_id = result.get("session_id", "")
        if not session_id:
            return None
        token_count = result.get("tokenCount", 0)
        request_count = result.get("requestsCount", 0)
        total_cost = result.get("totalCost", 0.0)

        return SessionUsageData(
            session_id=session_id,
            token_count=token_count,
            request_count=request_count,
            total_cost=total_cost,
        )

    def get_tenant_usage(
        self,
        tenant_id: str,
        start_time: str,
        end_time: str,
    ) -> TenantUsageData | Any:
        """
        Get tenant usage data.

        Args:
            tenant_id: Tenant identifier
            start_time: Start time for the usage data (in ISO 8601 UTC format)
            end_time: End time for the usage data (in ISO 8601 UTC format)

        Returns:
            TenantUsageData: Tenant usage data
        """
        if not tenant_id:
            logger.error("netra.usage: tenant_id is required to fetch tenant usage")
            return None
        if not start_time or not end_time:
            logger.error("netra.usage: start_time and end_time are required to fetch tenant usage")
            return None
        result = self._client.get_tenant_usage(tenant_id, start_time=start_time, end_time=end_time)
        tenant_id = result.get("tenant_id", "")
        if not tenant_id:
            return None
        organisation_id = result.get("organisation_id")
        token_count = result.get("tokenCount", 0)
        request_count = result.get("requestsCount", 0)
        session_count = result.get("sessionsCount", 0)
        total_cost = result.get("totalCost", 0.0)
        return TenantUsageData(
            tenant_id=tenant_id,
            organisation_id=organisation_id,
            token_count=token_count,
            request_count=request_count,
            session_count=session_count,
            total_cost=total_cost,
        )

    def list_traces(
        self,
        start_time: str,
        end_time: str,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        sort_field: Optional[str] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> TracesPage | Any:
        """
        List all traces.

        Args:
            start_time: Start time for the traces (in ISO 8601 UTC format)
            end_time: End time for the traces (in ISO 8601 UTC format)
            search: Search with trace_id, session_id, user_id, or tenant_id for the traces
            limit: Maximum number of traces to return
            cursor: Cursor for pagination
            direction: Direction of pagination
            sort_field: Field to sort by
            sort_order: Order to sort by

        Returns:
            TracesPage: Traces page
        """
        if not start_time or not end_time:
            logger.error("netra.usage: start_time and end_time are required to list traces")
            return None

        result = self._client.list_traces(
            start_time=start_time,
            end_time=end_time,
            search=search,
            limit=limit,
            cursor=cursor,
            direction=direction,
            sort_field=sort_field,
            sort_order=sort_order,
        )

        if not isinstance(result, dict):
            return result

        data_block = result.get("data", {}) or {}
        items = data_block.get("data", []) or []
        page_info = data_block.get("pageInfo", {}) or {}
        has_next_page = bool(page_info.get("hasNextPage", False))
        next_cursor: Optional[str] = None
        if items:
            last_item = items[-1]
            if isinstance(last_item, dict):
                next_cursor = last_item.get("cursor")

        traces = [TraceSummary(**item) for item in items if isinstance(item, dict)]
        return TracesPage(traces=traces, has_next_page=has_next_page, next_cursor=next_cursor)

    def list_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        limit: Optional[int] = None,
        search: Optional[str] = None,
    ) -> SpansPage | Any:
        """
        List all spans for a given trace.

        Args:
            trace_id: Trace identifier
            cursor: Cursor for pagination
            direction: Direction of pagination
            limit: Maximum number of spans to return
            search: Search with span name or span kind name for the spans

        Returns:
            SpansPage: Spans page
        """
        if not trace_id:
            logger.error("netra.usage: trace_id is required to list spans")
            return None

        result = self._client.list_spans_by_trace_id(
            trace_id=trace_id,
            cursor=cursor,
            direction=direction,
            limit=limit,
            search=search,
        )

        if not isinstance(result, dict):
            return result

        data_block = result.get("data", {}) or {}
        items = data_block.get("data", []) or []
        page_info = data_block.get("pageInfo", {}) or {}
        has_next_page = bool(page_info.get("hasNextPage", False))
        next_cursor: Optional[str] = None
        if items:
            last_item = items[-1]
            if isinstance(last_item, dict):
                next_cursor = last_item.get("cursor")

        spans = [TraceSpan(**item) for item in items if isinstance(item, dict)]
        return SpansPage(spans=spans, has_next_page=has_next_page, next_cursor=next_cursor)

    def iter_traces(
        self,
        start_time: str,
        end_time: str,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        direction: Optional[Literal["up", "down"]] = "down",
        sort_field: Optional[str] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> Iterator[TraceSummary]:
        """
        Iterate over traces using cursor-based pagination.

        This is a thin convenience helper over list_traces that repeatedly
        fetches pages and yields individual TraceSummary items.

        Args:
            start_time: Start time for the traces (in ISO 8601 UTC format)
            end_time: End time for the traces (in ISO 8601 UTC format)
            search: Search with trace_id, session_id, user_id, or tenant_id for the traces
            limit: Maximum number of traces to return
            cursor: Cursor for pagination
            direction: Direction of pagination
            sort_field: Field to sort by
            sort_order: Order to sort by

        Returns:
            Iterator[TraceSummary]: Iterator over TraceSummary items
        """
        if not start_time or not end_time:
            logger.error("netra.usage: start_time and end_time are required to iterate traces")
            return

        current_cursor = cursor
        while True:
            page = self.list_traces(
                start_time=start_time,
                end_time=end_time,
                search=search,
                limit=limit,
                cursor=current_cursor,
                direction=direction,
                sort_field=sort_field,
                sort_order=sort_order,
            )

            if not isinstance(page, TracesPage):
                # If backend or configuration error returns a non-page payload,
                # stop iteration instead of raising.
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
        search: Optional[str] = None,
    ) -> Iterator[TraceSpan]:
        """
        Iterate over spans for a given trace using cursor-based pagination.

        This is a thin convenience helper over list_spans_by_trace_id that
        repeatedly fetches pages and yields individual TraceSpan items.

        Args:
            trace_id: Trace identifier
            cursor: Cursor for pagination
            direction: Direction of pagination
            limit: Maximum number of spans to return
            search: Search with span name or span kind name for the spans

        Returns:
            Iterator[TraceSpan]: Iterator over TraceSpan items
        """

        if not trace_id:
            logger.error("netra.usage: trace_id is required to iterate spans")
            return

        current_cursor = cursor
        while True:
            page = self.list_spans_by_trace_id(
                trace_id=trace_id,
                cursor=current_cursor,
                direction=direction,
                limit=limit,
                search=search,
            )

            if not isinstance(page, SpansPage):
                break

            for span in page.spans:
                yield span

            if not page.has_next_page or not page.next_cursor:
                break

            current_cursor = page.next_cursor
