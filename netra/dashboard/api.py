import logging
from typing import Any, Iterator, List, Literal, Optional

from netra.config import Config
from netra.dashboard.client import DashboardHttpClient
from netra.dashboard.models import (
    ChartType,
    Dimension,
    FilterConfig,
    Metrics,
    Scope,
    SessionFilter,
    SessionFilterConfig,
    SessionStatsData,
    SessionStatsResult,
    SortField,
)

logger = logging.getLogger(__name__)


class Dashboard:
    """Public entry-point exposed as Netra.dashboard"""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the dashboard client.

        Args:
            cfg: Configuration object with dashboard settings
        """
        self._config = cfg
        self._client = DashboardHttpClient(cfg)

    def query_data(
        self,
        scope: Scope,
        chart_type: ChartType,
        metrics: Metrics,
        filter: FilterConfig,
        dimension: Optional[Dimension] = None,
    ) -> Any:
        """
        Execute a dynamic query for dashboards to retrieve metrics and time-series data.

        Args:
            scope: The scope of data to query (DashboardScope.SPANS or DashboardScope.TRACES).
            chart_type: The type of chart visualization.
            metrics: Metrics configuration with measure and aggregation.
            filter: Filter configuration with time range, groupBy, and optional filters.
            dimension: Optional dimension configuration for grouping results.

        Returns:
            Dict containing timeRange and data, or None on error.
        """

        if not isinstance(scope, Scope):
            raise TypeError(f"scope must be a Scope, got {type(scope).__name__}")
        if not isinstance(chart_type, ChartType):
            raise TypeError(f"chart_type must be a ChartType, got {type(chart_type).__name__}")
        if not isinstance(metrics, Metrics):
            raise TypeError(f"metrics must be a Metrics, got {type(metrics).__name__}")
        if not isinstance(filter, FilterConfig):
            raise TypeError(f"filter must be a FilterConfig, got {type(filter).__name__}")
        if dimension is not None and not isinstance(dimension, Dimension):
            raise TypeError(f"dimension must be a Dimension or None, got {type(dimension).__name__}")

        result = self._client.query_data(
            scope=scope,
            chart_type=chart_type,
            metrics=metrics,
            filter=filter,
            dimension=dimension,
        )
        return result

    def get_session_stats(
        self,
        start_time: str,
        end_time: str,
        filters: Optional[List[SessionFilter]] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        search: Optional[str] = None,
        sort_field: Optional[SortField] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> SessionStatsResult | Any:
        """
        Get session statistics with pagination.

        Args:
            start_time: Start of the time window (ISO 8601 UTC timestamp).
            end_time: End of the time window (ISO 8601 UTC timestamp).
            filters: Optional list of SessionFilter conditions.
            limit: Maximum number of results to return in this page.
            page: Page number to get the respective paginated result.
            search: Keyword to filter out the results.
            sort_field: Field to sort results by.
            sort_order: Sort order (asc/desc).

        Returns:
            SessionStatsResult containing session data and pagination info.
        """
        if not start_time or not end_time:
            logger.error("netra.dashboard: start_time and end_time are required to fetch session stats")
            return None

        result = self._client.get_session_stats(
            start_time=start_time,
            end_time=end_time,
            filters=filters,
            limit=limit,
            page=page,
            search=search,
            sort_field=sort_field,
            sort_order=sort_order,
        )

        if not isinstance(result, dict):
            return result

        sessions = result.get("sessions", []) or []
        cursor = result.get("cursor", {}) or {}
        has_next_page = bool(cursor.get("hasMore", False))
        next_page: Optional[int] = None
        if has_next_page:
            current_page = cursor.get("pageNo", 1)
            next_page = current_page + 1

        data = [
            SessionStatsData(
                session_id=item.get("session_id", ""),
                start_time=item.get("start_time", ""),
                total_requests=item.get("totalRequests", 0),
                total_cost=item.get("totalCost", 0.0),
                session_duration=item.get("session_duration", ""),
            )
            for item in sessions
            if isinstance(item, dict)
        ]

        return SessionStatsResult(data=data, has_next_page=has_next_page, next_page=next_page)

    def iter_session_stats(
        self,
        start_time: str,
        end_time: str,
        filters: Optional[List[SessionFilter]] = None,
        limit: Optional[int] = None,
        search: Optional[str] = None,
        sort_field: Optional[SortField] = None,
        sort_order: Optional[Literal["asc", "desc"]] = None,
    ) -> Iterator[SessionStatsData]:
        """
        Iterate over session statistics using page-based pagination.

        This is a convenience helper over get_session_stats that repeatedly
        fetches pages and yields individual SessionStatsData items.

        Args:
            start_time: Start of the time window (ISO 8601 UTC timestamp).
            end_time: End of the time window (ISO 8601 UTC timestamp).
            filters: Optional list of SessionFilter conditions.
            limit: Maximum number of results to return per page.
            search: Keyword to filter out the results.
            sort_field: Field to sort results by.
            sort_order: Sort order (asc/desc).

        Yields:
            SessionStatsData items from all pages.
        """
        if not start_time or not end_time:
            logger.error("netra.dashboard: start_time and end_time are required to iterate session stats")
            return

        current_page: Optional[int] = None
        while True:
            result = self.get_session_stats(
                start_time=start_time,
                end_time=end_time,
                filters=filters,
                limit=limit,
                page=current_page,
                search=search,
                sort_field=sort_field,
                sort_order=sort_order,
            )

            if not isinstance(result, SessionStatsResult):
                break

            for session in result.data:
                yield session

            if not result.has_next_page or not result.next_page:
                break

            current_page = result.next_page

    def get_session_summary(self, filter: SessionFilterConfig) -> Any:

        if not filter.start_time:
            raise SyntaxError(f"start_time value was not provided")
        if not filter.end_time:
            raise SyntaxError(f"end_time value was not provided")
        if not isinstance(filter, SessionFilterConfig):
            raise TypeError(f"filter must be a SessionFilterConfig, got {type(filter).__name__}")
        if filter.filters is not None and not isinstance(filter.filters, list):
            raise TypeError("filters must be a list of SessionFilter")

        result = self._client.get_session_summary(
            start_time=filter.start_time, end_time=filter.end_time, filters=filter.filters
        )

        return result
