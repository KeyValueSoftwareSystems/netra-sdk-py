import logging
from typing import Any, Dict, List, Optional

from netra.client import BaseNetraClient
from netra.config import Config
from netra.dashboard.models import (
    ChartType,
    Dimension,
    DimensionField,
    FilterConfig,
    Metrics,
    Scope,
    SessionFilter,
    SortField,
    SortOrder,
)

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.dashboard"


class DashboardHttpClient(BaseNetraClient):
    """Internal HTTP client for Dashboard APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the dashboard HTTP client.

        Args:
            config: Configuration object with dashboard settings.
        """
        super().__init__(
            config,
            log_prefix=_LOG_PREFIX,
            timeout_env_var="NETRA_DASHBOARD_TIMEOUT",
            default_timeout=30.0,
            extra_headers={"Content-Type": "application/json"},
        )

    def query_data(
        self,
        scope: Scope,
        chart_type: ChartType,
        metrics: Metrics,
        filter: FilterConfig,
        dimension: Optional[Dimension] = None,
    ) -> Any:
        """
        Execute a dynamic query for dashboards.

        Args:
            scope: The scope of data to query (Scope.SPANS or Scope.TRACES).
            chart_type: The type of chart visualization.
            metrics: Metrics configuration with measure and aggregation.
            filter: Filter configuration with time range, groupBy, and optional filters.
            dimension: Optional dimension configuration for grouping results.

        Returns:
            The query response data or None on error.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot execute query", _LOG_PREFIX)
            return None

        try:
            url = "/public/dashboard/query-data"

            payload: Dict[str, Any] = {
                "scope": scope.value,
                "chartType": chart_type.value,
                "metrics": {
                    "measure": metrics.measure.value,
                    "aggregation": metrics.aggregation.value,
                },
            }

            if metrics.metric_name:
                payload["metrics"]["metricName"] = metrics.metric_name

            if filter:
                payload["filter"] = {
                    "startTime": filter.start_time,
                    "endTime": filter.end_time,
                    "groupBy": filter.group_by.value,
                }
                if filter.filters:
                    payload["filter"]["filters"] = [
                        {
                            "field": item.field.value if hasattr(item.field, "value") else item.field,
                            "operator": item.operator.value,
                            "type": item.type.value,
                            "value": item.value,
                            **({"key": item.key} if item.key else {}),
                        }
                        for item in filter.filters
                    ]

            if dimension:
                if dimension.field.value == DimensionField.CUSTOM.value:
                    payload["dimension"] = {"field": dimension.name}
                else:
                    payload["dimension"] = {"field": dimension.field.value}

            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(
                "%s: Failed to execute dashboard query: %s",
                _LOG_PREFIX,
                self._extract_error_message(exc),
            )
            return None

    def get_session_stats(
        self,
        start_time: str,
        end_time: str,
        filters: Optional[List[SessionFilter]],
        limit: Optional[int],
        cursor: Optional[str],
        sort_field: Optional[SortField],
        sort_order: Optional[SortOrder],
    ) -> Any:
        """
        Get session statistics with pagination.

        Args:
            start_time: Start time in ISO 8601 UTC format.
            end_time: End time in ISO 8601 UTC format.
            filters: Optional list of session filters.
            limit: Maximum number of results per page.
            cursor: Cursor for pagination.
            sort_field: Field to sort by.
            sort_order: Sort order (asc/desc).

        Returns:
            The session stats response data or None on error.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot fetch session stats", _LOG_PREFIX)
            return None

        try:
            url = "/public/dashboard/sessions/stats"

            payload: Dict[str, Any] = {
                "startTime": start_time,
                "endTime": end_time,
            }

            if filters:
                payload["filters"] = [
                    {
                        "field": filter_item.field.value,
                        "operator": filter_item.operator.value,
                        "type": filter_item.type.value,
                        "value": filter_item.value,
                    }
                    for filter_item in filters
                ]
            if limit or cursor:
                payload["pagination"] = {}
                if limit:
                    payload["pagination"]["limit"] = limit
                if cursor:
                    payload["pagination"]["cursor"] = cursor
            if sort_field:
                payload["sortField"] = sort_field.value
            if sort_order:
                payload["sortOrder"] = sort_order.value

            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(
                "%s: Failed to fetch session stats: %s",
                _LOG_PREFIX,
                self._extract_error_message(exc),
            )
            return None

    def get_session_summary(self, start_time: str, end_time: str, filters: Optional[List[SessionFilter]]) -> Any:
        """
        Get aggregated session metrics including total sessions, costs, latency, and cost breakdown by model.

        Args:
            start_time: Start time in ISO 8601 UTC format.
            end_time: End time in ISO 8601 UTC format.
            filters: Optional list of session filters.

        Returns:
            The session summary response data or None on error.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot fetch session summary", _LOG_PREFIX)
            return None

        try:
            url = "/public/dashboard/sessions/summary"
            payload: Dict[str, Any] = {
                "filter": {
                    "startTime": start_time,
                    "endTime": end_time,
                }
            }

            if filters:
                payload["filter"]["filters"] = [
                    {
                        "field": filter_item.field.value,
                        "operator": filter_item.operator.value,
                        "type": filter_item.type.value,
                        "value": filter_item.value,
                    }
                    for filter_item in filters
                ]

            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(
                "%s: Failed to fetch session summary: %s",
                _LOG_PREFIX,
                self._extract_error_message(exc),
            )
            return None
