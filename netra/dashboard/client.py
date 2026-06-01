import logging
from typing import Any, Dict, List, Optional

import httpx

from netra.client import BaseHttpClient
from netra.dashboard.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    URL_QUERY_DATA,
    URL_SESSION_STATS,
    URL_SESSION_SUMMARY,
)
from netra.dashboard.models import (
    ChartType,
    Dimension,
    FilterConfig,
    Metrics,
    Scope,
    SessionFilter,
    SortField,
    SortOrder,
)
from netra.dashboard.utils import (
    build_query_data_payload,
    build_session_stats_payload,
    build_session_summary_payload,
)

logger = logging.getLogger(__name__)


class DashboardHttpClient(BaseHttpClient):
    """Internal HTTP client for Dashboard APIs."""

    __slots__ = ()

    _LOG_PREFIX = LOG_PREFIX
    _ENV_TIMEOUT = ENV_TIMEOUT
    _DEFAULT_TIMEOUT = DEFAULT_TIMEOUT

    def query_data(
        self,
        scope: Scope,
        chart_type: ChartType,
        metrics: Metrics,
        filter: FilterConfig,
        dimension: Optional[Dimension] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a dynamic query for dashboards.

        Args:
            scope: The scope of data to query (Scope.SPANS or Scope.TRACES).
            chart_type: The type of chart visualization.
            metrics: Metrics configuration with measure and aggregation.
            filter: Filter configuration with time range, groupBy, and optional filters.
            dimension: Optional dimension configuration for grouping results.

        Returns:
            The query response data or None on error.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            payload = build_query_data_payload(
                scope=scope,
                chart_type=chart_type,
                metrics=metrics,
                filter=filter,
                dimension=dimension,
            )
            response = client.post(URL_QUERY_DATA, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.error("%s: Unexpected response type from query-data endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to execute dashboard query: %s", LOG_PREFIX, error_msg)
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
    ) -> Optional[Dict[str, Any]]:
        """Get session statistics with pagination.

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
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            payload = build_session_stats_payload(
                start_time=start_time,
                end_time=end_time,
                filters=filters,
                limit=limit,
                cursor=cursor,
                sort_field=sort_field,
                sort_order=sort_order,
            )

            response = client.post(URL_SESSION_STATS, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.warning("%s: Unexpected response type from session stats endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to fetch session stats: %s", LOG_PREFIX, error_msg)
            return None

    def get_session_summary(
        self,
        start_time: str,
        end_time: str,
        filters: Optional[List[SessionFilter]],
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated session metrics.

        Args:
            start_time: Start time in ISO 8601 UTC format.
            end_time: End time in ISO 8601 UTC format.
            filters: Optional list of session filters.

        Returns:
            The session summary response data or None on error.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            payload = build_session_summary_payload(
                start_time=start_time,
                end_time=end_time,
                filters=filters,
            )

            response = client.post(URL_SESSION_SUMMARY, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.warning("%s: Unexpected response type from session summary endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to fetch session summary: %s", LOG_PREFIX, error_msg)
            return None
