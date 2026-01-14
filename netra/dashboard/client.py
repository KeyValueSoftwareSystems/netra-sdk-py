import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config
from netra.dashboard.models import (
    ChartType,
    Dimension,
    FilterConfig,
    Metrics,
    Scope,
)

logger = logging.getLogger(__name__)


class DashboardHttpClient:
    """Internal HTTP client for Dashboard APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the dashboard HTTP client.

        Args:
            config: Configuration object with dashboard settings
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """
        Create an HTTP client for dashboard endpoints.

        Args:
            config: The configuration object.

        Returns:
            An HTTP client for dashboard endpoints, or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.dashboard: NETRA_OTLP_ENDPOINT is required for dashboard APIs")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.dashboard: Failed to initialize dashboard HTTP client: %s", exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """
        Resolve base URL from endpoint.

        Args:
            endpoint: The endpoint to resolve.

        Returns:
            The resolved base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """
        Build Headers for Dashboard Client.

        Args:
            config: The configuration object.

        Returns:
            The headers for dashboard client.
        """
        headers: Dict[str, str] = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def _get_timeout(self) -> float:
        """
        Get timeout for dashboard client.

        Returns:
            The timeout for dashboard client.
        """
        timeout_env = os.getenv("NETRA_DASHBOARD_TIMEOUT")
        if not timeout_env:
            return 30.0
        try:
            return float(timeout_env)
        except ValueError:
            logger.warning(
                "netra.dashboard: Invalid NETRA_DASHBOARD_TIMEOUT value '%s', using default 30.0",
                timeout_env,
            )
            return 30.0

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
            logger.error("netra.dashboard: Dashboard client is not initialized; cannot execute query")
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
                payload["dimension"] = {"field": dimension.field.value}

            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.dashboard: Failed to execute dashboard query: %s",
                response_json.get("error").get("message", ""),
            )
            return None
