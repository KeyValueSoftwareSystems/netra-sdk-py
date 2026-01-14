from typing import Any, Optional

from netra.config import Config
from netra.dashboard.client import DashboardHttpClient
from netra.dashboard.models import (
    ChartType,
    DashboardDimension,
    DashboardFilterConfig,
    DashboardMetrics,
    DashboardScope,
)


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
        scope: DashboardScope,
        chart_type: ChartType,
        metrics: DashboardMetrics,
        filter: DashboardFilterConfig,
        dimension: Optional[DashboardDimension] = None,
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

        if not isinstance(scope, DashboardScope):
            raise TypeError(f"scope must be a DashboardScope, got {type(scope).__name__}")
        if not isinstance(chart_type, ChartType):
            raise TypeError(f"chart_type must be a ChartType, got {type(chart_type).__name__}")
        if not isinstance(metrics, DashboardMetrics):
            raise TypeError(f"metrics must be a DashboardMetrics, got {type(metrics).__name__}")
        if not isinstance(filter, DashboardFilterConfig):
            raise TypeError(f"filter must be a DashboardFilterConfig, got {type(filter).__name__}")
        if dimension is not None and not isinstance(dimension, DashboardDimension):
            raise TypeError(f"dimension must be a DashboardDimension or None, got {type(dimension).__name__}")

        result = self._client.query_data(
            scope=scope,
            chart_type=chart_type,
            metrics=metrics,
            filter=filter,
            dimension=dimension,
        )
        return result
