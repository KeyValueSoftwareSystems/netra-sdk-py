from typing import Any, Dict, List, Optional

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


def build_query_data_payload(
    scope: Scope,
    chart_type: ChartType,
    metrics: Metrics,
    filter: FilterConfig,
    dimension: Optional[Dimension] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for the query-data endpoint.

    Args:
        scope: The scope of data to query.
        chart_type: The type of chart visualization.
        metrics: Metrics configuration.
        filter: Filter configuration with time range and groupBy.
        dimension: Optional dimension configuration.

    Returns:
        Dictionary payload for the POST request.
    """
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
                    "field": item.field.value,
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

    return payload


def build_session_stats_payload(
    start_time: str,
    end_time: str,
    filters: Optional[List[SessionFilter]] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    sort_field: Optional[SortField] = None,
    sort_order: Optional[SortOrder] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for the session-stats endpoint.

    Args:
        start_time: Start time in ISO 8601 UTC format.
        end_time: End time in ISO 8601 UTC format.
        filters: Optional list of session filters.
        limit: Maximum number of results per page.
        cursor: Cursor for pagination.
        sort_field: Field to sort by.
        sort_order: Sort order (asc/desc).

    Returns:
        Dictionary payload for the POST request.
    """
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

    return payload


def build_session_summary_payload(
    start_time: str,
    end_time: str,
    filters: Optional[List[SessionFilter]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for the session-summary endpoint.

    Args:
        start_time: Start time in ISO 8601 UTC format.
        end_time: End time in ISO 8601 UTC format.
        filters: Optional list of session filters.

    Returns:
        Dictionary payload for the POST request.
    """
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

    return payload
