import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from netra.usage.constants import LOG_PREFIX
from netra.usage.models import TraceSpan, TraceSummary

logger = logging.getLogger(__name__)


def build_session_usage_params(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, str]:
    """Build query parameters for the session-usage endpoint.

    Args:
        start_time: Optional start time in ISO 8601 UTC format.
        end_time: Optional end time in ISO 8601 UTC format.

    Returns:
        Dictionary of query parameters.
    """
    params: Dict[str, str] = {}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    return params


def build_tenant_usage_params(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, str]:
    """Build query parameters for the tenant-usage endpoint.

    Args:
        start_time: Optional start time in ISO 8601 UTC format.
        end_time: Optional end time in ISO 8601 UTC format.

    Returns:
        Dictionary of query parameters.
    """
    params: Dict[str, str] = {}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    return params


def build_list_traces_payload(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    direction: Optional[str] = None,
    sort_field: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for the list-traces endpoint.

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
        Dictionary payload for the POST request.
    """
    payload: Dict[str, Any] = {}
    if start_time is not None:
        payload["startTime"] = start_time
    if end_time is not None:
        payload["endTime"] = end_time

    filters = []
    filter_mapping = {
        "trace_id": trace_id,
        "session_id": session_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
    }

    for field, value in filter_mapping.items():
        if value is not None:
            filters.append({"field": field, "operator": "equals", "type": "string", "value": value})

    payload["filters"] = filters

    pagination: Dict[str, Any] = {}
    if limit is not None:
        pagination["limit"] = limit
    if cursor is not None:
        pagination["cursor"] = cursor
    if direction is not None:
        pagination["direction"] = direction
    if pagination:
        payload["pagination"] = pagination

    if sort_field is not None:
        payload["sortField"] = sort_field
    if sort_order is not None:
        payload["sortOrder"] = sort_order

    return payload


def build_list_spans_params(
    cursor: Optional[str] = None,
    direction: Optional[str] = None,
    limit: Optional[int] = None,
    span_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build query parameters for the list-spans-by-trace endpoint.

    Args:
        cursor: Cursor for pagination.
        direction: Direction of pagination.
        limit: Maximum number of spans to return.
        span_name: Search query for the spans.

    Returns:
        Dictionary of query parameters.
    """
    params: Dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if direction is not None:
        params["direction"] = direction
    if limit is not None:
        params["limit"] = limit
    if span_name is not None:
        params["spanName"] = span_name
    return params


def parse_trace_summaries(items: List[Dict[str, Any]]) -> List[TraceSummary]:
    """Safely parse raw dicts into TraceSummary objects, skipping invalid entries.

    Args:
        items: List of raw trace dicts from the API response.

    Returns:
        List of successfully parsed TraceSummary objects.
    """
    traces: List[TraceSummary] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            traces.append(TraceSummary(**item))
        except ValidationError as exc:
            logger.exception(
                "%s: Skipping malformed trace item: %s",
                LOG_PREFIX,
                exc,
            )
    return traces


def parse_trace_spans(items: List[Dict[str, Any]]) -> List[TraceSpan]:
    """Safely parse raw dicts into TraceSpan objects, skipping invalid entries.

    Args:
        items: List of raw span dicts from the API response.

    Returns:
        List of successfully parsed TraceSpan objects.
    """
    spans: List[TraceSpan] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            spans.append(TraceSpan(**item))
        except ValidationError as exc:
            logger.exception(
                "%s: Skipping malformed span item: %s",
                LOG_PREFIX,
                exc,
            )
    return spans
