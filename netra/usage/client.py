import logging
from typing import Any, Dict, Optional

from netra.client import BaseNetraClient
from netra.config import Config

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.usage"


class UsageHttpClient(BaseNetraClient):
    """Internal HTTP client for usage APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the usage HTTP client.

        Args:
            config: Configuration object with usage settings.
        """
        super().__init__(
            config,
            log_prefix=_LOG_PREFIX,
            timeout_env_var="NETRA_USAGE_TIMEOUT",
        )

    def get_session_usage(
        self, session_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> Any:
        """
        Get session usage data.

        Args:
            session_id: Session identifier.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            Session usage data.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot fetch session usage '%s'", _LOG_PREFIX, session_id)
            return None

        try:
            url = f"/usage/sessions/{session_id}"
            params: Dict[str, str] = {}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            response = self._client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error(
                "%s: Failed to fetch session usage '%s': %s", _LOG_PREFIX, session_id, self._extract_error_message(exc)
            )
            return None

    def get_tenant_usage(self, tenant_id: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Any:
        """
        Get tenant usage data.

        Args:
            tenant_id: Tenant identifier.
            start_time: Optional start time filter.
            end_time: Optional end time filter.

        Returns:
            Tenant usage data.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot fetch tenant usage '%s'", _LOG_PREFIX, tenant_id)
            return None

        try:
            url = f"/usage/tenants/{tenant_id}"
            params: Dict[str, str] = {}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            response = self._client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error(
                "%s: Failed to fetch tenant usage '%s': %s", _LOG_PREFIX, tenant_id, self._extract_error_message(exc)
            )
            return None

    def list_traces(
        self,
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
    ) -> Any:
        """
        List all traces.

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
            Traces data.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot list traces", _LOG_PREFIX)
            return None

        try:
            url = "/sdk/traces"
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

            response = self._client.post(url, json=payload or None)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("%s: Failed to list traces: %s", _LOG_PREFIX, self._extract_error_message(exc))
            return None

    def list_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: Optional[str] = None,
        direction: Optional[str] = None,
        limit: Optional[int] = None,
        span_name: Optional[str] = None,
    ) -> Any:
        """
        List all spans for a given trace.

        Args:
            trace_id: Trace identifier.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            limit: Maximum number of spans to return.
            span_name: Search query for the spans.

        Returns:
            Spans data.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot list spans for trace '%s'", _LOG_PREFIX, trace_id)
            return None

        try:
            url = f"/sdk/traces/{trace_id}/spans"
            params: Dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            if direction is not None:
                params["direction"] = direction
            if limit is not None:
                params["limit"] = limit
            if span_name is not None:
                params["spanName"] = span_name

            response = self._client.get(url, params=params or None)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error(
                "%s: Failed to list spans for trace '%s': %s", _LOG_PREFIX, trace_id, self._extract_error_message(exc)
            )
            return None
