import logging
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from netra.client import BaseHttpClient
from netra.usage.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    URL_SESSION_USAGE,
    URL_SPANS_BY_TRACE,
    URL_TENANT_USAGE,
    URL_TRACES,
)
from netra.usage.utils import (
    build_list_spans_params,
    build_list_traces_payload,
    build_session_usage_params,
    build_tenant_usage_params,
)

logger = logging.getLogger(__name__)


class UsageHttpClient(BaseHttpClient):
    """Internal HTTP client for usage APIs."""

    __slots__ = ()

    _LOG_PREFIX = LOG_PREFIX
    _ENV_TIMEOUT = ENV_TIMEOUT
    _DEFAULT_TIMEOUT = DEFAULT_TIMEOUT

    def get_session_usage(
        self,
        session_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get session usage data.

        Args:
            session_id: Session identifier.
            start_time: Optional start time in ISO 8601 UTC format.
            end_time: Optional end time in ISO 8601 UTC format.

        Returns:
            Session usage data dict, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_SESSION_USAGE.format(session_id=quote(session_id, safe=""))
            params = build_session_usage_params(start_time=start_time, end_time=end_time)
            response = client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data")
            if isinstance(data, dict):
                return data
            logger.error("%s: Unexpected response type from session usage endpoint", LOG_PREFIX)
            return None
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to fetch session usage '%s': %s", LOG_PREFIX, session_id, error_msg)
            return None

    def get_tenant_usage(
        self,
        tenant_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get tenant usage data.

        Args:
            tenant_id: Tenant identifier.
            start_time: Optional start time in ISO 8601 UTC format.
            end_time: Optional end time in ISO 8601 UTC format.

        Returns:
            Tenant usage data dict, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_TENANT_USAGE.format(tenant_id=quote(tenant_id, safe=""))
            params = build_tenant_usage_params(start_time=start_time, end_time=end_time)
            response = client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data")
            if isinstance(data, dict):
                return data
            logger.error("%s: Unexpected response type from tenant usage endpoint", LOG_PREFIX)
            return None
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to fetch tenant usage '%s': %s", LOG_PREFIX, tenant_id, error_msg)
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
    ) -> Optional[Dict[str, Any]]:
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
            Traces response data dict, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            payload = build_list_traces_payload(
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

            response = client.post(URL_TRACES, json=payload or None)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.error("%s: Unexpected response type from list traces endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to list traces: %s", LOG_PREFIX, error_msg)
            return None

    def list_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: Optional[str] = None,
        direction: Optional[str] = None,
        limit: Optional[int] = None,
        span_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """List all spans for a given trace.

        Args:
            trace_id: Trace identifier.
            cursor: Cursor for pagination.
            direction: Direction of pagination.
            limit: Maximum number of spans to return.
            span_name: Search query for the spans.

        Returns:
            Spans response data dict, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_SPANS_BY_TRACE.format(trace_id=quote(trace_id, safe=""))
            params = build_list_spans_params(
                cursor=cursor,
                direction=direction,
                limit=limit,
                span_name=span_name,
            )

            response = client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.error("%s: Unexpected response type from list spans endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to list spans for trace '%s': %s", LOG_PREFIX, trace_id, error_msg)
            return None
