import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)


class UsageHttpClient:
    """Internal HTTP client for usage APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the usage HTTP client.

        Args:
            config: Configuration object with usage settings
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.usage: NETRA_OTLP_ENDPOINT is required for usage APIs")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.usage: Failed to initialize usage HTTP client: %s", exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        headers: Dict[str, str] = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _get_timeout(self) -> float:
        timeout_env = os.getenv("NETRA_USAGE_TIMEOUT")
        if not timeout_env:
            return 10.0
        try:
            return float(timeout_env)
        except ValueError:
            logger.warning(
                "netra.usage: Invalid NETRA_USAGE_TIMEOUT value '%s', using default 10.0",
                timeout_env,
            )
            return 10.0

    def get_session_usage(self, session_id: str, start_time: str | None = None, end_time: str | None = None) -> Any:
        """
        Get session usage data.

        Args:
            session_id: Session identifier

        Returns:
            Any: Session usage data
        """
        if not self._client:
            logger.error(
                "netra.usage: Usage client is not initialized; cannot fetch session usage '%s'",
                session_id,
            )
            return {}

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
            logger.error("netra.usage: Failed to fetch session usage '%s': %s", session_id, exc)
            return {}

    def get_tenant_usage(self, tenant_id: str, start_time: str | None = None, end_time: str | None = None) -> Any:
        """
        Get tenant usage data.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Any: Tenant usage data
        """
        if not self._client:
            logger.error(
                "netra.usage: Usage client is not initialized; cannot fetch tenant usage '%s'",
                tenant_id,
            )
            return {}

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
            logger.error("netra.usage: Failed to fetch tenant usage '%s': %s", tenant_id, exc)
            return {}

    def list_traces(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        search: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        direction: str | None = None,
        sort_field: str | None = None,
        sort_order: str | None = None,
    ) -> Any:
        if not self._client:
            logger.error("netra.usage: Usage client is not initialized; cannot list traces")
            return {}

        try:
            url = "/sdk/traces"
            payload: Dict[str, Any] = {}
            if start_time is not None:
                payload["startTime"] = start_time
            if end_time is not None:
                payload["endTime"] = end_time
            if search is not None:
                payload["search"] = search

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
            data = response.json()
            return data
        except Exception as exc:
            logger.error("netra.usage: Failed to list traces: %s", exc)
            return {}

    def list_spans_by_trace_id(
        self,
        trace_id: str,
        cursor: str | None = None,
        direction: str | None = None,
        limit: int | None = None,
        search: str | None = None,
    ) -> Any:
        if not self._client:
            logger.error("netra.usage: Usage client is not initialized; cannot list spans for trace '%s'", trace_id)
            return {}

        try:
            url = f"/sdk/traces/{trace_id}/spans"
            params: Dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            if direction is not None:
                params["direction"] = direction
            if limit is not None:
                params["limit"] = limit
            if search is not None:
                params["search"] = search

            response = self._client.get(url, params=params or None)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as exc:
            logger.error("netra.usage: Failed to list spans for trace '%s': %s", trace_id, exc)
            return {}
