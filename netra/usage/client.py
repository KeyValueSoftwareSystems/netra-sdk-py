import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)


class _UsageHttpClient:
    """Internal HTTP client for usage APIs."""

    def __init__(self, config: Config) -> None:
        """Initialize the usage HTTP client."""
        self._client: Optional[httpx.Client] = None
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.usage: NETRA_OTLP_ENDPOINT is required for usage APIs")
            return

        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]

        headers: Dict[str, str] = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key

        timeout_env = os.getenv("NETRA_USAGE_TIMEOUT")
        try:
            timeout = float(timeout_env) if timeout_env else 10.0
        except ValueError:
            logger.warning(
                "netra.usage: Invalid NETRA_USAGE_TIMEOUT value '%s', using default 10.0",
                timeout_env,
            )
            timeout = 10.0

        try:
            self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.usage: Failed to initialize usage HTTP client: %s", exc)
            self._client = None

    def get_session_usage(self, session_id: str) -> Any:
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
            url = f"/usage/tokens/sessions/{session_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error("netra.usage: Failed to fetch session usage '%s': %s", session_id, exc)
            return {}

    def get_tenant_usage(self, tenant_id: str) -> Any:
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
            url = f"/usage/tokens/tenants/{tenant_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error("netra.usage: Failed to fetch tenant usage '%s': %s", tenant_id, exc)
            return {}
