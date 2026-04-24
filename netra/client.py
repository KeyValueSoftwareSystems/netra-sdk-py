"""Base HTTP client shared by all Netra API sub-clients."""

import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)

_TELEMETRY_SUFFIX = "/telemetry"
_API_KEY_HEADER = "x-api-key"


class BaseNetraClient:
    """Shared foundation for every Netra HTTP sub-client.

    Provides endpoint resolution, header construction, timeout parsing,
    httpx client creation, and safe error-message extraction so that
    sub-clients only need to define domain-specific endpoints.

    Args:
        config: Netra SDK configuration.
        log_prefix: Short prefix used in all log messages (e.g. ``"netra.dashboard"``).
        timeout_env_var: Name of the environment variable that overrides the
            default timeout (e.g. ``"NETRA_DASHBOARD_TIMEOUT"``).
        default_timeout: Fallback timeout in seconds when the env var is unset.
        extra_headers: Additional headers merged on top of the standard set.
    """

    def __init__(
        self,
        config: Config,
        *,
        log_prefix: str,
        timeout_env_var: str,
        default_timeout: float = 10.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._log_prefix = log_prefix
        self._timeout_env_var = timeout_env_var
        self._default_timeout = default_timeout
        self._extra_headers = extra_headers or {}
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """Build an ``httpx.Client`` from the shared configuration.

        Args:
            config: Netra SDK configuration.

        Returns:
            A configured client, or ``None`` if the endpoint is missing or
            client creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("%s: NETRA_OTLP_ENDPOINT is required", self._log_prefix)
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("%s: Failed to create HTTP client: %s", self._log_prefix, exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Strip trailing slash and ``/telemetry`` suffix from an endpoint URL.

        Args:
            endpoint: The raw endpoint URL.

        Returns:
            The cleaned base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith(_TELEMETRY_SUFFIX):
            base_url = base_url[: -len(_TELEMETRY_SUFFIX)]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """Construct request headers from configuration.

        Args:
            config: Netra SDK configuration.

        Returns:
            A dictionary of HTTP headers.
        """
        headers: Dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers[_API_KEY_HEADER] = config.api_key
        headers.update(self._extra_headers)
        return headers

    def _get_timeout(self) -> float:
        """Read timeout from the environment or fall back to the default.

        Returns:
            Timeout value in seconds.
        """
        raw = os.getenv(self._timeout_env_var)
        if not raw:
            return self._default_timeout
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "%s: Invalid %s value '%s', using default %.1f",
                self._log_prefix,
                self._timeout_env_var,
                raw,
                self._default_timeout,
            )
            return self._default_timeout

    def close(self) -> None:
        """Close the underlying HTTP client and release connection-pool resources.

        Safe to call multiple times or when the client was never created.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("%s: Error closing HTTP client", self._log_prefix, exc_info=True)
            finally:
                self._client = None

    def __enter__(self) -> "BaseNetraClient":
        """Support ``with`` blocks for short-lived client usage."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the client when exiting a ``with`` block."""
        self.close()

    def _extract_error_message(self, exc: Exception) -> str:
        """Derive a human-readable error string from an exception.

        For HTTP status errors whose ``response`` attribute carries a body,
        this tries to extract the backend JSON error payload.  Falls back to
        ``str(exc)`` in all other cases.

        Args:
            exc: The exception that was raised.

        Returns:
            A descriptive error message.
        """
        response: Any = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.json()
                error_data = body.get("error", {})
                if isinstance(error_data, dict):
                    msg = error_data.get("message")
                    if msg:
                        return str(msg)
            except Exception:
                pass
        return str(exc)
