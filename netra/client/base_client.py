import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)

TELEMETRY_SUFFIX = "/telemetry"


class BaseHttpClient:
    """Base HTTP client providing common setup for all Netra API clients.

    Subclasses must define:
        _LOG_PREFIX: str — module-specific log prefix (e.g. "netra.dashboard").
        _ENV_TIMEOUT: str — env var name for timeout override.
        _DEFAULT_TIMEOUT: float — default timeout in seconds.

    Attributes:
        _client: The underlying httpx client instance.
    """

    __slots__ = ("_client",)

    _LOG_PREFIX: str = "netra"
    _ENV_TIMEOUT: str = ""
    _DEFAULT_TIMEOUT: float = 10.0

    def __init__(self, config: Config) -> None:
        """Initialize the HTTP client.

        Args:
            config: The Netra configuration object.
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def close(self) -> None:
        """Close the underlying HTTP client and release connection resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                logger.exception("%s: Error closing HTTP client", self._LOG_PREFIX)
            finally:
                self._client = None

    def _ensure_client(self) -> Optional[httpx.Client]:
        """Return the underlying client, logging an error if it is not initialized.

        Returns:
            The httpx client, or None if not available.
        """
        if not self._client:
            logger.error("%s: Client not initialized", self._LOG_PREFIX)
        return self._client

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """Create and configure the HTTP client.

        Args:
            config: The Netra configuration object.

        Returns:
            Configured httpx client or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("%s: NETRA_OTLP_ENDPOINT is required", self._LOG_PREFIX)
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._parse_env_float(self._ENV_TIMEOUT, self._DEFAULT_TIMEOUT)

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception:
            logger.exception("%s: Failed to create HTTP client", self._LOG_PREFIX)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Extract base URL, removing telemetry suffix if present.

        Args:
            endpoint: The raw endpoint URL.

        Returns:
            The cleaned base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith(TELEMETRY_SUFFIX):
            base_url = base_url[: -len(TELEMETRY_SUFFIX)]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """Build request headers from configuration.

        Args:
            config: The Netra configuration object.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: Dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    def _extract_error_message(
        self,
        response: Optional[httpx.Response],
        exc: Exception,
    ) -> Any:
        """Extract error message from response or exception.

        Args:
            response: The HTTP response object, if available.
            exc: The exception that was raised.

        Returns:
            A descriptive error message string.
        """
        if response is not None:
            try:
                response_json = response.json()
                if isinstance(response_json, dict):
                    error_data = response_json.get("error", {})
                    if isinstance(error_data, dict):
                        return error_data.get("message", str(exc))
            except Exception:
                logger.exception(
                    "%s: Could not parse error from response body",
                    self._LOG_PREFIX,
                )
        return str(exc)

    @staticmethod
    def _parse_env_float(env_var: str, default: float) -> float:
        """Read an environment variable and parse it as a positive float.

        Values that are zero or negative are treated as invalid and the
        default is returned instead.

        Args:
            env_var: Name of the environment variable.
            default: Value to return when the variable is unset or invalid.

        Returns:
            The parsed float (> 0), or default on failure.
        """
        if not env_var:
            return default
        raw = os.getenv(env_var)
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            logger.exception(
                "Invalid value '%s' for %s, using default %.1f",
                raw,
                env_var,
                default,
            )
            return default

        if value <= 0:
            logger.exception(
                "Timeout value must be positive, got %.1f for %s; using default %.1f",
                value,
                env_var,
                default,
            )
            return default

        return value
