import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)


class PromptsHttpClient:
    """
    Internal HTTP client for prompts APIs.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the prompts HTTP client.

        Args:
            config: Configuration object containing API key and base URL
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """
        Create and configure the HTTP client.

        Args:
            config: Configuration object containing API key and base URL

        Returns:
            Configured HTTP client or None if initialization fails
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.prompts: NETRA_OTLP_ENDPOINT is required for prompts APIs")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.prompts: Failed to initialize prompts HTTP client: %s", exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """
        Resolve the base URL by removing /telemetry suffix if present.

        Args:
            endpoint: The endpoint URL

        Returns:
            Resolved base URL
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """
        Build HTTP headers for API requests.

        Args:
            config: Configuration object containing API key and base URL

        Returns:
            Dictionary of HTTP headers
        """
        headers: Dict[str, str] = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _get_timeout(self) -> float:
        """
        Get the timeout value from environment variable or use default.

        Returns:
            Timeout value in seconds
        """
        timeout_env = os.getenv("NETRA_PROMPTS_TIMEOUT")
        if not timeout_env:
            return 10.0
        try:
            return float(timeout_env)
        except ValueError:
            logger.warning(
                "netra.prompts: Invalid NETRA_PROMPTS_TIMEOUT value '%s', using default 10.0",
                timeout_env,
            )
            return 10.0

    def get_prompt_version(self, prompt_name: str, label: str) -> Any:
        """
        Fetch a prompt version by name and label.

        Args:
            prompt_name: Name of the prompt
            label: Label of the prompt version

        Returns:
            Prompt version data or empty dict if not found
        """
        if not self._client:
            logger.error(
                "netra.prompts: Prompts client is not initialized; cannot fetch prompt version for '%s'",
                prompt_name,
            )
            return {}

        try:
            url = "/sdk/prompts/version"
            payload: Dict[str, Any] = {"promptName": prompt_name, "label": label}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error(
                "netra.prompts: Failed to fetch prompt version for '%s' (label=%s): %s",
                prompt_name,
                label,
                exc,
            )
            return {}
