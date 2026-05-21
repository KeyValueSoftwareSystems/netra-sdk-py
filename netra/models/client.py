import logging
import os
from typing import Any, Dict, Optional

import httpx

from netra.config import Config

logger = logging.getLogger(__name__)


class ModelsHttpClient:
    """Internal HTTP client for models APIs."""

    def __init__(self, config: Config) -> None:
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.models: NETRA_OTLP_ENDPOINT is required for models APIs")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.models: Failed to initialize models HTTP client: %s", exc)
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
        timeout_env = os.getenv("NETRA_MODELS_TIMEOUT")
        if not timeout_env:
            return 10.0
        try:
            return float(timeout_env)
        except ValueError:
            logger.warning(
                "netra.models: Invalid NETRA_MODELS_TIMEOUT value '%s', using default 10.0",
                timeout_env,
            )
            return 10.0

    def list_models(self) -> Any:
        if not self._client:
            logger.error("netra.models: Models client is not initialized.")
            return {}

        try:
            response = self._client.get("/evaluations/models")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("netra.models: Failed to fetch models: %s", exc)
            return {}
