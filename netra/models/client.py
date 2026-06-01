import logging
from typing import Any, Dict, Optional

import httpx

from netra.client import BaseHttpClient
from netra.models.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    URL_MODELS,
)
from netra.models.utils import build_model_pricing_params

logger = logging.getLogger(__name__)


class ModelsHttpClient(BaseHttpClient):
    """Internal HTTP client for models APIs."""

    __slots__ = ()

    _LOG_PREFIX = LOG_PREFIX
    _ENV_TIMEOUT = ENV_TIMEOUT
    _DEFAULT_TIMEOUT = DEFAULT_TIMEOUT

    def get_model_pricing(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch models from the /sdk/models endpoint.

        Args:
            name: Optional model name to filter results.

        Returns:
            Raw JSON response dict, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            params = build_model_pricing_params(name=name)
            response = client.get(URL_MODELS, params=params or None)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.warning("%s: Unexpected response type from models endpoint", LOG_PREFIX)
                return None
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to fetch models: %s", LOG_PREFIX, error_msg)
            return None
