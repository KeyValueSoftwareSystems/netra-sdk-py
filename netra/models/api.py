import logging
from typing import Any, List, Optional

from netra.config import Config
from netra.models.client import ModelsHttpClient
from netra.models.constants import LOG_PREFIX

logger = logging.getLogger(__name__)


class Models:
    """Public entry-point exposed as Netra.models."""

    __slots__ = ("_config", "_client")

    def __init__(self, config: Config) -> None:
        """Initialize the models client.

        Args:
            config: The configuration object.
        """
        self._config = config
        self._client = ModelsHttpClient(config)

    def close(self) -> None:
        """Release resources held by the models client."""
        self._client.close()

    def get_model_pricing(self, name: Optional[str] = None) -> Optional[List[Any]]:
        """Fetch models for the project associated with the configured API key.

        Args:
            name: Optional model name to filter results.

        Returns:
            List of model dicts from the API response, or None on failure.
        """
        result = self._client.get_model_pricing(name=name)

        if result is None:
            return None

        items = result.get("data", []) or []
        if not isinstance(items, list):
            logger.error("%s: Unexpected response format; 'data' is not a list", LOG_PREFIX)
            return None

        return items
