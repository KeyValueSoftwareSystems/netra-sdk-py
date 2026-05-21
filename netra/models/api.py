import logging
from typing import Any, List

from netra.config import Config
from netra.models.client import ModelsHttpClient

logger = logging.getLogger(__name__)


class Models:
    """Public entry-point exposed as Netra.models"""

    def __init__(self, config: Config) -> None:
        """
        Initialize the models client.

        Args:
            config: The configuration object.
        """
        self._config = config
        self._client = ModelsHttpClient(config)

    def get_model_pricing(self) -> List[Any] | Any:
        """
        Fetch models for the project associated with the configured API key.

        Returns:
            List of model dicts from the API response, or None on failure.
        """
        result = self._client.get_model_pricing()

        if not isinstance(result, dict):
            return result

        items = result.get("data", []) or []
        if not isinstance(items, list):
            logger.error("netra.models: Unexpected response format; 'data' is not a list")
            return []

        return items
