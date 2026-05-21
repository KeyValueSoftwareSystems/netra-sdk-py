import logging
from typing import Any, List

from netra.config import Config
from netra.models.client import ModelsHttpClient
from netra.models.models import Model, ModelPrice

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

    def list_models(self) -> List[Model] | Any:
        """
        Fetch models for the project associated with the configured API key.

        Returns:
            List of Model objects containing pricing information, or None on failure.
        """
        result = self._client.list_models()

        if not isinstance(result, dict):
            return result

        items = result.get("data", []) or []
        if not isinstance(items, list):
            logger.error("netra.models: Unexpected response format; 'data' is not a list")
            return []

        models = []
        for item in items:
            if not isinstance(item, dict):
                continue
            prices = [
                ModelPrice(
                    usage_type=p.get("usageType", ""),
                    min_units=p.get("minUnits", 0),
                    max_units=p.get("maxUnits"),
                    price=p.get("price"),
                    unit_value=p.get("unitValue", 0),
                )
                for p in item.get("prices", [])
                if isinstance(p, dict)
            ]
            models.append(
                Model(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    match_pattern=item.get("matchPattern", ""),
                    project_id=item.get("projectId"),
                    created_at=item.get("createdAt"),
                    updated_at=item.get("updatedAt"),
                    deleted_at=item.get("deletedAt"),
                    prices=prices,
                )
            )
        return models
