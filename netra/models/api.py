import logging
from typing import Any, List, Optional

from netra.cache import TTLCache
from netra.config import Config
from netra.models.client import ModelsHttpClient

logger = logging.getLogger(__name__)

MODEL_PRICING_CACHE_TTL_SECONDS = 300


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
        self._cache: TTLCache[Any] = TTLCache(default_ttl=MODEL_PRICING_CACHE_TTL_SECONDS)

    def clear_cache(self) -> None:
        """Clear all cached model pricing entries."""
        self._cache.clear()

    def get_model_pricing(
        self,
        name: Optional[str] = None,
        use_cache: bool = False,
        cache_ttl: Optional[int] = None,
    ) -> List[Any] | Any:
        """
        Fetch models for the project associated with the configured API key.

        Args:
            name: Optional model name to filter results.
            use_cache: When True, read/write the in-memory cache (default: False).
            cache_ttl: Per-call cache TTL in seconds (default: 300).

        Returns:
            List of model dicts from the API response, or empty list on failure.
            When use_cache is True, do not mutate the returned list or nested
            dicts/prices — the same objects may be served on later cache hits.
        """
        cache_key = f"model:pricing:{name or 'all'}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = self._client.get_model_pricing(name=name)

        if not isinstance(result, dict):
            return result

        # Client failure sentinel is {}; do not treat as a successful empty list.
        if not result:
            return []

        items = result.get("data", []) or []
        if not isinstance(items, list):
            logger.error("netra.models: Unexpected response format; 'data' is not a list")
            return []

        if use_cache:
            self._cache.set(cache_key, items, cache_ttl)

        return items
