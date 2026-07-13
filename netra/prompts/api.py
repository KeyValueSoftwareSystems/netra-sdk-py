import logging
from typing import Any, Optional

from netra.cache import TTLCache
from netra.config import Config
from netra.prompts.client import PromptsHttpClient

logger = logging.getLogger(__name__)

PROMPT_CACHE_TTL_SECONDS = 60


class Prompts:
    """
    Public entry-point exposed as Netra.prompts
    """

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the Prompts client.

        Args:
            cfg: Configuration object containing API key and base URL
        """
        self._config = cfg
        self._client = PromptsHttpClient(cfg)
        self._cache: TTLCache[Any] = TTLCache(default_ttl=PROMPT_CACHE_TTL_SECONDS)

    def clear_cache(self) -> None:
        """Clear all cached prompt entries."""
        self._cache.clear()

    def get_prompt(
        self,
        name: str,
        label: str = "production",
        use_cache: bool = False,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        """
        Fetch a prompt version by name and label.

        Args:
            name: Name of the prompt
            label: Label of the prompt version (default: "production")
            use_cache: When True, read/write the in-memory cache (default: False)
            cache_ttl: Per-call cache TTL in seconds (default: PROMPT_CACHE_TTL_SECONDS)

        Returns:
            Prompt version data or None/empty dict if not found
        """
        if not name:
            logger.error("netra.prompts: name is required to fetch a prompt")
            return None

        cache_key = f"prompt:{name}:{label}"

        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = self._client.get_prompt_version(prompt_name=name, label=label)

        if use_cache and result is not None and result != {}:
            self._cache.set(cache_key, result, cache_ttl)

        return result
