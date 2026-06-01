import logging
from typing import Any, Optional

from netra.config import Config
from netra.prompts.client import PromptsHttpClient
from netra.prompts.constants import LOG_PREFIX

logger = logging.getLogger(__name__)


class Prompts:
    """Public entry-point exposed as Netra.prompts."""

    __slots__ = ("_config", "_client")

    def __init__(self, cfg: Config) -> None:
        """Initialize the Prompts client.

        Args:
            cfg: Configuration object containing API key and base URL.
        """
        self._config = cfg
        self._client = PromptsHttpClient(cfg)

    def close(self) -> None:
        """Release resources held by the prompts client."""
        self._client.close()

    def get_prompt(self, name: str, label: str = "production") -> Optional[Any]:
        """Fetch a prompt version by name and label.

        Args:
            name: Name of the prompt.
            label: Label of the prompt version (default: "production").

        Returns:
            Prompt version data, or None on failure.
        """
        if not name:
            logger.error("%s: name is required to fetch a prompt", LOG_PREFIX)
            return None

        return self._client.get_prompt_version(prompt_name=name, label=label)
