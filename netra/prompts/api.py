import logging
from typing import Any

from netra.config import Config
from netra.prompts.client import PromptsHttpClient

logger = logging.getLogger(__name__)


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

    def get_prompt(self, name: str, label: str = "production") -> Any:
        """
        Fetch a prompt version by name and label.

        Args:
            name: Name of the prompt
            label: Label of the prompt version (default: "production")

        Returns:
            Prompt version data or empty dict if not found
        """
        if not name:
            logger.error("netra.prompts: name is required to fetch a prompt")
            return None

        return self._client.get_prompt_version(prompt_name=name, label=label)
