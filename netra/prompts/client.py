import logging
from typing import Any, Dict

from netra.client import BaseNetraClient
from netra.config import Config

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.prompts"


class PromptsHttpClient(BaseNetraClient):
    """Internal HTTP client for prompts APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the prompts HTTP client.

        Args:
            config: Configuration object containing API key and base URL.
        """
        super().__init__(
            config,
            log_prefix=_LOG_PREFIX,
            timeout_env_var="NETRA_PROMPTS_TIMEOUT",
        )

    def get_prompt_version(self, prompt_name: str, label: str) -> Any:
        """
        Fetch a prompt version by name and label.

        Args:
            prompt_name: Name of the prompt.
            label: Label of the prompt version.

        Returns:
            Prompt version data or None if not found.
        """
        if not self._client:
            logger.error(
                "%s: Client is not initialized; cannot fetch prompt version for '%s'",
                _LOG_PREFIX,
                prompt_name,
            )
            return None

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
                "%s: Failed to fetch prompt version for '%s' (label=%s): %s",
                _LOG_PREFIX,
                prompt_name,
                label,
                self._extract_error_message(exc),
            )
            return None
