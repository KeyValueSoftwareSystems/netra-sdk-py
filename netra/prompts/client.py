import logging
from typing import Any, Optional

import httpx

from netra.client import BaseHttpClient
from netra.prompts.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    URL_PROMPT_VERSION,
)
from netra.prompts.utils import build_prompt_version_payload

logger = logging.getLogger(__name__)


class PromptsHttpClient(BaseHttpClient):
    """Internal HTTP client for prompts APIs."""

    __slots__ = ()

    _LOG_PREFIX = LOG_PREFIX
    _ENV_TIMEOUT = ENV_TIMEOUT
    _DEFAULT_TIMEOUT = DEFAULT_TIMEOUT

    def get_prompt_version(self, prompt_name: str, label: str) -> Optional[Any]:
        """Fetch a prompt version by name and label.

        Args:
            prompt_name: Name of the prompt.
            label: Label of the prompt version.

        Returns:
            Prompt version data, or None on failure.
        """
        client = self._ensure_client()
        if client is None:
            return None

        response: Optional[httpx.Response] = None
        try:
            payload = build_prompt_version_payload(prompt_name=prompt_name, label=label)
            response = client.post(URL_PROMPT_VERSION, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data")
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception(
                "%s: Failed to fetch prompt version for '%s' (label=%s): %s",
                LOG_PREFIX,
                prompt_name,
                label,
                error_msg,
            )
            return None
