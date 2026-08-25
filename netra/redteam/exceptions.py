"""Typed exceptions raised by the redteam module."""

from typing import Optional


class RedteamError(Exception):
    """Base class for all redteam-related errors.

    Attributes:
        run_id: The run this error relates to, when known, so a caller can
            manually cancel an orphaned run after a fatal failure.
    """

    def __init__(self, message: str, run_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class RedteamAuthError(RedteamError):
    """Missing/invalid API key, or the feature is disabled for the org."""


class RedteamConfigError(RedteamError):
    """Missing, malformed, or unusable config/run/prompt."""


class RedteamRunError(RedteamError):
    """A run is already active for this config, or not in a status that
    accepts the requested operation."""


class RedteamGenerationError(RedteamError):
    """Prompt generation failed on the backend."""


class RedteamGenerationTimeoutError(RedteamError):
    """Prompt generation didn't finish before the deadline."""
