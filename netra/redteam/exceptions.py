"""Typed exceptions raised by the redteam module."""

from typing import Optional


class RedTeamError(Exception):
    """Base class for all red-team-related errors.

    Attributes:
        run_id: The run this error relates to, when known, so a caller can
            manually cancel an orphaned run after a fatal failure.
    """

    def __init__(self, message: str, run_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class RedTeamAuthError(RedTeamError):
    """Missing/invalid API key, or the feature is disabled for the org."""


class RedTeamConfigError(RedTeamError):
    """Missing, malformed, or unusable config/run/prompt."""


class RedTeamRunError(RedTeamError):
    """A run is already active for this config, or not in a status that
    accepts the requested operation."""


class RedTeamGenerationError(RedTeamError):
    """Prompt generation failed on the backend."""


class RedTeamGenerationTimeoutError(RedTeamError):
    """Prompt generation didn't finish before the deadline."""
