"""Utility functions for the redteam module."""

import logging
import os
from typing import Any, Callable, Optional

from netra.redteam.constants import DEFAULT_MAX_CONCURRENCY, LOG_PREFIX

logger = logging.getLogger(__name__)


def parse_env_float(env_var: str, default: float) -> float:
    """Read an environment variable and parse it as a float.

    Args:
        env_var: Name of the environment variable.
        default: Value to return when the variable is unset or invalid.

    Returns:
        The parsed float, or *default* on failure.
    """
    raw = os.getenv(env_var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s: Invalid value '%s' for %s, using default %.1f",
            LOG_PREFIX,
            raw,
            env_var,
            default,
        )
        return default


def validate_redteam_inputs(
    config_id: str,
    handler: Optional[Callable[..., Any]],
    max_concurrency: Optional[int],
) -> bool:
    """Validate required inputs for ``run_redteam`` before any network call.

    Args:
        config_id: The red-team config identifier.
        handler: The user-supplied per-turn callback.
        max_concurrency: The requested concurrency bound, or ``None``.

    Returns:
        True if inputs are valid, False otherwise.
    """
    if not config_id:
        logger.error("%s: config_id is required", LOG_PREFIX)
        return False
    if not callable(handler):
        logger.error("%s: handler must be a callable", LOG_PREFIX)
        return False
    if max_concurrency is not None and (not isinstance(max_concurrency, int) or max_concurrency <= 0):
        logger.error("%s: max_concurrency must be a positive integer", LOG_PREFIX)
        return False
    return True


def resolve_max_concurrency(max_concurrency: Optional[int]) -> int:
    """Resolve the effective concurrency bound, capped at ``DEFAULT_MAX_CONCURRENCY``."""
    requested = max_concurrency if max_concurrency is not None else DEFAULT_MAX_CONCURRENCY
    return min(DEFAULT_MAX_CONCURRENCY, requested)


def unwrap_envelope(raw: Any) -> Any:
    """Unwrap one level of the backend's ``{success, data, error, meta}`` envelope."""
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw
