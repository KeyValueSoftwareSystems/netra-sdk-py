"""General utility helpers for Netra SDK.

This module centralizes common helpers that can be reused across the codebase.
"""

from __future__ import annotations

import logging
from typing import AbstractSet, Any, Optional, Set

import httpx

from netra.config import get_attribute_max_len
from netra.instrumentation.instruments import (
    DEFAULT_INSTRUMENTS_FOR_ROOT,
    InstrumentSet,
    NetraInstruments,
)

logger = logging.getLogger(__name__)


def extract_error_message(response: Optional[httpx.Response], exc: Exception) -> str:
    """Extract a human-readable error message from a Netra backend HTTP error.

    Netra backend error bodies follow the shape ``{"error": {"message": ...}}``
    (produced by the backend's global HTTP exception filter). When the body is
    absent, not JSON, or missing that field, fall back to ``str(exc)`` so the
    caller always has something loggable.

    Args:
        response: The HTTP response tied to the failure, or ``None`` when the
            request failed before a response was received (e.g. a connection
            error or timeout).
        exc: The exception that was raised.

    Returns:
        The backend-provided error message, or ``str(exc)`` as a fallback.
    """
    if response is not None:
        try:
            body = response.json()
            error_data = body.get("error", {})
            if isinstance(error_data, dict) and "message" in error_data:
                message = error_data["message"]
                return message if isinstance(message, str) else str(message)
        except Exception:
            logger.debug("utils: could not parse error from response body", exc_info=True)
    return str(exc)


def truncate_string(value: str, max_len: int) -> str:
    """Truncate a string to max_len characters.

    Args:
        value: The string to truncate
        max_len: The maximum length of the string
    """
    try:
        if not isinstance(value, str):
            return value
        return value if len(value) <= max_len else value[:max_len]
    except Exception:
        return value


def truncate_and_repair_json(content: Any, max_len: int) -> Any:
    """Truncate a dict/list by JSON-serializing and hard-cutting, then attempt repair.

    Args:
        content: The content to truncate
        max_len: The maximum length of the content
    """
    try:
        import json

        json_str = json.dumps(content, default=str)
        if len(json_str) <= max_len:
            return content

        truncated = json_str[:max_len]

        # Try json_repair if available
        repaired_obj: Any = None
        try:
            try:
                from json_repair import repair_json as _repair_json
            except Exception:  # pragma: no cover - optional dependency not installed
                _repair_json = None

            if _repair_json is not None:
                repaired_str = _repair_json(truncated)
                repaired_obj = json.loads(repaired_str)
        except Exception:
            repaired_obj = None

        if repaired_obj is not None:
            return repaired_obj

        # Fallback: safe container preserving a preview
        return {"__truncated__": True, "preview": truncated}
    except Exception:
        # If anything goes wrong, return original content as-is
        return content


def process_content_for_max_len(content: Any, max_len: int) -> Any:
    """Ensure the content fits within max_len when serialized.

    Args:
        content: The content to process
        max_len: The maximum length of the content
    """
    try:
        if isinstance(content, str):
            return truncate_string(content, max_len)
        if isinstance(content, (dict, list)):
            return truncate_and_repair_json(content, max_len)
        return content
    except Exception:
        return content


def serialize_value(value: Any) -> str:
    """Serialize *value* to a string capped at the active config's attribute max length."""
    if value is None:
        return ""
    try:
        import json

        serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        return truncate_string(serialized, get_attribute_max_len())
    except Exception:
        logger.debug("utils: failed to serialize value", exc_info=True)
        return ""


def resolve_root_instruments(
    root_instruments: Optional[AbstractSet[NetraInstruments]],
    block_instruments: Optional[AbstractSet[NetraInstruments]],
) -> Optional[Set[str]]:
    """Resolve the effective root instrument allow-list for the
    ``RootInstrumentFilterProcessor``.

    ``root_instruments`` is resolved independently of the non-root
    ``instruments`` set.  ``block_instruments`` is subtracted from the
    resolved root set.

    Args:
        root_instruments: User-supplied root instrument set.  ``None`` falls
            back to ``DEFAULT_INSTRUMENTS_FOR_ROOT``.  A set containing
            ``InstrumentSet.ALL`` enables all root instruments.
        block_instruments: Instruments to block.  ``None`` means no
            instruments are blocked.  Subtracted from the resolved root
            set.  A set containing ``InstrumentSet.ALL`` blocks everything.

    Returns:
        A set of instrumentation-name strings to pass to the
        ``RootInstrumentFilterProcessor``, or ``None`` when no filtering
        should be applied (every instrumentation may create root spans).
    """
    all_sentinel = InstrumentSet.ALL
    root_has_all = root_instruments is not None and all_sentinel in root_instruments
    block_has_all = block_instruments is not None and all_sentinel in block_instruments

    if block_has_all:
        if root_has_all:
            logger.error(
                "root_instruments=ALL is contradicted by "
                "block_instruments=ALL; all root instrumentation is disabled."
            )
        else:
            logger.warning("block_instruments contains ALL; all instrumentation will be disabled.")

    all_instrument_values: Set[str] = {m.value for m in NetraInstruments if m is not all_sentinel}

    blocked_root_values: Set[str] = set()
    if block_has_all:
        blocked_root_values = all_instrument_values.copy()
    elif block_instruments:
        blocked_root_values = {m.value for m in block_instruments if m is not all_sentinel}

    resolved_root: Optional[Set[str]] = None
    if root_has_all:
        if blocked_root_values:
            resolved_root = all_instrument_values - blocked_root_values
        else:
            resolved_root = None
    else:
        effective_root = root_instruments if root_instruments is not None else DEFAULT_INSTRUMENTS_FOR_ROOT
        resolved_root = {m.value for m in effective_root if m is not all_sentinel} - blocked_root_values

    return resolved_root
