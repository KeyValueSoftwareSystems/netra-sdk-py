"""General utility helpers for Netra SDK.

This module centralizes common helpers that can be reused across the codebase.
"""

from __future__ import annotations

from typing import Any


def truncate_string(value: str, max_len: int) -> str:
    """Truncate a string to max_len characters.

    If value is not a string, it is returned unchanged.
    """
    try:
        if not isinstance(value, str):
            return value
        return value if len(value) <= max_len else value[:max_len]
    except Exception:
        return value


def truncate_and_repair_json(content: Any, max_len: int) -> Any:
    """Truncate a dict/list by JSON-serializing and hard-cutting, then attempt repair.

    The function will:
    - json.dumps(content, default=str)
    - hard-cut the string to max_len
    - try to repair using `json-repair` (optional dependency)
    - parse back with json.loads

    On failure, returns a minimal safe container with a preview of the truncated text.
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

    - If content is a string: truncate to max_len.
    - If content is a dict or list: attempt truncate+repair to keep it valid JSON.
    """
    try:
        if isinstance(content, str):
            return truncate_string(content, max_len)
        if isinstance(content, (dict, list)):
            return truncate_and_repair_json(content, max_len)
        return content
    except Exception:
        return content
