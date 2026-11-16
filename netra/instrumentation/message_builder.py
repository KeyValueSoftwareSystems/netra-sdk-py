"""
Shared builders for structured ``input``/``output`` span attributes.
Output format: ``[{"role": "…", "content": "…"}, …]``
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from netra.config import get_attribute_max_len, get_conversation_max_len
from netra.instrumentation.capture.bounded_capture import TRUNCATION_ELLIPSIS, TRUNCATION_MARKER_KEY
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# ``json.dumps`` default item separator, counted when fitting entries to a budget.
_ITEM_SEPARATOR_LEN = len(", ")

# Max re-measure rounds when shortening contents to fit.
_MAX_FIT_ROUNDS = 8


def model_as_dict(obj: Any) -> Dict[str, Any]:
    """Convert an SDK model object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        result = obj.model_dump()
        return result if isinstance(result, dict) else {}
    if hasattr(obj, "to_dict"):
        result = obj.to_dict()
        return result if isinstance(result, dict) else {}
    return {}


def to_json(value: Any) -> str:
    """Serialise *value* to JSON, encoding SDK model objects as dicts and anything else as text."""
    return json.dumps(value, default=lambda obj: model_as_dict(obj) or str(obj))


def build_messages(entries: Sequence[Any]) -> str:
    """Serialise message entries to a JSON array that fits the attribute length limit.

    Over the limit, every entry keeps its position and long contents are shortened
    and marked truncated. The result is always valid JSON.
    """
    capped = [cap_message_content(entry) for entry in entries]
    serialised = to_json(capped)
    max_len = get_attribute_max_len()
    if len(serialised) <= max_len:
        return serialised
    return _fit_all(capped, max_len)


def _string_content(entry: Any) -> Optional[str]:
    """Return an entry's ``content`` if it is a string, else ``None``."""
    content = entry.get("content") if isinstance(entry, dict) else None
    return content if isinstance(content, str) else None


def _content_text(entry: Any) -> Optional[str]:
    """Return an entry's ``content`` as text that can be shortened, or ``None`` when it has none."""
    if not isinstance(entry, dict) or entry.get("content") is None:
        return None
    content = entry["content"]
    return content if isinstance(content, str) else to_json(content)


def _with_truncated_content(entry: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Copy *entry* with *content* plus the ellipsis, marked as truncated."""
    return {**entry, "content": content + TRUNCATION_ELLIPSIS, TRUNCATION_MARKER_KEY: True}


def cap_message_content(entry: Any) -> Any:
    """Cap an entry's string ``content`` at the conversation content limit, marking it truncated.

    Returns the entry unchanged when it has no string content or already fits.
    """
    content = _string_content(entry)
    limit = get_conversation_max_len()
    if content is None or len(content) <= limit:
        return entry
    return _with_truncated_content(entry, content[: max(limit - len(TRUNCATION_ELLIPSIS), 0)])


def _fit_all(entries: List[Any], max_len: int) -> str:
    """Shorten contents so all *entries* fit in *max_len*, keeping each entry in its position."""
    texts = [_content_text(entry) for entry in entries]
    skeleton = [entry if text is None else _with_truncated_content(entry, "") for entry, text in zip(entries, texts)]
    sizes = [0 if text is None else len(to_json(text)) - len('""') for text in texts]
    budget = max_len - len(to_json(skeleton))
    for _ in range(_MAX_FIT_ROUNDS):
        if budget <= 0:
            break
        limits = _fair_shares(sizes, budget)
        fitted = [_shorten(entry, text, size, limit) for entry, text, size, limit in zip(entries, texts, sizes, limits)]
        serialised = to_json(fitted)
        overflow = len(serialised) - max_len
        if overflow <= 0:
            return serialised
        budget -= overflow
    return _fit_leading(skeleton, max_len)


def _fair_shares(sizes: List[int], budget: int) -> List[int]:
    """Split *budget* across *sizes*; sizes under an equal share stay whole."""
    limits = [0] * len(sizes)
    remaining = budget
    left = sum(1 for size in sizes if size)
    for index in sorted(range(len(sizes)), key=sizes.__getitem__):
        if not sizes[index]:
            continue
        limits[index] = min(sizes[index], remaining // left)
        remaining -= limits[index]
        left -= 1
    return limits


def _shorten(entry: Any, text: Optional[str], size: int, limit: int) -> Any:
    """Cut *entry*'s content to about *limit* serialised characters."""
    if text is None or limit >= size:
        return entry
    return _with_truncated_content(entry, text[: len(text) * limit // size])


def _fit_leading(skeleton: List[Any], max_len: int) -> str:
    """Serialise the leading entries of *skeleton* that fit in *max_len*."""
    kept: List[Any] = []
    budget = max_len - len("[]")
    for entry in skeleton:
        size = len(to_json(entry)) + (_ITEM_SEPARATOR_LEN if kept else 0)
        if size > budget:
            break
        kept.append(entry)
        budget -= size
    logger.debug("Kept %d of %d message(s) to fit the attribute length limit", len(kept), len(skeleton))
    return to_json(kept)


def _extract_tool_call(tc: Any) -> Dict[str, Any]:
    """Extract a tool-call into a ``{role, content}`` entry, preserving ``tool_call_id``."""
    if isinstance(tc, dict):
        func = tc.get("function", {})
        name = func.get("name", "") if isinstance(func, dict) else ""
        arguments = func.get("arguments", "") if isinstance(func, dict) else ""
        tc_id = tc.get("id")
    else:
        func = getattr(tc, "function", None)
        if func is None:
            return {}
        name = getattr(func, "name", "") or ""
        arguments = getattr(func, "arguments", "") or ""
        tc_id = getattr(tc, "id", None)

    entry: Dict[str, Any] = {
        "role": "assistant",
        "content": to_json({"name": name, "arguments": arguments}),
    }
    if tc_id:
        entry["tool_call_id"] = tc_id
    return entry


def build_chat_input(messages: Any) -> Optional[str]:
    """Build input JSON from OpenAI Chat Completions messages.

    Compatible with OpenAI, LiteLLM, Groq, Cerebras, MistralAI.
    Returns ``None`` when there are no messages to record.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return None

    entries: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            message = model_as_dict(message)
        if not message:
            continue

        role = message.get("role", "user")

        if content := message.get("content"):
            entry: Dict[str, Any] = {"role": role, "content": str(content)}
            if role == "tool" and message.get("tool_call_id"):
                entry["tool_call_id"] = message["tool_call_id"]
            entries.append(entry)

        for tc in message.get("tool_calls") or []:
            entry = _extract_tool_call(tc)
            if entry:
                entries.append(entry)

    return build_messages(entries) if entries else None


def build_response_api_input(kwargs: Dict[str, Any]) -> Optional[str]:
    """Build input JSON from OpenAI Responses API kwargs.

    Returns ``None`` when there are no messages to record.
    """
    entries: List[Dict[str, Any]] = []

    if instructions := kwargs.get("instructions"):
        entries.append({"role": "system", "content": instructions})

    input_data = kwargs.get("input")
    if isinstance(input_data, str):
        entries.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list) and input_data:
        for message in input_data:
            if not isinstance(message, dict):
                continue

            msg_type = message.get("type", "")

            if msg_type == "function_call":
                name = message.get("name", "")
                arguments = message.get("arguments", "")
                if name or arguments:
                    entries.append(
                        {
                            "role": "assistant",
                            "content": to_json({"name": name, "arguments": arguments}),
                        }
                    )
            elif msg_type == "function_call_output":
                output_val = str(message.get("output") or "")
                if output_val:
                    entries.append({"role": "tool", "content": output_val})
            elif msg_type in ("reasoning", "reasoning_summary"):
                continue
            else:
                role = message.get("role", "user")
                content = message.get("content")
                if content:
                    entries.append({"role": role, "content": str(content)})

    return build_messages(entries) if entries else None


def build_completion_output(response_dict: Any) -> Optional[str]:
    """Build output JSON from an OpenAI-compatible response dict.

    Handles Responses API (output_text, output[]) and Chat Completions
    API (choices[].message, choices[].delta, choices[].text).
    Returns ``None`` when the response carries no output messages.
    """
    if not isinstance(response_dict, dict):
        return None

    entries: List[Dict[str, Any]] = []

    # Responses API
    if output_text := response_dict.get("output_text"):
        entries.append({"role": "assistant", "content": output_text})

    if output := response_dict.get("output"):
        for element in output:
            if not isinstance(element, dict):
                continue
            if element.get("type") == "function_call":
                name = element.get("name", "")
                arguments = element.get("arguments", "")
                entries.append(
                    {
                        "role": "assistant",
                        "content": to_json({"name": name, "arguments": arguments}),
                    }
                )
            elif content := element.get("content"):
                for chunk in content:
                    if isinstance(chunk, dict):
                        if text := chunk.get("text"):
                            entries.append({"role": "assistant", "content": text})

    # Chat Completions / legacy Completions API
    if choices := response_dict.get("choices"):
        for choice in choices:
            if not isinstance(choice, dict):
                continue

            choice_entries: List[Dict[str, Any]] = []
            message = choice.get("message") or choice.get("delta")
            if isinstance(message, dict):
                if content := message.get("content"):
                    choice_entries.append({"role": message.get("role", "assistant"), "content": content})
                for tc in message.get("tool_calls") or []:
                    tc_entry = _extract_tool_call(tc)
                    if tc_entry:
                        choice_entries.append(tc_entry)
            elif text := choice.get("text"):
                choice_entries.append({"role": "assistant", "content": text})

            # finish_reason belongs to the choice; record it on the choice's last entry.
            if choice_entries and (finish_reason := choice.get("finish_reason")):
                choice_entries[-1]["finish_reason"] = finish_reason
            entries.extend(choice_entries)

    return build_messages(entries) if entries else None
