"""
Shared builders for structured ``input``/``output`` span attributes.
Output format: ``[{"role": "…", "content": "…"}, …]``
"""

import json
import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


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


def build_messages(entries: Sequence[Dict[str, str]]) -> str:
    """Serialise a list of ``{role, content}`` dicts to a JSON string."""
    return json.dumps(list(entries))


def _extract_tool_call(tc: Any) -> Dict[str, str]:
    """Extract a tool-call into a ``{role, content}`` entry."""
    if isinstance(tc, dict):
        func = tc.get("function", {})
        name = func.get("name", "") if isinstance(func, dict) else ""
        arguments = func.get("arguments", "") if isinstance(func, dict) else ""
    else:
        func = getattr(tc, "function", None)
        if func is None:
            return {}
        name = getattr(func, "name", "") or ""
        arguments = getattr(func, "arguments", "") or ""

    return {
        "role": "assistant",
        "content": json.dumps({"name": name, "arguments": arguments}),
    }


def build_chat_input(messages: Any) -> str:
    """Build input JSON from OpenAI Chat Completions messages.

    Compatible with OpenAI, LiteLLM, Groq, Cerebras, MistralAI.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return "[]"

    entries: List[Dict[str, str]] = []

    for message in messages:
        if not isinstance(message, dict):
            message = model_as_dict(message)
        if not message:
            continue

        role = message.get("role", "user")

        if content := message.get("content"):
            entries.append({"role": role, "content": str(content)})

        for tc in message.get("tool_calls") or []:
            entry = _extract_tool_call(tc)
            if entry:
                entries.append(entry)

    return json.dumps(entries)


def build_response_api_input(kwargs: Dict[str, Any]) -> str:
    """Build input JSON from OpenAI Responses API kwargs."""
    entries: List[Dict[str, str]] = []

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
                            "content": json.dumps({"name": name, "arguments": arguments}),
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

    return json.dumps(entries)


def build_completion_output(response_dict: Any) -> str:
    """Build output JSON from an OpenAI-compatible response dict.

    Handles Responses API (output_text, output[]) and Chat Completions
    API (choices[].message, choices[].delta).
    """
    if not isinstance(response_dict, dict):
        return "[]"

    entries: List[Dict[str, str]] = []

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
                        "content": json.dumps({"name": name, "arguments": arguments}),
                    }
                )
            elif content := element.get("content"):
                for chunk in content:
                    if isinstance(chunk, dict):
                        if text := chunk.get("text"):
                            entries.append({"role": "assistant", "content": text})

    # Chat Completions API
    if choices := response_dict.get("choices"):
        for choice in choices:
            if not isinstance(choice, dict):
                continue

            if message := choice.get("message"):
                if isinstance(message, dict):
                    if content := message.get("content"):
                        entries.append(
                            {
                                "role": message.get("role", "assistant"),
                                "content": content,
                            }
                        )
                    for tc in message.get("tool_calls") or []:
                        entry = _extract_tool_call(tc)
                        if entry:
                            entries.append(entry)

            elif delta := choice.get("delta"):
                if isinstance(delta, dict):
                    if content := delta.get("content"):
                        entries.append(
                            {
                                "role": delta.get("role", "assistant"),
                                "content": content,
                            }
                        )
                    for tc in delta.get("tool_calls") or []:
                        entry = _extract_tool_call(tc)
                        if entry:
                            entries.append(entry)

    return json.dumps(entries)
