import json
import math
import os
from typing import Any, Dict, List, Optional, TypeGuard

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace import Span

from netra.config import Config, get_attribute_max_len
from netra.instrumentation.llm_common import is_unset_sentinel, to_jsonable
from netra.netra_logger import get_logger
from netra.span_wrapper import ATTRIBUTE, UsageModel

logger = get_logger(__name__)

# A provider-reported cost covers the whole call (prompt, completion, cache read and write).
PROVIDER_REPORTED_USAGE_TYPE = "total"
CUSTOM_USAGE_ATTRIBUTE = f"{Config.LIBRARY_NAME}.{ATTRIBUTE.USAGE}"
LLM_USAGE_CACHE_CREATION_INPUT_TOKENS_30M = "gen_ai.usage.cache_creation_input_tokens_30m"
LLM_USAGE_CACHE_CREATION_INPUT_TOKENS_12H = "gen_ai.usage.cache_creation_input_tokens_12h"
# OpenAI splits GPT-5.6+ cache writes by TTL; each tier is billed at a different rate.
_CACHE_WRITE_TTL_FIELDS: Dict[str, str] = {
    "ttl_30m_tokens": LLM_USAGE_CACHE_CREATION_INPUT_TOKENS_30M,
    "ttl_12h_tokens": LLM_USAGE_CACHE_CREATION_INPUT_TOKENS_12H,
}
GEN_AI_REQUEST_TOOLS = "gen_ai.request.tools"
LLM_REQUEST_TOOL_CHOICE = "llm.request.tool_choice"
# Tool types whose name/description/schema live under a key named after the type in
# Chat Completions (``{"type": "function", "function": {...}}``) but flat in the Responses API.
_NAMED_TOOL_TYPES = ("function", "custom")
# Structured conversation attributes: one JSON array of ``{"role", "content"}`` entries
# each, replacing the per-message indexed ``gen_ai.prompt.N.*`` / ``gen_ai.completion.N.*`` keys.
INPUT_ATTRIBUTE = "input"
OUTPUT_ATTRIBUTE = "output"
# Decision input item types that carry a tool invocation, mapped to the key holding its arguments.
_DECISION_TOOL_CALL_ARGUMENT_KEYS: Dict[str, str] = {"function_call": "arguments", "custom_tool_call": "input"}
_DECISION_TOOL_OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
# A data URL's header (``data:<mime>;base64,``) is short; only this many leading
# characters are scanned for its comma so a large non-data-URL string is not walked.
_MAX_DATA_URL_HEADER_LEN = 256
# Separator ``json.dumps`` puts between array items; used when arrays are assembled from pre-encoded items.
_JSON_ITEM_SEPARATOR = ", "
# Decisions API (``client.decisions.create``): the questions are recorded as the last
# ``input`` entry and the answers as the ``output`` entry, each holding a JSON array.
_DECISION_QUESTIONS_ROLE = "user"
_DECISION_ANSWERS_ROLE = "assistant"
# The questions entry may take at most 1/N of the ``input`` attribute, leaving the rest for the input items.
_DECISION_QUESTIONS_MAX_SHARE = 2


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def should_send_prompts() -> bool:
    """Whether prompt/completion content may be recorded on spans.

    Content capture is on unless TRACELOOP_TRACE_CONTENT is "false" (Config sets that
    env var from its ``trace_content`` setting), and can be force-enabled per-context
    via ``override_enable_content_tracing``.
    """
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


def model_as_dict(input_object: Any) -> Any:
    """Convert OpenAI model object to dictionary"""
    if hasattr(input_object, "model_dump"):
        return input_object.model_dump()

    elif hasattr(input_object, "to_dict"):
        return input_object.to_dict()

    elif isinstance(input_object, dict):
        return input_object

    else:
        return {}


def set_request_attributes(span: Span, kwargs: Dict[str, Any], operation_type: str) -> None:
    """Set request attributes on span"""
    if not span.is_recording():
        logger.debug("Span is not recording")
        return

    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, operation_type)

    ATTRIBUTE_MAPPINGS = {
        "model": SpanAttributes.LLM_REQUEST_MODEL,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_completion_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_output_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "frequency_penalty": SpanAttributes.LLM_FREQUENCY_PENALTY,
        "presence_penalty": SpanAttributes.LLM_PRESENCE_PENALTY,
        "reasoning_effort": SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
        "stop": SpanAttributes.LLM_CHAT_STOP_SEQUENCES,
        "stream": SpanAttributes.LLM_IS_STREAMING,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
        "dimensions": "gen_ai.request.dimensions",
    }

    for key, attribute in ATTRIBUTE_MAPPINGS.items():
        value = kwargs.get(key)
        if value is not None and not is_unset_sentinel(value):
            span.set_attribute(attribute, value)

    # ``reasoning`` may be a dict, a pydantic model (``Reasoning``) or an unset sentinel
    # (``NOT_GIVEN`` / ``omit``, which openai-agents passes on every call).
    if (reasoning := to_jsonable(kwargs.get("reasoning"))) is not None:
        span.set_attribute(SpanAttributes.LLM_REQUEST_REASONING_EFFORT, json.dumps(reasoning, default=str))

    if operation_type == "chat":
        _set_chat_completion_input(span, kwargs.get("messages"))
    elif operation_type == "response":
        _set_chat_response_input(span, kwargs)
    elif operation_type == "decision":
        _set_decision_request_attributes(span, kwargs)

    _set_tool_attributes(span, kwargs)


def _set_decision_request_attributes(span: Span, kwargs: Dict[str, Any]) -> None:
    """Record a Decisions API request's input and questions as ``input``, without ever failing the call.

    Both are caller content, so nothing is recorded when content tracing is off.
    Runs before the OpenAI call and serialises caller-supplied content, so any
    failure is logged instead of aborting the user's request; questions that
    cannot be serialised leave the input items still recorded.
    """
    if not should_send_prompts():
        return

    questions_entry: Optional[str] = None
    try:
        questions_entry = _decision_questions_entry(kwargs.get("questions"))
    except Exception:  # noqa: BLE001 - question capture is best-effort; never block the user's request
        logger.warning("netra.instrumentation.libraries.openai: failed to record decision questions", exc_info=True)

    try:
        if input_json := _build_decision_input(kwargs.get("input"), questions_entry):
            span.set_attribute(INPUT_ATTRIBUTE, input_json)
    except Exception:  # noqa: BLE001 - input capture is best-effort; never block the user's request
        logger.warning("netra.instrumentation.libraries.openai: failed to record decision input", exc_info=True)


def _data_url_placeholder(value: str) -> Optional[str]:
    """Return a short placeholder if ``value`` is a base64 data URL, else None.

    ``data:image/png;base64,...`` becomes ``[image]``; any other media type
    (e.g. an inline PDF in ``file_data``) becomes ``[file]``.
    """
    if not value.startswith("data:"):
        return None
    header_end = value.find(",", 0, _MAX_DATA_URL_HEADER_LEN)
    if header_end == -1 or not value[:header_end].endswith(";base64"):
        return None
    return "[image]" if value.startswith("data:image/") else "[file]"


class _CharBudget:
    """Characters of string content one copy may still keep, shared across the whole walk."""

    __slots__ = ("remaining",)

    def __init__(self, remaining: int) -> None:
        self.remaining = remaining

    def take(self, text: str) -> str:
        """Return as much of ``text`` as the budget still allows, and charge it."""
        kept = text[: max(self.remaining, 0)]
        self.remaining -= len(kept)
        return kept


def _without_inline_data(value: Any, budget: Optional[_CharBudget] = None) -> Any:
    """Copy ``value`` with every base64 data URL replaced by a short placeholder.

    Image and file content parts usually carry their bytes as base64 data URLs;
    serialising those in full only for the attribute length limit to cut them
    leaves unreadable base64 on the span. When a ``budget`` is given, all the
    strings in the copy together keep at most that many characters, and list
    items past the point it runs out are left out, so serialising the copy
    never encodes more text than the attribute could hold, however the content
    is split into parts. The caller's objects are never mutated.
    """
    if isinstance(value, str):
        if placeholder := _data_url_placeholder(value):
            if budget is not None:
                budget.remaining -= len(placeholder)
            return placeholder
        return value if budget is None else budget.take(value)
    if isinstance(value, dict):
        return {key: _without_inline_data(item, budget) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        copy: List[Any] = []
        for item in value:
            if budget is not None and budget.remaining <= 0:
                break
            copy.append(_without_inline_data(item, budget))
        return copy
    if as_dict := model_as_dict(value):
        return _without_inline_data(as_dict, budget)
    return value


def _as_prompt_content(content: Any, max_len: int) -> str:
    """Render message content as a string of at most ``max_len`` characters.

    Text is kept as-is and content parts become JSON, with inline data elided
    and at most ``max_len`` characters of text across all parts kept before
    anything is encoded.
    """
    elided = _without_inline_data(content, _CharBudget(max_len))
    rendered = elided if isinstance(elided, str) else json.dumps(elided, default=str)
    return rendered[:max_len]


def _decision_input_entry(item: Dict[str, Any], max_content_len: int) -> Optional[Dict[str, Any]]:
    """Convert one Decisions API input item to a ``{"role", "content"}`` entry, or None to skip it.

    The entry's ``content`` is at most ``max_content_len`` characters long.
    """
    item_type = item.get("type") or "message"
    entry: Dict[str, Any]
    if item_type in _DECISION_TOOL_CALL_ARGUMENT_KEYS:
        arguments = item.get(_DECISION_TOOL_CALL_ARGUMENT_KEYS[item_type], "")
        if isinstance(arguments, str):
            arguments = arguments[:max_content_len]
        call = json.dumps({"name": item.get("name", ""), "arguments": arguments}, default=str)
        entry = {"role": "assistant", "content": call[:max_content_len]}
    elif item_type in _DECISION_TOOL_OUTPUT_TYPES:
        if not (output := item.get("output")):
            return None
        entry = {"role": "tool", "content": _as_prompt_content(output, max_content_len)}
    else:
        if not (message_content := item.get("content")):
            return None
        entry = {"role": item.get("role", "user"), "content": _as_prompt_content(message_content, max_content_len)}

    if item_type != "message" and (call_id := item.get("call_id")):
        entry["tool_call_id"] = call_id
    return entry


def _encode_entry_within(entry: Dict[str, Any], max_len: int) -> Optional[str]:
    """JSON-encode ``entry``, shortening its ``content`` until the encoding fits ``max_len``.

    Escaping makes the encoded content longer than the raw text (quotes and
    newlines double, non-ASCII characters become ``\\uXXXX``), so the content is
    cut in proportion to how far over the limit the encoding is and re-encoded.

    Returns:
        The encoded entry, or None when not even an empty ``content`` fits.
    """
    content: str = entry["content"]
    overhead = len(json.dumps({**entry, "content": ""}, default=str))
    if overhead > max_len:
        return None
    content_budget = max_len - overhead
    while True:
        encoded = json.dumps({**entry, "content": content}, default=str)
        encoded_content_len = len(encoded) - overhead
        if encoded_content_len <= content_budget:
            return encoded
        keep = len(content) * content_budget // encoded_content_len
        content = content[: min(keep, len(content) - 1)]


def _build_decision_input(input_data: Any, questions_entry: Optional[str] = None) -> Optional[str]:
    """Build the ``input`` attribute JSON for a Decisions API request.

    ``input`` is either a string (recorded as one user message) or a list of
    messages, function / custom tool calls and their outputs. Tool calls are
    recorded as assistant entries holding ``{"name", "arguments"}`` and tool
    outputs as ``tool`` entries; both carry their ``call_id`` as ``tool_call_id``.
    ``questions_entry``, the already-encoded questions entry, is appended last
    so the questions read after the content they are asked about.
    Only lists and tuples are walked: the SDK accepts any iterable, and
    consuming a one-shot iterator here would empty the request the caller is
    about to send.

    The result never exceeds the configured attribute length. The questions
    entry's room is set aside first, so a long input never pushes the questions
    out; each input entry's content is then cut to the budget the earlier
    entries left over and later entries are dropped once it is used up, so the
    attribute stays valid JSON and a multi-megabyte input is never encoded in full.

    Returns:
        A JSON array of ``{"role", "content"[, "tool_call_id"]}`` entries, or
        None when there is nothing to record.
    """
    if isinstance(input_data, str):
        items: Any = [{"role": "user", "content": input_data}] if input_data else []
    elif isinstance(input_data, (list, tuple)):
        items = input_data
    else:
        items = []

    encoded_entries: List[str] = []
    remaining = get_attribute_max_len() - len("[]")
    if questions_entry:
        remaining -= len(questions_entry) + len(_JSON_ITEM_SEPARATOR)
    for item in items:
        separator_len = len(_JSON_ITEM_SEPARATOR) if encoded_entries else 0
        budget = remaining - separator_len
        if budget <= 0:
            break
        if not isinstance(item, dict):
            item = model_as_dict(item)
        if not item or not (entry := _decision_input_entry(item, budget)):
            continue
        if (encoded := _encode_entry_within(entry, budget)) is None:
            break
        encoded_entries.append(encoded)
        remaining = budget - len(encoded)

    if questions_entry:
        encoded_entries.append(questions_entry)
    return _join_json_array(encoded_entries) if encoded_entries else None


def _join_json_array(encoded_items: List[str]) -> str:
    """Join already-encoded JSON values into a JSON array, formatted as ``json.dumps`` would."""
    return "[" + _JSON_ITEM_SEPARATOR.join(encoded_items) + "]"


def _fitting_prefix_len(item_lengths: List[int], max_len: int) -> int:
    """How many leading items fit in a JSON array of at most ``max_len`` characters.

    ``item_lengths`` are the encoded length of each item; the array adds its
    brackets and a separator between items.
    """
    used = len("[]")
    for count, item_len in enumerate(item_lengths):
        used += item_len + (len(_JSON_ITEM_SEPARATOR) if count else 0)
        if used > max_len:
            return count
    return len(item_lengths)


def _encode_items_within(items: List[Any], max_len: int) -> List[str]:
    """JSON-encode ``items`` in order, stopping once they no longer fit a ``max_len`` array.

    Inline data is elided first, and each item's text is copied only up to the
    room left in the array, so no item costs more to encode than the attribute
    could still hold. Items are kept whole: an item whose text alone uses up
    that room cannot fit once encoded, so it and everything after it is left out.
    """
    encoded: List[str] = []
    used = len("[]")
    for item in items:
        room = max_len - used - (len(_JSON_ITEM_SEPARATOR) if encoded else 0)
        budget = _CharBudget(room)
        copy = _without_inline_data(item, budget)
        if budget.remaining <= 0:
            break
        value = json.dumps(copy, default=str)
        used += len(value) + (len(_JSON_ITEM_SEPARATOR) if encoded else 0)
        if used > max_len:
            break
        encoded.append(value)
    return encoded


def _items_entry(items: List[Any], role: str, max_len: int) -> Optional[str]:
    """Encode ``{"role": role, "content": "<JSON array of items>"}`` in at most ``max_len`` characters.

    The items' JSON is escaped again inside ``content``, so each item is sized
    as such and only the leading items that fit are kept, whole, so ``content``
    stays valid JSON.

    Returns:
        The encoded entry, or None when ``items`` is non-empty and not even the
        first item fits.
    """
    encoded = _encode_items_within(items, max_len)
    envelope_len = len(json.dumps({"role": role, "content": ""}))
    escaped_lengths = [len(json.dumps(value)) - len('""') for value in encoded]
    kept = encoded[: _fitting_prefix_len(escaped_lengths, max_len - envelope_len)]
    if items and not kept:
        return None
    return json.dumps({"role": role, "content": _join_json_array(kept)})


def _decision_questions_entry(questions: Any) -> Optional[str]:
    """Encode the Decisions API ``questions`` as the ``user`` entry that ends the ``input`` attribute.

    Its ``content`` is the questions as a JSON array, recorded as given
    (``type``, ``instructions``, ``name`` and the ``choices`` / ``levels`` they
    offer) with inline images in their instructions replaced by a placeholder.
    The entry may use at most half the attribute length, so however many
    questions are asked the input they are about still has room; whole
    questions are dropped from the end past that. As with ``input``, only
    lists and tuples are read so a one-shot iterable reaches the API intact.

    Returns:
        The encoded entry, or None when there are no questions to record.
    """
    if not isinstance(questions, (list, tuple)) or not questions:
        return None
    max_len = get_attribute_max_len() // _DECISION_QUESTIONS_MAX_SHARE - len("[]")
    return _items_entry(list(questions), _DECISION_QUESTIONS_ROLE, max_len)


def _set_tool_attributes(span: Span, kwargs: Dict[str, Any]) -> None:
    """Record ``tools`` and ``tool_choice`` without ever failing the request.

    Runs before the OpenAI call and walks caller-supplied schemas (which may be
    circular or hold objects whose ``model_dump`` raises), so any failure here is
    logged and the attributes are left unset instead of aborting the user's call.
    """
    try:
        _set_tool_definitions(span, kwargs.get("tools"))
    except Exception:  # noqa: BLE001 - tool capture is best-effort; never block the user's request
        logger.warning("netra.instrumentation.libraries.openai: failed to record tool definitions", exc_info=True)

    try:
        _set_tool_choice(span, kwargs.get("tool_choice"))
    except Exception:  # noqa: BLE001 - tool capture is best-effort; never block the user's request
        logger.warning("netra.instrumentation.libraries.openai: failed to record tool_choice", exc_info=True)


def _set_tool_choice(span: Span, tool_choice: Any) -> None:
    """Record ``tool_choice`` as given (string) or as JSON (dict)."""
    # Anything other than str/dict (e.g. the SDK's NOT_GIVEN sentinel) means no choice was made.
    if isinstance(tool_choice, str):
        span.set_attribute(LLM_REQUEST_TOOL_CHOICE, tool_choice)
    elif isinstance(tool_choice, dict):
        span.set_attribute(LLM_REQUEST_TOOL_CHOICE, json.dumps(tool_choice, default=str))


def _set_tool_definitions(span: Span, tools: Any) -> None:
    """Record declared tools as one JSON array under ``gen_ai.request.tools``.

    Each entry is ``{"type", "name", "description", "parameters"}``, with
    ``description``/``parameters`` omitted when absent. Handles both the Chat
    Completions shape (fields nested under ``function`` / ``custom``) and the
    flat Responses API shape. Built-in tools (``web_search_preview``,
    ``file_search``, ``mcp``, ...) have no schema, so their entry is just
    ``{"type"}``. Tools without a ``type`` are skipped; if none remain, the
    attribute is not set.
    """
    if not isinstance(tools, (list, tuple)):
        return

    tool_definitions: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            tool = model_as_dict(tool)
        if (tool_type := tool.get("type")) is None:
            continue

        if tool_type not in _NAMED_TOOL_TYPES:
            tool_definitions.append({"type": tool_type})
            continue

        definition = tool.get(tool_type) or tool
        if not isinstance(definition, dict):
            definition = model_as_dict(definition)
        tool_definition = {"type": tool_type, "name": definition.get("name", "")}
        for field in ("description", "parameters"):
            if (value := definition.get(field)) is not None:
                tool_definition[field] = value
        tool_definitions.append(tool_definition)

    if not tool_definitions:
        return
    # Request attributes are set before the OpenAI call, so a schema value json can't
    # encode must degrade to str rather than raise and abort the user's request.
    span.set_attribute(GEN_AI_REQUEST_TOOLS, json.dumps(tool_definitions, default=str))


def _set_chat_completion_input(span: Span, messages: Any) -> None:
    """Set completion API input attributes"""
    if not isinstance(messages, list) or not messages:
        return

    prompt_index = 0
    for message in messages:
        if not isinstance(message, dict):
            message = model_as_dict(message)
        if not message:
            continue

        role = message.get("role", "user")

        if content := message.get("content"):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", role)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content", str(content))
            if role == "tool" and message.get("tool_call_id"):
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.tool_call_id", message["tool_call_id"])
            prompt_index += 1

        for tc in message.get("tool_calls") or []:
            func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if func is None:
                continue
            name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
            arguments = func.get("arguments", "") if isinstance(func, dict) else getattr(func, "arguments", "")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", "assistant")
            span.set_attribute(
                f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                json.dumps({"name": name, "arguments": arguments}),
            )
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.tool_call_id", tc_id)
            prompt_index += 1


def _set_chat_response_input(span: Span, kwargs: Dict[str, Any]) -> None:
    """Set response API input attributes"""
    message_index = 0

    # Handle instructions as system message
    if instructions := kwargs.get("instructions"):
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "system")
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", instructions)
        message_index += 1

    # Handle input messages
    if input_data := kwargs.get("input"):
        if isinstance(input_data, str):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "user")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", input_data)
        elif isinstance(input_data, list) and input_data:
            for message in input_data:
                if isinstance(message, dict):
                    msg_type = message.get("type", "")
                    if msg_type == "function_call":
                        name = message.get("name", "")
                        arguments = message.get("arguments", "")
                        if name or arguments:
                            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "assistant")
                            span.set_attribute(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content",
                                json.dumps({"name": name, "arguments": arguments}),
                            )
                            message_index += 1
                    elif msg_type == "function_call_output":
                        output_val = str(message.get("output") or "")
                        if output_val:
                            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "tool")
                            span.set_attribute(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content",
                                output_val,
                            )
                            message_index += 1
                    elif msg_type in ("reasoning", "reasoning_summary"):
                        continue
                    else:
                        role = message.get("role", "user")
                        content = message.get("content")
                        if content:
                            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", role)
                            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", str(content))
                            message_index += 1


def set_response_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        logger.debug("Span is not recording")
        return

    if model := response_dict.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_RESPONSE_MODEL}", model)

    if usage := response_dict.get("usage"):
        _set_usage_attributes(span, usage)
        _set_custom_usage_attribute(span, usage, response_dict.get("model") or "")

    _set_response_message_attributes(span, response_dict)
    _set_decision_answer_attributes(span, response_dict)


def _set_decision_answer_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Record a Decisions API response's ``answers``, in question order, as the span's ``output``.

    ``output`` holds a single assistant entry whose ``content`` is the answers
    as a JSON array; each answer is a predicate, choice, score or refusal
    object, as returned. Only as many whole answers as fit the attribute length
    are kept (counting the escaping the nested JSON needs), so the attribute
    stays valid JSON. Nothing is recorded when content tracing is off, or when
    not even one answer fits. Responses without an ``answers`` list (every
    other OpenAI API) are ignored.
    """
    answers = response_dict.get("answers")
    if not isinstance(answers, list) or not should_send_prompts():
        return
    try:
        if entry := _items_entry(answers, _DECISION_ANSWERS_ROLE, get_attribute_max_len() - len("[]")):
            span.set_attribute(OUTPUT_ATTRIBUTE, _join_json_array([entry]))
    except Exception:  # noqa: BLE001 - runs after the call succeeded; never cost the user their response
        logger.warning("netra.instrumentation.libraries.openai: failed to record decision answers", exc_info=True)


def _first_present(usage: Dict[str, Any], *keys: str) -> Any:
    """Return the first of ``keys`` whose value in ``usage`` is not None.

    Used to resolve token fields that differ between OpenAI APIs but mean the
    same thing (e.g. ``prompt_tokens`` in Chat Completions vs ``input_tokens``
    in the Responses API). Keys are checked in the order given, so pass the
    preferred alias first. A present key holding ``0`` is returned as-is; only
    missing or explicitly-None values are skipped.

    Args:
        usage: The usage payload to read from.
        *keys: Candidate keys to try, in priority order.

    Returns:
        The value of the first key present with a non-None value, or None if
        none of the keys are present with a non-None value.
    """
    for key in keys:
        if (value := usage.get(key)) is not None:
            return value
    return None


def _set_usage_attributes(span: Span, usage: Dict[str, Any]) -> None:
    """Set usage/token attributes from an OpenAI usage payload.

    Handles both the Chat Completions shape (``prompt_tokens``/``completion_tokens``
    with ``prompt_tokens_details``) and the Responses API shape
    (``input_tokens``/``output_tokens`` with ``input_tokens_details``). Token
    counts are compared with ``is not None`` rather than truthiness so that a
    legitimate ``0`` (e.g. a cache hit that wrote nothing) is recorded instead of
    silently dropped.
    """
    prompt_tokens = _first_present(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _first_present(usage, "completion_tokens", "output_tokens")

    if prompt_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)

    if completion_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)

    input_tokens_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if input_tokens_details:
        cache_read_tokens = input_tokens_details.get("cached_tokens")
        if cache_read_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens)

        # cache_write_tokens landed in the OpenAI SDK's usage breakdown alongside GPT-5.x
        # prompt caching; map it to the Anthropic-style cache-creation attribute for parity.
        cache_write_tokens = input_tokens_details.get("cache_write_tokens")
        if cache_write_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_write_tokens)

        try:
            _set_cache_write_ttl_attributes(span, input_tokens_details.get("cache_write_tokens_details"))
        except Exception:  # noqa: BLE001 - runs after the call succeeded; never cost the user their response
            logger.warning(
                "netra.instrumentation.libraries.openai: failed to record cache-write TTL tiers", exc_info=True
            )

    output_tokens_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if output_tokens_details:
        reasoning_tokens = output_tokens_details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_REASONING_TOKENS, reasoning_tokens)

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)


def _set_cache_write_ttl_attributes(span: Span, cache_write_details: Any) -> None:
    """Record the per-TTL cache-write token split; a non-dict breakdown is ignored."""
    if not isinstance(cache_write_details, dict):
        return
    for field, attribute in _CACHE_WRITE_TTL_FIELDS.items():
        if (tokens := cache_write_details.get(field)) is not None:
            span.set_attribute(attribute, tokens)


def _is_valid_cost(value: object) -> TypeGuard[float]:
    """Return True for a finite, non-negative int or float (bools are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _resolve_provider_reported_cost(usage: Dict[str, Any]) -> Optional[float]:
    """Resolve the USD cost a provider reported in ``usage.cost`` (e.g. OpenRouter).

    OpenRouter reports ``cost`` in credits, where one credit is one US dollar.
    For BYOK calls (``is_byok`` is True) ``cost`` is only OpenRouter's fee, so
    ``cost_details.upstream_inference_cost`` is added to give the real spend.
    For non-BYOK calls the upstream cost equals ``cost`` and is not added.

    Returns:
        The cost in USD, or None when ``cost`` is absent or invalid, or when a
        BYOK call lacks a valid upstream cost. None means the backend should
        price the span from its token counts instead.
    """
    cost = usage.get("cost")
    if not _is_valid_cost(cost):
        return None

    if usage.get("is_byok") is not True:
        return float(cost)

    cost_details = usage.get("cost_details")
    upstream_cost = cost_details.get("upstream_inference_cost") if isinstance(cost_details, dict) else None
    if not _is_valid_cost(upstream_cost):
        # Recording only OpenRouter's fee would under-report spend; leave pricing to the backend.
        logger.debug("BYOK usage has no valid upstream_inference_cost; skipping provider-reported cost")
        return None
    return float(cost) + float(upstream_cost)


def _set_custom_usage_attribute(span: Span, usage: Dict[str, Any], model: str) -> None:
    """Record a provider-reported cost as a single ``total`` Netra custom usage entry.

    Token counts are not repeated here (``units_used`` is omitted from the JSON)
    because they are already recorded as ``gen_ai.usage.*`` attributes.
    """
    cost_in_usd = _resolve_provider_reported_cost(usage)
    if cost_in_usd is None:
        return

    entry = UsageModel(model=model, usage_type=PROVIDER_REPORTED_USAGE_TYPE, cost_in_usd=cost_in_usd)
    span.set_attribute(CUSTOM_USAGE_ATTRIBUTE, json.dumps([entry.model_dump(exclude={"units_used"})]))


def _set_response_message_attributes(span: Span, response_dict: Dict[str, Any]) -> Any:
    """Helper to set response message attributes."""
    message_index = 0

    if output_text := response_dict.get("output_text"):
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", output_text)
        message_index += 1

    if output := response_dict.get("output"):
        for element in output:
            if element.get("type") == "function_call":
                name = element.get("name", "")
                arguments = element.get("arguments", "")
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                    json.dumps({"name": name, "arguments": arguments}),
                )
                message_index += 1
            elif content := element.get("content"):
                for chunk in content:
                    if text := chunk.get("text"):
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", text)
                        message_index += 1

    if choices := response_dict.get("choices"):
        for choice in choices:
            if message := choice.get("message"):
                if content := message.get("content"):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", message.get("role", "assistant")
                    )
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", content)
                    message_index += 1
                for tc in message.get("tool_calls") or []:
                    func = tc.get("function", {})
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                        json.dumps({"name": func.get("name", ""), "arguments": func.get("arguments", "")}),
                    )
                    if tc_id := tc.get("id"):
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.tool_call_id", tc_id)
                    message_index += 1
            elif delta := choice.get("delta"):
                if content := delta.get("content"):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", delta.get("role", "assistant")
                    )
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", content)
                    message_index += 1
                for tc in delta.get("tool_calls") or []:
                    func = tc.get("function", {})
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                        json.dumps({"name": func.get("name", ""), "arguments": func.get("arguments", "")}),
                    )
                    if tc_id := tc.get("id"):
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.tool_call_id", tc_id)
                    message_index += 1

            if finish_reason := choice.get("finish_reason"):
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.finish_reason", finish_reason)

    return message_index
