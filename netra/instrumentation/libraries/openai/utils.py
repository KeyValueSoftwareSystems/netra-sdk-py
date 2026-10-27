import json
import logging
import math
from typing import Any, Dict, List, Optional, TypeGuard

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace import Span

from netra.config import Config
from netra.span_wrapper import ATTRIBUTE, UsageModel

logger = logging.getLogger(__name__)

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


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


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
        if (value := kwargs.get(key)) is not None:
            span.set_attribute(attribute, value)

    if (reasoning := kwargs.get("reasoning")) is not None:
        span.set_attribute(SpanAttributes.LLM_REQUEST_REASONING_EFFORT, json.dumps(reasoning))

    if operation_type == "chat":
        _set_chat_completion_input(span, kwargs.get("messages"))
    elif operation_type == "response":
        _set_chat_response_input(span, kwargs)

    _set_tool_attributes(span, kwargs)


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
