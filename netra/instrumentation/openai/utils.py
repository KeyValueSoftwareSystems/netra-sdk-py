import json
import logging
from typing import Any, Dict

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


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


def _set_chat_completion_input(span: Span, messages: Any) -> None:
    """Set completion API input attributes"""
    if not isinstance(messages, list) or not messages:
        return

    for index, message in enumerate(messages):
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", content)


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
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "assistant")
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content",
                            json.dumps({"name": name, "arguments": arguments}),
                        )
                    elif msg_type == "function_call_output":
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "tool")
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content",
                            str(message.get("output", "")),
                        )
                    else:
                        role = message.get("role", "user")
                        content = str(message.get("content", ""))
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", role)
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", content)
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

    output_tokens_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if output_tokens_details:
        reasoning_tokens = output_tokens_details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_REASONING_TOKENS, reasoning_tokens)

    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)


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
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", message.get("role", "assistant")
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", message.get("content") or ""
                )
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
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", delta.get("role", "assistant")
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", delta.get("content") or ""
                )
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
