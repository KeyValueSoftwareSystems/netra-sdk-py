import json
import logging
from typing import Any, Dict

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span

from netra.instrumentation.message_builder import (
    build_chat_input,
    build_completion_output,
    build_response_api_input,
)

logger = logging.getLogger(__name__)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


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
    """Set structured input from Chat Completions messages."""
    if not isinstance(messages, list) or not messages:
        return
    span.set_attribute("input", build_chat_input(messages))


def _set_chat_response_input(span: Span, kwargs: Dict[str, Any]) -> None:
    """Set structured input from Responses API kwargs."""
    span.set_attribute("input", build_response_api_input(kwargs))


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


def _set_usage_attributes(span: Span, usage: Dict[str, Any]) -> None:
    """Helper to set usage-related attributes"""
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    if prompt_tokens:
        span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", prompt_tokens)

    if completion_tokens:
        span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", completion_tokens)

    if prompt_tokens_details := (usage.get("prompt_tokens_details") or usage.get("input_tokens_details")):
        if cache_tokens := prompt_tokens_details.get("cached_tokens"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", cache_tokens)

    if completion_tokens_details := (usage.get("completion_tokens_details") or usage.get("output_tokens_details")):
        if reasoning_tokens := completion_tokens_details.get("reasoning_tokens"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_REASONING_TOKENS}", reasoning_tokens)

    if total_tokens := usage.get("total_tokens"):
        span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", total_tokens)


def _set_response_message_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set structured output from a response dict."""
    span.set_attribute("output", build_completion_output(response_dict))
