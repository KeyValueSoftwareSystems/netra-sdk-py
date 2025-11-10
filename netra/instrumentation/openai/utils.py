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


def _set_response_message_attributes(span: Span, response_dict: Dict[str, Any]) -> Any:
    """Helper to set response message attributes."""
    message_index = 0

    if output_text := response_dict.get("output_text"):
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", output_text)
        message_index += 1

    if output := response_dict.get("output"):
        for element in output:
            if content := element.get("content"):
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
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", message.get("content", "")
                )
                message_index += 1
            elif delta := choice.get("delta"):
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", delta.get("role", "assistant")
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", delta.get("content", "")
                )
                message_index += 1

            if finish_reason := choice.get("finish_reason"):
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.finish_reason", finish_reason)

    return message_index
