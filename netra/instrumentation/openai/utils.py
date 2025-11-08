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

    span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TYPE}", operation_type)

    if kwargs.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MODEL}", kwargs.get("model"))

    if kwargs.get("temperature"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", kwargs.get("temperature"))

    if kwargs.get("max_tokens"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs.get("max_tokens"))

    if kwargs.get("max_completion_tokens"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs.get("max_completion_tokens"))

    if kwargs.get("frequency_penalty"):
        span.set_attribute(f"{SpanAttributes.LLM_FREQUENCY_PENALTY}", kwargs.get("frequency_penalty"))

    if kwargs.get("presence_penalty"):
        span.set_attribute(f"{SpanAttributes.LLM_PRESENCE_PENALTY}", kwargs.get("presence_penalty"))

    if kwargs.get("reasoning_effort"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_REASONING_EFFORT}", kwargs.get("reasoning_effort"))

    if kwargs.get("stop"):
        span.set_attribute(f"{SpanAttributes.LLM_CHAT_STOP_SEQUENCES}", kwargs.get("stop"))

    if kwargs.get("stream"):
        span.set_attribute(f"{SpanAttributes.LLM_IS_STREAMING}", kwargs.get("stream"))

    if kwargs.get("top_p"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TOP_P}", kwargs.get("top_p"))

    if kwargs.get("max_output_tokens"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs.get("max_output_tokens"))

    if kwargs.get("reasoning"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_REASONING_EFFORT}", json.dumps(kwargs.get("reasoning")))

    if kwargs.get("dimensions"):
        span.set_attribute(f"gen_ai.request.dimensions", kwargs.get("dimensions"))

    # Message - Chat Completion API
    if operation_type == "chat" and kwargs.get("messages"):
        messages = kwargs["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            for index, message in enumerate(messages):
                if isinstance(message, dict):
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", message.get("role", "user"))
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(message.get("content", "")))

    # Message - Response API
    if operation_type == "response" or operation_type == "embedding":
        message_index = 0
        if kwargs.get("instructions"):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "system")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", kwargs["instructions"])
            message_index += 1
        if kwargs.get("input"):
            if isinstance(kwargs["input"], list) and len(kwargs["input"]) > 0:
                for message in kwargs["input"]:
                    if isinstance(message, dict):
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", message.get("role", "user")
                        )
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", str(message.get("content", ""))
                        )
                        message_index += 1
            elif isinstance(kwargs["input"], str):
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "user")
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", kwargs["input"])


def set_response_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        logger.debug("Span is not recording")
        return

    if response_dict.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_RESPONSE_MODEL}", response_dict["model"])

    if response_dict.get("usage"):
        usage = response_dict.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            completion_tokens_details = (
                usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
            )
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
            prompt_tokens_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            cache_read_input_tokens = prompt_tokens_details.get("cached_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

        if prompt_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", prompt_tokens)

        if completion_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", completion_tokens)

        if cache_read_input_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", cache_read_input_tokens)

        if total_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", total_tokens)

        if reasoning_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_REASONING_TOKENS}", reasoning_tokens)

    # Response API
    message_index = 0
    if response_dict.get("output_text"):
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
        span.set_attribute(
            f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", response_dict.get("output_text", "")
        )
        message_index += 1

    if response_dict.get("output"):
        output = response_dict.get("output", [])
        for element in output:
            content = element.get("content", [])
            if content:
                for chunk in content:
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant")
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", chunk.get("text", "")
                    )
                    message_index += 1

    # Chat Completion API
    choices = response_dict.get("choices", [])
    if choices:
        for choice in choices:
            if choice.get("message"):
                message = choice["message"]
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", message.get("role", "assistant")
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", message.get("content", "")
                )
                message_index += 1

            elif choice.get("delta"):
                chunk = choice.get("delta")
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", chunk.get("role", "assistant")
                )
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", chunk.get("content", "")
                )
                message_index += 1

            if choice.get("finish_reason"):
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.finish_reason", choice.get("finish_reason", "")
                )
