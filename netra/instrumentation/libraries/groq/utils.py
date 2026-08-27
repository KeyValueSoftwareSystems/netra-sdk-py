import logging
from typing import Any, Dict

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, SpanAttributes
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed (OpenAI-style)."""
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def model_as_dict(input_object: Any) -> Any:
    """Convert SDK model object to a plain dict."""
    if hasattr(input_object, "model_dump"):
        return input_object.model_dump()
    if hasattr(input_object, "to_dict"):
        return input_object.to_dict()
    if isinstance(input_object, dict):
        return input_object
    return {}


def set_request_attributes(span: Span, kwargs: Dict[str, Any], operation_type: str) -> None:
    """Set request attributes for Groq chat completions."""
    if not span.is_recording():
        return

    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, operation_type)

    ATTRIBUTE_MAPPINGS = {
        "model": SpanAttributes.LLM_REQUEST_MODEL,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_completion_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_tokens_to_sample": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "reasoning_effort": SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
        "frequency_penalty": SpanAttributes.LLM_FREQUENCY_PENALTY,
        "presence_penalty": SpanAttributes.LLM_PRESENCE_PENALTY,
        "stop": SpanAttributes.LLM_CHAT_STOP_SEQUENCES,
        "stream": SpanAttributes.LLM_IS_STREAMING,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    }

    for key, attribute in ATTRIBUTE_MAPPINGS.items():
        if (value := kwargs.get(key)) is not None:
            span.set_attribute(attribute, value)

    _set_chat_input(span, kwargs.get("messages"), kwargs.get("prompt"))


def _set_chat_input(span: Span, messages: Any, prompt: Any) -> None:
    if isinstance(messages, list) and messages:
        for index, message in enumerate(messages):
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(content))
            elif hasattr(message, "role") and hasattr(message, "content"):
                role = message.role
                content = message.content
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(content))
        return

    if isinstance(prompt, str) and prompt:
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt)


def set_response_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set response attributes for Groq chat completions."""
    if not span.is_recording():
        return

    if model := response_dict.get("model"):
        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if usage := response_dict.get("usage"):
        _set_usage_attributes(span, usage)

    _set_response_message_attributes(span, response_dict)


def _set_usage_attributes(span: Span, usage: Dict[str, Any]) -> None:
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")

    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    if prompt_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)

    if prompt_details := (usage.get("prompt_tokens_details") or usage.get("input_tokens_details")):
        if cached := prompt_details.get("cached_tokens"):
            span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)

    if total := usage.get("total_tokens"):
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)


def _set_response_message_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    message_index = 0

    # Completion API-like
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
