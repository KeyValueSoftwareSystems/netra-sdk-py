import logging
from typing import Any

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_ID
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Span

from netra.instrumentation.message_builder import (
    build_chat_input,
    build_completion_output,
    build_messages,
    model_as_dict,
)

logger = logging.getLogger(__name__)

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_CEREBRAS = "cerebras"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed via OTel context keys."""
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def set_request_attributes(span: Span, llm_request_type: LLMRequestTypeValues, kwargs: dict[Any, Any]) -> None:
    """Set request attributes on the span based on Cerebras request kwargs."""
    if not span.is_recording():
        return

    if model := kwargs.get("model"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    if max_tokens := kwargs.get("max_tokens"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)
    if temperature := kwargs.get("temperature"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, temperature)
    if top_p := kwargs.get("top_p"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_TOP_P, top_p)
    if frequency_penalty := kwargs.get("frequency_penalty"):
        span.set_attribute(SpanAttributes.LLM_FREQUENCY_PENALTY, frequency_penalty)
    if presence_penalty := kwargs.get("presence_penalty"):
        span.set_attribute(SpanAttributes.LLM_PRESENCE_PENALTY, presence_penalty)

    if llm_request_type == LLMRequestTypeValues.COMPLETION:
        if prompt := kwargs.get("prompt"):
            span.set_attribute("input", build_messages([{"role": "user", "content": prompt}]))
    elif llm_request_type == LLMRequestTypeValues.CHAT:
        if messages := kwargs.get("messages"):
            if isinstance(messages, list):
                span.set_attribute("input", build_chat_input(messages))


def set_response_attributes(span: Span, response: Any) -> None:
    """Set response-related attributes, always capturing usage and cached tokens."""
    if not span.is_recording():
        return

    response_dict = model_as_dict(response)

    try:
        usage = response_dict.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens"):
                span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))
            if usage.get("completion_tokens"):
                span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens"))
            if usage.get("total_tokens"):
                span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
            if usage.get("prompt_tokens_details"):
                details = usage.get("prompt_tokens_details")
                if details and hasattr(details, "cached_tokens"):
                    span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, details.cached_tokens)
                elif isinstance(details, dict) and "cached_tokens" in details:
                    span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, details["cached_tokens"])

        if response_dict.get("id"):
            span.set_attribute(GEN_AI_RESPONSE_ID, response_dict.get("id"))

        span.set_attribute("output", build_completion_output(response_dict))

    except Exception:
        logger.exception("Failed to set response attributes")
