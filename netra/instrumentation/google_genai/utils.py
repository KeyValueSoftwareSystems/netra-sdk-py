from typing import Any, Dict, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed for GenAI."""
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def set_request_attributes(span: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Set request attributes (model, parameters, prompts) for Google GenAI."""
    if not span.is_recording():
        return

    if model := kwargs.get("model"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)

    if config := kwargs.get("config"):
        config_mappings = [
            ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
            ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
            ("top_k", SpanAttributes.LLM_TOP_K),
            ("max_output_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
            ("stop_sequences", SpanAttributes.LLM_CHAT_STOP_SEQUENCES),
            ("presence_penalty", SpanAttributes.LLM_PRESENCE_PENALTY),
            ("frequency_penalty", SpanAttributes.LLM_FREQUENCY_PENALTY),
        ]

    for attr_name, span_attr in config_mappings:
        if (value := getattr(config, attr_name, None)) is not None:
            span.set_attribute(span_attr, value)

    if contents := kwargs.get("contents"):
        if isinstance(contents, str):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", contents)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            return

        content_list = contents if isinstance(contents, list) else [contents]

        for index, content in enumerate(content_list):
            role = getattr(content, "role", "user")
            text = None

            if hasattr(content, "parts"):
                text = next((part.text for part in content.parts if hasattr(part, "text") and part.text), None)
            elif hasattr(content, "text"):
                text = content.text
                role = "user"
            elif hasattr(content, "function_call"):
                fc = content.function_call
                text = f"Function Call: {fc.name}, Args: {fc.args}"
                role = "model"
            else:
                text = str(content)
                role = "user"

            if text:
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", text)
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)


def set_response_attributes(span: Any, response: Any) -> None:
    """Set response attributes for Google GenAI."""
    if not span.is_recording():
        return

    if isinstance(response, str):
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response)
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        return

    usage = _extract_usage_metadata(response)
    if usage is not None:
        if (total := getattr(usage, "total_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)

        output = 0
        if isinstance(ct := getattr(usage, "candidates_token_count", None), int):
            output += ct
        if isinstance(tt := getattr(usage, "thoughts_token_count", None), int):
            output += tt
        if output > 0:
            span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output)

        if (cached := getattr(usage, "cached_content_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)

        if (prompt := getattr(usage, "prompt_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt)

        if isinstance(response, dict):
            if isinstance(text := response.get("content"), str) and text:
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", text)
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
                return

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list):
            for index, candidate in enumerate(candidates):
                content = getattr(candidate, "content", None)
                if content is None:
                    continue

                parts = getattr(content, "parts", None)
                if not isinstance(parts, list):
                    continue

                for part in parts:
                    if isinstance(text := getattr(part, "text", None), str):
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", text)
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", "assistant")
                        break


def _extract_usage_metadata(response: Any) -> Any:
    """Extract usage metadata from response."""
    if hasattr(response, "usage_metadata"):
        return response.usage_metadata
    if hasattr(response, "get"):
        chunk = response.get("chunk", {})
        return getattr(chunk, "usage_metadata", None)
    return None


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
