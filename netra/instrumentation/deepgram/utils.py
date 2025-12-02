import logging
from typing import Any, Dict

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed."""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def set_request_attributes(span: Span, kwargs: Dict[str, Any], source_type: str) -> None:
    """
    Set the request attributes for the span.

    Args:
        span: The span to set the attributes on.
        kwargs: The keyword arguments to extract the attributes from.
        source_type: The type of the source (e.g. "url" or "file").
    """
    if not span.is_recording():
        return

    span.set_attribute("gen_ai.request.source_type", source_type)

    GENERIC_ATTRIBUTES = [
        "callback",
        "extra",
        "model",
        "language",
        "encoding",
        "multichannel",
        "diarize",
        "detect_language",
        "detect_entities",
        "sentiment",
        "summarize",
        "topics",
        "intents",
        "tag",
        "custom_topic",
        "custom_intent",
    ]

    for key in GENERIC_ATTRIBUTES:
        value = kwargs.get(key)
        if value is not None:
            span.set_attribute(f"gen_ai.request.{key}", value)

    if source_type == "url" and "url" in kwargs:
        span.set_attribute("gen_ai.request.url", str(kwargs["url"]))

    if source_type == "file" and "request" in kwargs:
        request_value = kwargs["request"]
        try:
            if isinstance(request_value, (bytes, bytearray)):
                span.set_attribute("gen_ai.request.size_bytes", len(request_value))
        except Exception as e:
            logger.debug("Failed to set Deepgram request size: %s", e)


def set_response_attributes(span: Span, response: Any) -> None:
    """
    Set the response attributes for the span.

    Args:
        span: The span to set the attributes on.
        response: The response to extract the attributes from.
    """
    if not span.is_recording():
        return

    try:
        if results := getattr(response, "results", None):
            if channels := getattr(results, "channels", None):
                first_channel = next(iter(channels))
                if alternatives := getattr(first_channel, "alternatives", None):
                    first_alt = next(iter(alternatives))
                    if transcript := getattr(first_alt, "transcript", None):
                        span.set_attribute(f"gen_ai.completion.0.role", "Transcribed Text")
                        span.set_attribute(f"gen_ai.completion.0.content", transcript)

        if metadata := getattr(response, "metadata", None):
            if request_id := getattr(metadata, "request_id", None):
                span.set_attribute("gen_ai.response.request_id", str(request_id))

            if duration := getattr(metadata, "duration", None):
                span.set_attribute("gen_ai.audio.duration", str(duration))

            if channel_count := getattr(metadata, "channels", None):
                span.set_attribute("gen_ai.response.channels", channel_count)

            if models := getattr(metadata, "models", None):
                span.set_attribute(
                    "gen_ai.response.model_list", list(models) if not isinstance(models, str) else [models]
                )

            if model_info := getattr(metadata, "model_info", None):
                if isinstance(model_info, dict) and model_info:
                    first_model_key = next(iter(model_info))
                    info = model_info.get(first_model_key, {})
                if name := info.get("name", None):
                    span.set_attribute("gen_ai.request.model", str(name))
                if version := info.get("version", None):
                    span.set_attribute("gen_ai.request.model_version", str(version))
                if arch := info.get("arch", None):
                    span.set_attribute("gen_ai.request.model_arch", str(arch))

    except Exception as e:
        logger.debug("Failed to set Deepgram response attributes: %s", e)
