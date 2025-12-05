import logging
from typing import Any, Dict

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed."""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def set_request_attributes(span: Any, kwargs: Dict[str, Any]) -> None:
    """Set request attributes (model, parameters, prompts) for Elevenlabs."""

    if not span.is_recording():
        return

    GENERIC_ATTRIBUTES = [
        "callback",
        "extra",
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
        "additional_formats",
    ]

    for key in GENERIC_ATTRIBUTES:
        value = kwargs.get(key)
        if value is not None:
            span.set_attribute(f"gen_ai.request.{key}", value)

    if model_id := kwargs.get("model_id"):
        span.set_attribute("gen_ai.request.model", model_id)

    REQUEST_PARAMS = [
        "text",
        "voice_id" "previous_text",
        "next_text",
    ]

    for param in REQUEST_PARAMS:
        value = kwargs.get(param)
        if text := kwargs.get("text"):
            character_count = len(text)
            span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
            span.set_attribute("gen_ai.prompt.0.role", "Input")
            span.set_attribute("gen_ai.prompt.0.content", str(text))
            continue
        if value:
            span.set_attribute(f"gen_ai.request.{param}", value)

    if cloud_storage_url := kwargs.get("cloud_storage_url"):
        logger.info(f"csu {cloud_storage_url}")
        span.set_attribute("gen_ai.request.file_type", cloud_storage_url)
    else:
        span.set_attribute("gen_ai.request.file_type", "file")

    if webhook := kwargs.get("webhook"):
        logger.info(f"Webhook is set to {webhook}")
        span.set_attribute("gen_ai.request.webhook_id", kwargs.get("webhook_id"))

    if use_multi_channels := kwargs.get("use_multi_channel"):
        span.set_attribute("gen_ai.use_multi_channel", use_multi_channels)

    if prompt := kwargs.get("prompt"):
        character_count = len(prompt)
        span.set_attribute("gen_ai.prompt.0.role", "Input")
        span.set_attribute("gen_ai.prompt.0.content", str(prompt))
        span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
    elif composition_plan := kwargs.get("composition_plan"):
        span.set_attribute("gen_ai.prompt.0.role", "Input")
        character_count = 0
        prompt = ""
        for section in composition_plan:
            for line in section.lines:
                character_count += len(line)
                prompt += str(section.lines) + "\n"
        span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
        span.set_attribute("gen_ai.prompt.0.content", str(prompt))

    if music_length_ms := kwargs.get("music_length_ms"):
        span.set_attribute("gen_ai.music_length_ms", music_length_ms)

    if loop := kwargs.get("loop"):
        span.set_attribute("gen_ai.loop", loop)

    character_count = 0
    prompt = ""
    if inputs := kwargs.get("inputs"):
        for line in inputs:
            logger.info(f"line : {line["text"]}")
            character_count += len(line["text"])
            prompt += str(line["text"]) + "\n"
        logger.info(f"prompt: {prompt}")
        span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
        span.set_attribute("gen_ai.prompt.0.content", str(prompt))
        span.set_attribute("gen_ai.prompt.0.role", "Input")

    logger.info(f"Span after setting request attributes: {span}")


def set_response_attributes(span: Span, response: Any) -> None:
    """
    Set the response attributes for the span.

    Args:
        span: The span to set the attributes on.
        response: The response to extract the attributes from.
    """
    if not span.is_recording():
        return

    logger.info(f"Response : {response}")
    try:
        if hasattr(response, "__dict__"):
            span.set_attribute("gen_ai.response.type", "object")

            if hasattr(response, "text") and response.text:
                span.set_attribute("gen_ai.prompt.0.content", response.text)
                span.set_attribute("gen_ai.prompt.0.role", "Output")
                span.set_attribute("gen_ai.usage.prompt.character_count", len(response.text))

            if hasattr(response, "words") and response.words:
                span.set_attribute("gen_ai.response.duration", response.words[-1].end)

            if hasattr(response, "transcription_id") and response.transcription_id:
                span.set_attribute("gen_ai.response.transcription_id", response.transcription_id)

            duration_seconds = None

            # Case 1: responses with .words timing
            if hasattr(response, "words") and response.words:
                last_word = response.words[-1]
                if hasattr(last_word, "end") and last_word.end:
                    duration_seconds = last_word.end

            # Case 2: responses with character_end_times_seconds
            elif hasattr(response, "character_end_times_seconds") and response.character_end_times_seconds:
                duration_seconds = response.character_end_times_seconds[-1]

            elif hasattr(response, "normalized_alignment") and hasattr(
                response.normalized_alignment, "character_end_times_seconds"
            ):
                if response.normalized_alignment.character_end_times_seconds:
                    duration_seconds = response.normalized_alignment.character_end_times_seconds[-1]

            # Case 3: responses with voice_segments
            elif hasattr(response, "voice_segments") and response.voice_segments:
                last_segment = response.voice_segments[-1]
                if hasattr(last_segment, "end_time_seconds"):
                    duration_seconds = last_segment.end_time_seconds

            if duration_seconds is not None:
                span.set_attribute("gen_ai.audio.duration", duration_seconds)
            return

    except Exception as e:
        logger.error(f"Error setting response attributes for Elevenlabs. {e}")
        return
