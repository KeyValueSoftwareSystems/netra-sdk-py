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

    if model_id := kwargs.get("model_id"):
        span.set_attribute("gen_ai.request.model", model_id)

    if transcript := kwargs.get("transcript"):
        character_count = len(transcript)
        span.set_attribute("gen_ai.usage.prompt.character_count", character_count)
        span.set_attribute("gen_ai.prompt.0.role", "Input")
        span.set_attribute("gen_ai.prompt.0.content", str(transcript))

    if voice := kwargs.get("voice"):
        if isinstance(voice, dict):
            if mode := voice.get("mode"):
                span.set_attribute("gen_ai.request.voice.mode", mode)
            if voice_id := voice.get("id"):
                span.set_attribute("gen_ai.request.voice_id", voice_id)
    elif voice_id := kwargs.get("voice_id"):
        span.set_attribute("gen_ai.request.voice_id", voice_id)

    if output_format := kwargs.get("output_format"):
        if isinstance(output_format, dict):
            if container := output_format.get("container"):
                span.set_attribute("gen_ai.request.output_format.container", container)
            if encoding := output_format.get("encoding"):
                span.set_attribute("gen_ai.request.output_format.encoding", encoding)
            if sample_rate := output_format.get("sample_rate"):
                span.set_attribute("gen_ai.request.output_format.sample_rate", sample_rate)

    if language := kwargs.get("language"):
        span.set_attribute("gen_ai.request.language", language)

    if language := kwargs.get("language"):
        span.set_attribute("gen_ai.request.language", language)

    if voice := kwargs.get("voice"):
        span.set_attribute("gen_ai.request.voice_id", voice["id"])
    elif voice := kwargs.get("voice_id"):
        span.set_attribute("gen_ai.request.voice_id", voice["id"])


def set_response_attributes(span: Span, response: Any) -> None:
    """Set the response attributes for both dict & object response formats."""

    if not span.is_recording():
        return

    logger.info(f"Response : {response}")

    try:
        if isinstance(response, dict):
            span.set_attribute("gen_ai.response.type", response.get("type", "unknown"))
            span.set_attribute("gen_ai.prompt.1.role", "Output")
            if "duration" in response:
                span.set_attribute("gen_ai.audio.duration", response["duration"])

            return

        if hasattr(response, "__dict__"):
            span.set_attribute("gen_ai.response.type", "object")

            if hasattr(response, "duration") and response.duration:
                span.set_attribute("gen_ai.audio.duration", response.duration)

            if hasattr(response, "text") and response.text:
                span.set_attribute("gen_ai.response.text", response.text)
                span.set_attribute("gen_ai.prompt.1.content", response.text)
                span.set_attribute("gen_ai.prompt.1.role", "Output")

            return

    except Exception as e:
        logger.error(f"Error setting response attributes for Cartesia STT: {e}")
