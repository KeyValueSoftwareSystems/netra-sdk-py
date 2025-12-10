import logging
import time
from typing import Any, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.elevenlabs.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

CREATE_SPEECH_SPAN_NAME = "elevenlabs.create_speech"
CREATE_SPEECH_WITH_TIMESTAMPS_SPAN_NAME = "elevenlabs.create_speech_with_timestamp"
CREATE_SPEECH_STREAM_SPAN_NAME = "elevenlabs.create_speech_stream"
CREATE_SPEECH_STREAM_WITH_TIMESTAMPS_SPAN_NAME = "elevenlabs.create_speech_stream_with_timestamps"
CREATE_TRANSCRIPT_SPAN_NAME = "elevenlabs.create_transcript"
CREATE_DIALOGUE_SPAN_NAME = "elevenlabs.create_dialogue"
CREATE_DIALOGUE_STREAM_SPAN_NAME = "elevenlabs.create_dialogue_stream"
CREATE_DIALOGUE_WITH_TIMESTAMPS_SPAN_NAME = "elevenlabs.create_dialogue_with_timestamps"
CREATE_DIALOGUE_STREAM_WITH_TIMESTAMPS_SPAN_NAME = "elevenlabs.create_dialogue_stream_with_timestamps"
CREATE_MUSIC_SPAN_NAME = "elevenlabs.create_music"
CREATE_MUSIC_WITH_STREAM_SPAN_NAME = "elevenlabs.create_music_stream"
VOICE_CHANGER_SPAN_NAME = "elevenlabs.voice_changer"
VOICE_CHANGER_STREAM_SPAN_NAME = "elevenlabs.voice_changer_stream"
CREATE_SOUND_EFFECT_SPAN_NAME = "elevenlabs.create_sound_effect"
DESIGN_VOICE_SPAN_NAME = "elevenlabs.design_voice"


def _wrap_speech(tracer: Tracer, span_name: str, request_type: str) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Elevenlabs", "gen_ai.request.type": request_type},
        ) as span:
            try:
                set_request_attributes(span, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("elevenlabs.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.elevenlabs: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def stream_dialogue_wrapper(tracer: Tracer, span_name: str, request_type: str) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "ElevenLabs", "gen_ai.request.type": request_type},
        )

        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, kwargs)
            start_time = time.time()
            response = wrapped(*args, **kwargs)  # generator
            return ElevenLabsStreamingWrapper(span=span, response=response, start_time=start_time)

        except Exception as e:
            logger.error("netra.instrumentation.elevenlabs: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(context)

    return wrapper


def create_speech_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SPEECH_SPAN_NAME, "speech")


def create_speech_with_timestamp_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SPEECH_WITH_TIMESTAMPS_SPAN_NAME, "speech")


def create_speech_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the stream method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SPEECH_STREAM_SPAN_NAME, "speech")


def create_speech_stream_with_timestamp_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the stream_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SPEECH_STREAM_WITH_TIMESTAMPS_SPAN_NAME, "speech")


def create_transcript_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_TRANSCRIPT_SPAN_NAME, "transcript")


def create_dialogue_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_DIALOGUE_SPAN_NAME, "dialogue")


def create_dialogue_with_timestamps_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_DIALOGUE_WITH_TIMESTAMPS_SPAN_NAME, "dialogue")


def create_dialogue_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_DIALOGUE_STREAM_SPAN_NAME, "dialogue")


def create_dialogue_stream_with_timestamps_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return stream_dialogue_wrapper(tracer, CREATE_DIALOGUE_STREAM_WITH_TIMESTAMPS_SPAN_NAME, "dialogue")


def voice_changer_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, VOICE_CHANGER_SPAN_NAME, "voice_changer")


def voice_changer_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the stream method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, VOICE_CHANGER_STREAM_SPAN_NAME, "voice_changer")


def create_sound_effect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SOUND_EFFECT_SPAN_NAME, "sound_effect")


class ElevenLabsStreamingWrapper:
    def __init__(self, span: Span, response: Iterator[Any], start_time: float) -> None:
        self._span: Span = span
        self._start_time: float = start_time
        self._response: Iterator[Any] = response

        self._buffer: Dict[str, Any] = {"chunks": [], "duration": 0.0, "alignment": [], "voice_segments": []}

    def __iter__(self) -> "ElevenLabsStreamingWrapper":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._response)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize_span()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        if hasattr(chunk, "alignment") and chunk.alignment:
            self._span.add_event("elevenlabs.stream.alignment")
            self._buffer["alignment"].append(chunk.alignment)

        if hasattr(chunk, "voice_segments") and chunk.voice_segments:
            self._span.add_event("elevenlabs.stream.voice_segments")
            self._buffer["voice_segments"].extend(chunk.voice_segments)

        if hasattr(chunk, "alignment") and getattr(chunk.alignment, "character_end_times_seconds", None):
            self._buffer["duration"] = (chunk.alignment.character_end_times_seconds[-1]) / 60

        self._buffer["chunks"].append(chunk)

    def _finalize_span(self) -> None:
        end_time = time.time()
        self._span.set_attribute("gen_ai.response.duration", end_time - self._start_time)

        if self._buffer["duration"]:
            self._span.set_attribute("gen_ai.audio.duration", self._buffer["duration"])

        set_response_attributes(self._span, self._buffer)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
