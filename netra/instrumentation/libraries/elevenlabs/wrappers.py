import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.elevenlabs.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.span_utils import record_span_timing

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
                status_code = getattr(response, "status_code", None)
                if status_code and status_code != 200:
                    error_message = getattr(response, "body", "Unknown error")
                    raise Exception(f"ElevenLabs API Error: {error_message}")
                else:
                    span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
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
            attributes={
                "gen_ai.system": "ElevenLabs",
                "gen_ai.request.type": request_type,
            },
        )

        # The span is current only while the request is issued; each chunk is
        # pulled under it again by the stream wrapper. Keeping it attached
        # until the stream ends would leak it into the caller's code.
        token = context_api.attach(set_span_in_context(span))
        try:
            set_request_attributes(span, kwargs)
            response = wrapped(*args, **kwargs)  # generator
        except Exception as e:
            logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(token)

        return ElevenLabsStreamingWrapper(span=span, response=response)

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
    return stream_dialogue_wrapper(tracer, CREATE_SPEECH_STREAM_SPAN_NAME, "speech")


def create_speech_stream_with_timestamp_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the stream_with_timestamps method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return stream_dialogue_wrapper(tracer, CREATE_SPEECH_STREAM_WITH_TIMESTAMPS_SPAN_NAME, "speech")


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
    return stream_dialogue_wrapper(tracer, CREATE_DIALOGUE_STREAM_SPAN_NAME, "dialogue")


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
    return stream_dialogue_wrapper(tracer, VOICE_CHANGER_STREAM_SPAN_NAME, "voice_changer")


def create_sound_effect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the convert method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_speech(tracer, CREATE_SOUND_EFFECT_SPAN_NAME, "sound_effect")


def _wrap_speech_async(tracer: Tracer, span_name: str, request_type: str) -> Callable[..., Any]:
    """Async version of _wrap_speech for non-streaming methods."""

    async def async_wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Elevenlabs", "gen_ai.request.type": request_type},
        ) as span:
            try:
                set_request_attributes(span, kwargs)
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("elevenlabs.response.duration", end_time - start_time)
                status_code = getattr(response, "status_code", None)
                if status_code and status_code != 200:
                    error_message = getattr(response, "body", "Unknown error")
                    raise Exception(f"ElevenLabs API Error: {error_message}")
                else:
                    span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return async_wrapper


def async_generator_wrapper(tracer: Tracer, span_name: str, request_type: str) -> Callable[..., Any]:
    """Wrapper for methods that return async generators directly (not async context managers)."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "ElevenLabs", "gen_ai.request.type": request_type},
        )

        token = context_api.attach(set_span_in_context(span))
        try:
            set_request_attributes(span, kwargs)
            response = wrapped(*args, **kwargs)  # Returns async generator - NO await!
        except Exception as e:
            logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(token)

        return ElevenLabsAsyncStreamingWrapper(span=span, response=response)

    return wrapper


def create_speech_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for convert method."""
    return async_generator_wrapper(tracer, CREATE_SPEECH_SPAN_NAME, "speech")


def create_speech_with_timestamp_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for convert_with_timestamps method."""
    return _wrap_speech_async(tracer, CREATE_SPEECH_WITH_TIMESTAMPS_SPAN_NAME, "speech")


def create_speech_stream_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for stream method."""
    return async_generator_wrapper(tracer, CREATE_SPEECH_STREAM_SPAN_NAME, "speech")


def create_speech_stream_with_timestamp_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for stream_with_timestamps method."""
    return async_generator_wrapper(tracer, CREATE_SPEECH_STREAM_WITH_TIMESTAMPS_SPAN_NAME, "speech")


def create_transcript_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for transcript method."""
    return _wrap_speech_async(tracer, CREATE_TRANSCRIPT_SPAN_NAME, "transcript")


def create_dialogue_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for dialogue method."""
    return async_generator_wrapper(tracer, CREATE_DIALOGUE_SPAN_NAME, "dialogue")


def create_dialogue_with_timestamps_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for dialogue_with_timestamps method."""
    return _wrap_speech_async(tracer, CREATE_DIALOGUE_WITH_TIMESTAMPS_SPAN_NAME, "dialogue")


def create_dialogue_stream_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for dialogue_stream method."""
    return async_generator_wrapper(tracer, CREATE_DIALOGUE_STREAM_SPAN_NAME, "dialogue")


def create_dialogue_stream_with_timestamps_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for dialogue_stream_with_timestamps method."""
    return async_generator_wrapper(tracer, CREATE_DIALOGUE_STREAM_WITH_TIMESTAMPS_SPAN_NAME, "dialogue")


def voice_changer_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for voice_changer method."""
    return async_generator_wrapper(tracer, VOICE_CHANGER_SPAN_NAME, "voice_changer")


def voice_changer_stream_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for voice_changer_stream method."""
    return async_generator_wrapper(tracer, VOICE_CHANGER_STREAM_SPAN_NAME, "voice_changer")


def create_sound_effect_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for sound_effect method."""
    return async_generator_wrapper(tracer, CREATE_SOUND_EFFECT_SPAN_NAME, "sound_effect")


class _StreamSpanState:
    """Span bookkeeping shared by the sync and async stream wrappers.

    The span is made current only while a chunk is being pulled (so spans the
    SDK creates during the request are its children) and is ended exactly once:
    on exhaustion, on error, on ``close()``/``aclose()``, or on garbage collection.
    """

    def __init__(self, span: Span) -> None:
        self._span = span
        self._finalized = False
        self._buffer: Dict[str, Any] = {"duration": 0.0, "alignment": [], "voice_segments": []}

    def _process_chunk(self, chunk: Any) -> None:
        try:
            if hasattr(chunk, "alignment") and chunk.alignment:
                self._span.add_event("elevenlabs.stream.alignment")
                self._buffer["alignment"].append(chunk.alignment)

            if hasattr(chunk, "voice_segments") and chunk.voice_segments:
                self._span.add_event("elevenlabs.stream.voice_segments")
                self._buffer["voice_segments"].extend(chunk.voice_segments)

            if hasattr(chunk, "alignment") and getattr(chunk.alignment, "character_end_times_seconds", None):
                self._buffer["duration"] = (chunk.alignment.character_end_times_seconds[-1]) / 60
        except Exception:
            logger.debug("netra.instrumentation.libraries.elevenlabs: failed to process chunk", exc_info=True)

    def _finalize(self, error: Optional[BaseException] = None, completed: bool = True) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            record_span_timing(self._span, "gen_ai.response.duration")
            if error is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
                self._span.record_exception(error)
            else:
                if self._buffer["duration"]:
                    self._span.set_attribute("gen_ai.audio.duration", self._buffer["duration"])
                set_response_attributes(self._span, self._buffer)
                if completed:
                    self._span.set_status(Status(StatusCode.OK))
        finally:
            self._span.end()

    def __del__(self) -> None:
        try:
            self._finalize(completed=False)
        except Exception:
            pass


class ElevenLabsStreamingWrapper(_StreamSpanState):
    def __init__(self, span: Span, response: Iterator[Any]) -> None:
        super().__init__(span)
        self._response = response

    def __iter__(self) -> "ElevenLabsStreamingWrapper":
        return self

    def __next__(self) -> Any:
        token = context_api.attach(set_span_in_context(self._span))
        try:
            chunk = next(self._response)
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
            self._finalize(error=e)
            raise
        finally:
            context_api.detach(token)
        self._process_chunk(chunk)
        return chunk

    def close(self) -> None:
        """Close the underlying stream and end the span (for consumers that stop early)."""
        try:
            close = getattr(self._response, "close", None)
            if callable(close):
                close()
        finally:
            self._finalize(completed=False)


class ElevenLabsAsyncStreamingWrapper(_StreamSpanState):
    """Async wrapper for streaming responses (async generators)."""

    def __init__(self, span: Span, response: AsyncIterator[Any]) -> None:
        super().__init__(span)
        self._response: AsyncIterator[Any] = response

    def __aiter__(self) -> "ElevenLabsAsyncStreamingWrapper":
        """Support async iteration protocol."""
        return self

    async def __anext__(self) -> Any:
        """Iterate over async response chunks."""
        token = context_api.attach(set_span_in_context(self._span))
        try:
            chunk = await self._response.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            logger.error("netra.instrumentation.libraries.elevenlabs: %s", e)
            self._finalize(error=e)
            raise
        finally:
            context_api.detach(token)
        self._process_chunk(chunk)
        return chunk

    async def aclose(self) -> None:
        """Close the underlying stream and end the span (for consumers that stop early)."""
        try:
            aclose = getattr(self._response, "aclose", None)
            if callable(aclose):
                await aclose()
        finally:
            self._finalize(completed=False)
