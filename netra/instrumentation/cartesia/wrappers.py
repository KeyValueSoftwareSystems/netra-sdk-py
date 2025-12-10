import logging
import time
from typing import Any, Callable, Dict, Iterator, Literal, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import SpanKind, Tracer, set_span_in_context
from wrapt import ObjectProxy

from netra.instrumentation.cartesia.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

TTS_SPAN_NAME = "cartesia.tts"
STT_SPAN_NAME = "cartesia.stt"
VOICE_CHANGER_SPAN_NAME = "cartesia.voice_changer"


def _wrap_tts(
    tracer: Tracer,
    span_name: str,
) -> Callable[..., Any]:
    """
    Wrap the tts method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("cartesia.response.duration", end_time - start_time)
                return response
            except Exception as e:
                logger.error("netra.instrumentation.cartesia: %s", e)
                raise

    return wrapper


def tts_bytes_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the tts methods (bytes and stream) with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_tts(tracer, TTS_SPAN_NAME)


def stt_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the stt methods with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_tts(tracer, STT_SPAN_NAME)


def voice_changer_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the voice changer methods with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_tts(tracer, VOICE_CHANGER_SPAN_NAME)


class TtsWebSocketProxy:
    """Proxy for TTS WebSocket to add instrumentation."""

    def __init__(self, ws: Any, span: Any, start_time: float) -> None:
        self._ws = ws
        self._span = span
        self._start = start_time
        self._ended = False

    def __enter__(self) -> "TtsWebSocketProxy":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self._end_span(exc_val)
        try:
            self._ws.close()
        except Exception:
            pass
        return False

    def _end_span(self, error: Any = None) -> None:
        if self._ended:
            return
        self._ended = True

        self._span.set_attribute("cartesia.response.duration", time.time() - self._start)

        self._span.end()

    def send(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            set_request_attributes(self._span, kwargs)
            for chunk in self._ws.send(*args, **kwargs):
                yield chunk

        except Exception as e:
            self._end_span(e)
            raise

        self._end_span()

    def close(self) -> None:
        try:
            self._ws.close()
        finally:
            self._end_span()


class SttWebsocketProxy(ObjectProxy):  # type: ignore[misc]
    """Proxy for STT WebSocket to add instrumentation."""

    def __init__(self, websocket: Any, span: Any, start_time: float) -> None:
        super().__init__(websocket)
        self._span = span
        self._start_time = start_time
        self._ended = False
        self._bytes_sent = 0
        self._transcripts_received = 0

    def _end_span(self, error: Any = None) -> None:
        if self._ended:
            return
        self._ended = True

        end_time = time.time()
        duration = end_time - self._start_time

        try:
            self._span.set_attribute("cartesia.response.duration", duration)
            if error is not None:
                logger.info(f"Error while ending span: {error}")
        finally:
            self._span.end()

    def receive(self) -> Iterator[Dict[str, Any]]:
        """Wrap receive to track transcription results."""
        transcripts: list[str] = []
        try:
            for result in self.__wrapped__.receive():
                set_response_attributes(self._span, result)
                try:
                    if result.get("type") == "transcript":
                        text = result.get("text", "")
                        transcripts.append(text)
                    elif result.get("type") == "done":
                        full_text = "".join(transcripts)
                        self._span.set_attribute("gen_ai.prompt.1.content", full_text)
                        yield result
                except Exception as e:
                    logger.debug("Failed to set Cartesia STT response attributes from result: %s", e)
                yield result

        except Exception as e:
            self._end_span(error=e)
            raise

    def close(self) -> Any:
        """Wrap close to end the span."""
        try:
            result = self.__wrapped__.close()
            self._end_span()
            return result
        except Exception as e:
            self._end_span(error=e)
            raise


def _wrap_tts_ws(tracer: Tracer, span_name: str) -> Callable[..., Any]:
    """Wrap TTS WebSocket method."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        start_time = time.time()
        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, kwargs)
            ws = wrapped(*args, **kwargs)
            return TtsWebSocketProxy(ws, span, start_time)
        except Exception as e:
            logger.error("netra.instrumentation.cartesia.tts.websocket: %s", e)
            span.end()
            raise
        finally:
            context_api.detach(context)

    return wrapper


def _wrap_stt_ws(tracer: Tracer, span_name: str) -> Callable[..., Any]:
    """Wrap STT WebSocket method."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        start_time = time.time()
        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, kwargs)
            ws = wrapped(*args, **kwargs)
            return SttWebsocketProxy(ws, span, start_time)
        except Exception as e:
            logger.error("netra.instrumentation.cartesia.stt.websocket: %s", e)
            span.end()
            raise
        finally:
            context_api.detach(context)

    return wrapper


def tts_ws_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the TTS WebSocket methods with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_tts_ws(tracer, "cartesia.tts.websocket")


def stt_ws_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the STT WebSocket methods with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_stt_ws(tracer, "cartesia.stt.websocket")
