import logging
import time
from typing import Any, Callable, Dict, Tuple, cast

from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

from netra.instrumentation.deepgram.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

TRANSCRIBE_URL_SPAN_NAME = "deepgram.transcribe_url"
TRANSCRIBE_FILE_SPAN_NAME = "deepgram.transcribe_file"
ANALYZE_SPAN_NAME = "deepgram.analyze"
GENERATE_SPAN_NAME = "deepgram.generate"
LISTEN_V1_CONNECT_SPAN_NAME = "deepgram.listen.v1.connect"
LISTEN_V2_CONNECT_SPAN_NAME = "deepgram.listen.v2.connect"
SPEAK_V1_CONNECT_SPAN_NAME = "deepgram.speak.v1.connect"
AGENT_V1_CONNECT_SPAN_NAME = "deepgram.agent.v1.connect"


class WebSocketConnectionProxy(ObjectProxy):  # type: ignore[misc]

    def __init__(self, connection: Any, span: Any, start_time: float) -> None:
        super().__init__(connection)
        self._span = span
        self._start_time = start_time
        self._ended = False

    def _end_span(self, error: Any = None) -> None:
        if self._ended:
            return
        self._ended = True
        end_time = time.time()
        duration = end_time - self._start_time
        try:
            self._span.set_attribute("deepgram.websocket.duration", duration)
            if error is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
                self._span.record_exception(error)
            else:
                self._span.set_status(Status(StatusCode.OK))
        finally:
            self._span.end()

    def on(self, event_type: Any, handler: Callable[..., Any]) -> Any:
        event_name = getattr(event_type, "name", str(event_type))

        if event_name == "MESSAGE":

            def wrapped_message_handler(*args: Any, **kwargs: Any) -> Any:
                message = args[0] if args else kwargs.get("message")
                try:
                    set_response_attributes(self._span, message)
                except Exception:
                    logger.debug("Failed to set Deepgram websocket response attributes from message")
                return handler(*args, **kwargs)

            return self.__wrapped__.on(event_type, wrapped_message_handler)

        if event_name in ("CLOSE", "ERROR"):

            def wrapped_close_error_handler(*args: Any, **kwargs: Any) -> Any:
                error = args[0] if (event_name == "ERROR" and args) else None
                self._end_span(error)
                return handler(*args, **kwargs)

            return self.__wrapped__.on(event_type, wrapped_close_error_handler)

        return self.__wrapped__.on(event_type, handler)


class ContextManagerProxy:

    def __init__(self, context_manager: Any, span: Any, start_time: float) -> None:
        self._context_manager = context_manager
        self._span = span
        self._start_time = start_time
        self._connection_proxy: Any = None

    def __enter__(self) -> WebSocketConnectionProxy:
        connection = self._context_manager.__enter__()
        self._connection_proxy = WebSocketConnectionProxy(connection, self._span, self._start_time)
        return cast(WebSocketConnectionProxy, self._connection_proxy)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = self._context_manager.__exit__(exc_type, exc_val, exc_tb)
        if self._connection_proxy is not None:
            self._connection_proxy._end_span(exc_val)
        return result


class AsyncContextManagerProxy:
    """Async context manager proxy for wrapping async websocket connections."""

    def __init__(
        self,
        async_context_manager: Any,
        span: Any,
        start_time: float,
        request_kwargs: Dict[str, Any],
    ) -> None:
        self._async_context_manager = async_context_manager
        self._span = span
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._connection_proxy: Any = None

    async def __aenter__(self) -> WebSocketConnectionProxy:
        connection = await self._async_context_manager.__aenter__()
        self._connection_proxy = WebSocketConnectionProxy(connection, self._span, self._start_time)
        return cast(WebSocketConnectionProxy, self._connection_proxy)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        result = await self._async_context_manager.__aexit__(exc_type, exc_val, exc_tb)
        if self._connection_proxy is not None:
            self._connection_proxy._end_span(exc_val)
        return result


def _wrap_transcribe(
    tracer: Tracer,
    span_name: str,
    source_type: str,
) -> Callable[..., Any]:
    """
    Wrap the transcribe_url method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
        source_type: The type of the source (e.g. "url" or "file").
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs, source_type)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("deepgram.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.deepgram: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def transcribe_url_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the transcribe_url method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_transcribe(tracer, TRANSCRIBE_URL_SPAN_NAME, "url")


def transcribe_file_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the transcribe_file method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_transcribe(tracer, TRANSCRIBE_FILE_SPAN_NAME, "file")


def analyze_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the analyze method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(ANALYZE_SPAN_NAME, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("deepgram.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.deepgram: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def _wrap_connect(tracer: Tracer, span_name: str) -> Callable[..., Any]:

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        start_time = time.time()
        try:
            set_request_attributes(span, kwargs)
            context_manager = wrapped(*args, **kwargs)
            return ContextManagerProxy(context_manager, span, start_time)
        except Exception as e:
            logger.error("netra.instrumentation.deepgram: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    return wrapper


def _wrap_connect_async(tracer: Tracer, span_name: str) -> Callable[..., Any]:
    """Wrap async connect methods that return async context managers."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        start_time = time.time()
        try:
            set_request_attributes(span, kwargs)
            # wrapped(*args, **kwargs) returns an async context manager, not a coroutine
            async_context_manager = wrapped(*args, **kwargs)
            return AsyncContextManagerProxy(async_context_manager, span, start_time, kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.deepgram: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    return wrapper


def listen_v1_connect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect(tracer, LISTEN_V1_CONNECT_SPAN_NAME)


def listen_v2_connect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect(tracer, LISTEN_V2_CONNECT_SPAN_NAME)


def speak_v1_connect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect(tracer, SPEAK_V1_CONNECT_SPAN_NAME)


def agent_v1_connect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect(tracer, AGENT_V1_CONNECT_SPAN_NAME)


def listen_v1_aconnect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect_async(tracer, LISTEN_V1_CONNECT_SPAN_NAME)


def listen_v2_aconnect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect_async(tracer, LISTEN_V2_CONNECT_SPAN_NAME)


def speak_v1_aconnect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect_async(tracer, SPEAK_V1_CONNECT_SPAN_NAME)


def agent_v1_aconnect_wrapper(tracer: Tracer) -> Callable[..., Any]:
    return _wrap_connect_async(tracer, AGENT_V1_CONNECT_SPAN_NAME)


def generate_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the generate method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(GENERATE_SPAN_NAME, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("deepgram.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.deepgram: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper
