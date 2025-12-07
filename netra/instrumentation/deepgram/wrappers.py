import logging
import time
from typing import Any, Callable, Dict, Tuple, cast

from opentelemetry import context as context_api
from opentelemetry.trace import SpanKind, Tracer, set_span_in_context
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
        """
        Wrap a websocket connection with OpenTelemetry instrumentation.

        Args:
            connection: The websocket connection to wrap.
            span: The OpenTelemetry span to use for instrumentation.
            start_time: The start time of the span.
        """
        super().__init__(connection)
        self._span = span
        self._start_time = start_time
        self._ended = False

    def _end_span(self, error: Any = None) -> None:
        """
        End the span.

        Args:
            error: The error to set on the span.
        """
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
        """
        Connection event handler.

        Args:
            event_type: The event type.
            handler: The event handler.
        """
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
        """
        Wrap a context manager

        Args:
            context_manager: The context manager to wrap.
            span: The OpenTelemetry span to use for instrumentation.
            start_time: The start time of the span.
        """
        self._context_manager = context_manager
        self._span = span
        self._start_time = start_time
        self._connection_proxy: Any = None

    def __enter__(self) -> WebSocketConnectionProxy:
        """
        Enter the context manager.
        """
        connection = self._context_manager.__enter__()
        self._connection_proxy = WebSocketConnectionProxy(connection, self._span, self._start_time)
        return cast(WebSocketConnectionProxy, self._connection_proxy)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """
        Exit the context manager.
        """
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
        """
        Wrap an AsyncContextManager

        Args:
            async_context_manager: The AsyncContextManager to wrap.
            span: The OpenTelemetry span to use for instrumentation.
            start_time: The start time of the span.
            request_kwargs: The request kwargs to use for instrumentation.
        """
        self._async_context_manager = async_context_manager
        self._span = span
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._connection_proxy: Any = None

    async def __aenter__(self) -> WebSocketConnectionProxy:
        """
        Enter the async context manager.
        """
        connection = await self._async_context_manager.__aenter__()
        self._connection_proxy = WebSocketConnectionProxy(connection, self._span, self._start_time)
        return cast(WebSocketConnectionProxy, self._connection_proxy)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """
        Exit the async context manager.
        """
        result = await self._async_context_manager.__aexit__(exc_type, exc_val, exc_tb)
        if self._connection_proxy is not None:
            self._connection_proxy._end_span(exc_val)
        return result


def wrap_sync(
    tracer: Tracer,
    span_name: str,
    source_type: str | None = None,
) -> Callable[..., Any]:
    """
    Wrap a sync method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
        source_type: Optional type of the source (e.g. "url" or "file").
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


def wrap_async(
    tracer: Tracer,
    span_name: str,
    source_type: str | None = None,
) -> Callable[..., Any]:
    """
    Wrap an async method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
        source_type: Optional type of the source (e.g. "url" or "file").
    """

    async def async_wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs, source_type)
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
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

    return async_wrapper


def wrap_connect(tracer: Tracer, span_name: str) -> Callable[..., Any]:
    """
    Wrap the connect method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
    """

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


def wrap_connect_async(tracer: Tracer, span_name: str) -> Callable[..., Any]:
    """
    Wrap async connect methods that return async context managers.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
    """

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


def wrap_async_generator(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the async generate method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """

    async def async_wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            generator = wrapped(*args, **kwargs)
            try:
                async for chunk in generator:
                    yield chunk
            finally:
                await generator.aclose()
            return

        span = tracer.start_span(GENERATE_SPAN_NAME, kind=SpanKind.CLIENT)
        start_time = time.time()
        generator = None
        context = None
        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, kwargs)
            generator = wrapped(*args, **kwargs)
            async for chunk in generator:
                yield chunk
            end_time = time.time()
            span.set_attribute("deepgram.response.duration", end_time - start_time)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error("netra.instrumentation.deepgram: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            if generator is not None:
                await generator.aclose()
            if context is not None:
                try:
                    context_api.detach(context)
                except ValueError:
                    pass
            span.end()

    return async_wrapper
