import logging
from typing import Any, Callable, Dict, Optional, Tuple

from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.honcho import constants as attrs
from netra.instrumentation.honcho.utils import (
    RequestAttrFn,
    ResponseAttrFn,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)


def _safe_set_request_attrs(
    span: Span, fn: RequestAttrFn, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    try:
        fn(span, instance, args, kwargs)
    except Exception:
        logger.debug("%s: failed to set request attributes", attrs.LOG_PREFIX, exc_info=True)


def _safe_set_response_attrs(span: Span, fn: ResponseAttrFn, response: Any) -> None:
    try:
        fn(span, response)
    except Exception:
        logger.debug("%s: failed to set response attributes", attrs.LOG_PREFIX, exc_info=True)


def _record_span_error(span: Span, error: Exception) -> None:
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)


class _BaseStreamingWrapper:
    """Shared span lifecycle for sync and async streaming wrappers.

    Subclasses only need to implement the iteration protocol
    (``__iter__``/``__next__`` or ``__aiter__``/``__anext__``).
    """

    def __init__(self, span: Span, response: Any) -> None:
        self._span = span
        self._response = response
        self._span_ended = False

    def get_final_response(self) -> Dict[str, str]:
        result: Dict[str, str] = self._response.get_final_response()
        return result

    @property
    def is_complete(self) -> bool:
        return bool(self._response.is_complete)

    def _finalize_span(self, error: Optional[Exception] = None) -> None:
        if self._span_ended:
            return
        self._span_ended = True
        try:
            if error:
                _record_span_error(self._span, error)
            else:
                final = self.get_final_response()
                content = final.get("content", "")
                if content:
                    self._span.set_attribute(attrs.RESPONSE_LENGTH, len(content))
                    self._span.set_attribute(attrs.OUTPUT, content)
                self._span.set_status(Status(StatusCode.OK))
        except Exception:
            logger.debug("%s: failed to finalize streaming span", attrs.LOG_PREFIX, exc_info=True)
        finally:
            self._span.end()

    def __del__(self) -> None:
        try:
            self._finalize_span()
        except Exception:
            pass


class StreamingChatWrapper(_BaseStreamingWrapper):
    """Wraps a sync DialecticStreamResponse to finalize the span after iteration."""

    def __iter__(self) -> "StreamingChatWrapper":
        return self

    def __next__(self) -> str:
        try:
            chunk: str = next(self._response)
            return chunk
        except StopIteration:
            self._finalize_span()
            raise
        except Exception as e:
            self._finalize_span(error=e)
            raise


class AsyncStreamingChatWrapper(_BaseStreamingWrapper):
    """Wraps an async DialecticStreamResponse to finalize the span after iteration."""

    def __aiter__(self) -> "AsyncStreamingChatWrapper":
        return self

    async def __anext__(self) -> str:
        try:
            chunk: str = await self._response.__anext__()
            return chunk
        except StopAsyncIteration:
            self._finalize_span()
            raise
        except Exception as e:
            self._finalize_span(error=e)
            raise


def make_sync_wrapper(
    tracer: Tracer,
    span_name: str,
    request_attr_fn: RequestAttrFn,
    response_attr_fn: ResponseAttrFn,
) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            _safe_set_request_attrs(span, request_attr_fn, instance, args, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                _record_span_error(span, e)
                raise
            _safe_set_response_attrs(span, response_attr_fn, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def make_async_wrapper(
    tracer: Tracer,
    span_name: str,
    request_attr_fn: RequestAttrFn,
    response_attr_fn: ResponseAttrFn,
) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            _safe_set_request_attrs(span, request_attr_fn, instance, args, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                _record_span_error(span, e)
                raise
            _safe_set_response_attrs(span, response_attr_fn, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def make_chat_stream_sync_wrapper(
    tracer: Tracer,
    span_name: str,
    request_attr_fn: RequestAttrFn,
) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        _safe_set_request_attrs(span, request_attr_fn, instance, args, kwargs)
        span.set_attribute(attrs.REQUEST_STREAM, True)

        try:
            return StreamingChatWrapper(span, wrapped(*args, **kwargs))
        except Exception as e:
            _record_span_error(span, e)
            span.end()
            raise

    return wrapper


def make_chat_stream_async_wrapper(
    tracer: Tracer,
    span_name: str,
    request_attr_fn: RequestAttrFn,
) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        span = tracer.start_span(span_name, kind=SpanKind.CLIENT)
        _safe_set_request_attrs(span, request_attr_fn, instance, args, kwargs)
        span.set_attribute(attrs.REQUEST_STREAM, True)

        try:
            return AsyncStreamingChatWrapper(span, await wrapped(*args, **kwargs))
        except Exception as e:
            _record_span_error(span, e)
            span.end()
            raise

    return wrapper
