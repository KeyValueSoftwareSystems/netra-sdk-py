import logging
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Any, Callable, Dict, List, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.propagate import inject
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import remove_url_credentials
from wrapt import ObjectProxy

from netra.instrumentation.httpx.utils import (
    get_default_span_name,
    set_span_input,
    set_span_output,
    set_streaming_span_output,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)


class _BaseStreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Base proxy for streaming httpx responses; finalizes the span when the stream ends."""

    def __init__(self, response: Any, span: Span) -> None:
        super().__init__(response)
        self._span = span
        self._chunks: List[bytes] = []
        self._finalized = False

    def _finalize_span(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            set_streaming_span_output(self._span, self.__wrapped__, self._chunks)
        except Exception as e:
            logger.debug("netra.instrumentation.httpx: failed to finalize streaming span: %s", e)
        finally:
            self._span.end()

    def __del__(self) -> None:
        self._finalize_span()


class StreamingWrapper(_BaseStreamingWrapper):
    """Wraps a streaming httpx.Response, keeping the span open until the stream is closed.

    httpx streaming responses are consumed via ``iter_bytes()``, ``iter_text()``,
    ``iter_lines()``, or ``iter_raw()`` — not via ``iter(response)``.  This wrapper
    proxies those methods so that each yielded chunk is captured for the span
    output attribute, and overrides ``close()`` to finalize the span.
    """

    def _wrap_iter(self, inner: Iterator[Any]) -> Iterator[Any]:
        """Proxy a synchronous iterator, accumulating raw bytes for the span."""
        try:
            for chunk in inner:
                if isinstance(chunk, bytes):
                    self._chunks.append(chunk)
                elif isinstance(chunk, str):
                    self._chunks.append(chunk.encode("utf-8"))
                yield chunk
        except GeneratorExit:
            return
        except Exception as e:
            self._span.set_status(Status(StatusCode.ERROR, str(e)))
            self._span.record_exception(e)
            raise

    def iter_bytes(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        """Proxy ``Response.iter_bytes``, capturing chunks for span output."""
        return self._wrap_iter(self.__wrapped__.iter_bytes(*args, **kwargs))

    def iter_text(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        """Proxy ``Response.iter_text``, capturing chunks for span output."""
        return self._wrap_iter(self.__wrapped__.iter_text(*args, **kwargs))

    def iter_lines(self, *args: Any, **kwargs: Any) -> Iterator[str]:
        """Proxy ``Response.iter_lines``, capturing chunks for span output."""
        return self._wrap_iter(self.__wrapped__.iter_lines(*args, **kwargs))

    def iter_raw(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        """Proxy ``Response.iter_raw``, capturing chunks for span output."""
        return self._wrap_iter(self.__wrapped__.iter_raw(*args, **kwargs))

    def close(self) -> None:
        """Close the underlying response and finalize the span."""
        try:
            self.__wrapped__.close()
        finally:
            self._finalize_span()


class AsyncStreamingWrapper(_BaseStreamingWrapper):
    """Wraps a streaming httpx.Response from an AsyncClient, keeping the span open until closed.

    Mirrors :class:`StreamingWrapper` for the async iteration methods
    (``aiter_bytes``, ``aiter_text``, ``aiter_lines``, ``aiter_raw``) and
    overrides ``aclose()`` to finalize the span.
    """

    async def _wrap_aiter(self, inner: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Proxy an asynchronous iterator, accumulating raw bytes for the span."""
        try:
            async for chunk in inner:
                if isinstance(chunk, bytes):
                    self._chunks.append(chunk)
                elif isinstance(chunk, str):
                    self._chunks.append(chunk.encode("utf-8"))
                yield chunk
        except GeneratorExit:
            return
        except Exception as e:
            self._span.set_status(Status(StatusCode.ERROR, str(e)))
            self._span.record_exception(e)
            raise

    def aiter_bytes(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        """Proxy ``Response.aiter_bytes``, capturing chunks for span output."""
        return self._wrap_aiter(self.__wrapped__.aiter_bytes(*args, **kwargs))

    def aiter_text(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Proxy ``Response.aiter_text``, capturing chunks for span output."""
        return self._wrap_aiter(self.__wrapped__.aiter_text(*args, **kwargs))

    def aiter_lines(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Proxy ``Response.aiter_lines``, capturing chunks for span output."""
        return self._wrap_aiter(self.__wrapped__.aiter_lines(*args, **kwargs))

    def aiter_raw(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        """Proxy ``Response.aiter_raw``, capturing chunks for span output."""
        return self._wrap_aiter(self.__wrapped__.aiter_raw(*args, **kwargs))

    async def aclose(self) -> None:
        """Close the underlying response and finalize the span."""
        try:
            await self.__wrapped__.aclose()
        finally:
            self._finalize_span()


def send_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper factory for httpx.Client.send."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:

            request = args[0] if args else kwargs.get("request")
            if request is None:
                return wrapped(*args, **kwargs)
            method = request.method
            url = remove_url_credentials(str(request.url))
            span_name = get_default_span_name(method)
        except Exception as e:
            logger.debug("netra.instrumentation.httpx: failed to extract request metadata: %s", e)
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)
        if not is_streaming:
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                attributes={"http.request.method": method, "url.full": url},
            ) as span:
                try:
                    set_span_input(span, request)
                    headers = dict(request.headers)
                    inject(headers)
                    request.headers.update(headers)
                except Exception as e:
                    logger.debug("netra.instrumentation.httpx: failed to set span input: %s", e)

                try:
                    with suppress_http_instrumentation():
                        response = wrapped(*args, **kwargs)
                except Exception as e:
                    logger.error("netra.instrumentation.httpx: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

                try:
                    span.set_attribute("http.response.status_code", response.status_code)
                    set_span_output(span, response)
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    logger.debug("netra.instrumentation.httpx: failed to process response span: %s", e)

                return response

        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"http.request.method": method, "url.full": url},
        )
        try:
            context = context_api.attach(set_span_in_context(span))
            try:
                set_span_input(span, request)
                headers = dict(request.headers)
                inject(headers)
                request.headers.update(headers)
            except Exception as e:
                logger.debug("netra.instrumentation.httpx: failed to set span input: %s", e)

            try:
                with suppress_http_instrumentation():
                    response = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.httpx: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise

            try:
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("netra.instrumentation.httpx: failed to set response status on span: %s", e)

            return StreamingWrapper(response=response, span=span)
        finally:
            context_api.detach(context)

    return wrapper


def async_send_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Wrapper factory for httpx.AsyncClient.send."""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        try:
            request = args[0] if args else kwargs.get("request")
            if request is None:
                return await wrapped(*args, **kwargs)
            method = request.method
            url = remove_url_credentials(str(request.url))
            span_name = get_default_span_name(method)
        except Exception as e:
            logger.debug("netra.instrumentation.httpx: failed to extract request metadata: %s", e)
            return await wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)
        if not is_streaming:
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                attributes={"http.request.method": method, "url.full": url},
            ) as span:
                try:
                    set_span_input(span, request)
                    headers = dict(request.headers)
                    inject(headers)
                    request.headers.update(headers)
                except Exception as e:
                    logger.debug("netra.instrumentation.httpx: failed to set span input: %s", e)

                try:
                    with suppress_http_instrumentation():
                        response = await wrapped(*args, **kwargs)
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

                try:
                    span.set_attribute("http.response.status_code", response.status_code)
                    set_span_output(span, response)
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    logger.debug("netra.instrumentation.httpx: failed to process response span: %s", e)

                return response

        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"http.request.method": method, "url.full": url},
        )
        try:
            context = context_api.attach(set_span_in_context(span))
            try:
                set_span_input(span, request)
                headers = dict(request.headers)
                inject(headers)
                request.headers.update(headers)
            except Exception as e:
                logger.debug("netra.instrumentation.httpx: failed to set span input: %s", e)

            try:
                with suppress_http_instrumentation():
                    response = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise

            try:
                span.set_attribute("http.response.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("netra.instrumentation.httpx: failed to set response status on span: %s", e)

            return AsyncStreamingWrapper(response=response, span=span)
        finally:
            context_api.detach(context)

    return wrapper
