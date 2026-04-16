import logging
from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.propagate import inject
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import remove_url_credentials
from wrapt import ObjectProxy

from netra.instrumentation.requests.utils import (
    get_default_span_name,
    set_span_input,
    set_span_output,
    set_streaming_span_output,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)


class StreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Wraps a streaming requests.Response, keeping the span open until the stream is closed."""

    def __init__(self, response: Any, span: Span) -> None:
        """Initialize the streaming wrapper.

        Args:
            response: The streaming requests.Response to wrap.
            span: The open OpenTelemetry span to keep alive during streaming.
        """
        super().__init__(response)
        self._span = span
        self._chunks: List[bytes] = []
        self._finalized = False

    def _wrap_iter(self, inner: Iterator[Any]) -> Iterator[Any]:
        """Proxy a synchronous iterator, accumulating raw bytes for the span.

        Args:
            inner: The underlying iterator to wrap.

        Yields:
            Each chunk from *inner* unchanged.
        """
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

    def __iter__(self) -> Iterator[Any]:
        """Proxy direct iteration over the response, capturing chunks for span output.

        Returns:
            An iterator that yields response chunks and accumulates them for
            the span output attribute.
        """
        return self._wrap_iter(iter(self.__wrapped__))

    def iter_content(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        """Proxy ``Response.iter_content``, capturing chunks for span output.

        Args:
            *args: Positional arguments forwarded to ``Response.iter_content``.
            **kwargs: Keyword arguments forwarded to ``Response.iter_content``.

        Returns:
            An iterator that yields raw bytes chunks and accumulates them for
            the span output attribute.
        """
        return self._wrap_iter(self.__wrapped__.iter_content(*args, **kwargs))

    def iter_lines(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Proxy ``Response.iter_lines``, capturing chunks for span output.

        Args:
            *args: Positional arguments forwarded to ``Response.iter_lines``.
            **kwargs: Keyword arguments forwarded to ``Response.iter_lines``.

        Returns:
            An iterator that yields decoded line strings and accumulates the
            raw bytes for the span output attribute.
        """
        return self._wrap_iter(self.__wrapped__.iter_lines(*args, **kwargs))

    def _finalize_span(self) -> None:
        """Write accumulated chunk data to the span output attribute and end the span.

        Idempotent — subsequent calls after the first are no-ops.
        """
        if self._finalized:
            return
        self._finalized = True
        try:
            set_streaming_span_output(self._span, self.__wrapped__, self._chunks)
        except Exception as e:
            logger.debug("netra.instrumentation.requests: failed to finalize streaming span: %s", e)
        finally:
            self._span.end()

    def __enter__(self) -> "StreamingWrapper":
        """Return the wrapper itself so iteration methods are captured inside a with-block.

        Returns:
            This StreamingWrapper instance.
        """
        self.__wrapped__.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Close via the wrapper so the span is finalized.

        Args:
            *args: Exception info tuple (exc_type, exc_val, exc_tb) forwarded
                from the context manager protocol.
        """
        self.close()

    def close(self) -> None:
        """Close the underlying response and finalize the span.

        Calls ``Response.close()`` on the wrapped response, then invokes
        :meth:`_finalize_span` to record the accumulated output and end the span.
        """
        try:
            self.__wrapped__.close()
        finally:
            self._finalize_span()

    def __del__(self) -> None:
        """Finalize the span on garbage collection as a last-resort safety net."""
        self._finalize_span()


def send_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``requests.Session.send``.

    Args:
        tracer: The OpenTelemetry Tracer used to create spans.

    Returns:
        A callable suitable for use with ``wrap_function_wrapper``.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Intercept ``Session.send``, create a span, and capture request/response data.

        Args:
            wrapped: The original ``Session.send`` method.
            instance: The ``Session`` instance on which the method is called.
            args: Positional arguments passed to ``Session.send``; the first
                element is the ``PreparedRequest``.
            kwargs: Keyword arguments passed to ``Session.send``.

        Returns:
            The original ``Response`` for non-streaming requests, or a
            :class:`StreamingWrapper` that keeps the span open while the
            caller iterates over a streaming response.
        """
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            request = args[0] if args else kwargs.get("request")
            if request is None:
                raise ValueError("No request object found in arguments")
            method = (request.method or "").upper()
            url = remove_url_credentials(request.url or "")
            span_name = get_default_span_name(method)
        except Exception as e:
            logger.debug("netra.instrumentation.requests: failed to extract request metadata: %s", e)
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)
        if not is_streaming:
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                attributes={"http.method": method, "http.url": url},
            ) as span:
                try:
                    set_span_input(span, request)
                    inject(request.headers)
                except Exception as e:
                    logger.debug("netra.instrumentation.requests: failed to set span input: %s", e)

                try:
                    with suppress_http_instrumentation():
                        response = wrapped(*args, **kwargs)
                except Exception as e:
                    logger.error("netra.instrumentation.requests: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

                try:
                    span.set_attribute("http.status_code", response.status_code)
                    set_span_output(span, response)
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    logger.debug("netra.instrumentation.requests: failed to process response span: %s", e)

                return response

        span = tracer.start_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"http.method": method, "http.url": url},
        )
        try:
            context = context_api.attach(set_span_in_context(span))
            try:
                set_span_input(span, request)
                inject(request.headers)
            except Exception as e:
                logger.debug("netra.instrumentation.requests: failed to set span input: %s", e)

            try:
                with suppress_http_instrumentation():
                    response = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.requests: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise

            try:
                span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("netra.instrumentation.requests: failed to set response status on span: %s", e)

            return StreamingWrapper(response=response, span=span)
        finally:
            context_api.detach(context)

    return wrapper
