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
    """Wraps a streaming requests.Response, keeping the span open until the stream is exhausted."""

    def __init__(self, response: Any, span: Span) -> None:
        super().__init__(response)
        self._span = span
        self._chunks: List[bytes] = []
        self._finalized = False
        self._iterator: Iterator[Any] = iter(response)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
            if isinstance(chunk, bytes):
                self._chunks.append(chunk)
            elif isinstance(chunk, str):
                self._chunks.append(chunk.encode("utf-8"))
            return chunk
        except StopIteration:
            self._finalize_span()
            raise
        except Exception as e:
            self._span.set_status(Status(StatusCode.ERROR, str(e)))
            self._span.record_exception(e)
            self._finalize_span()
            raise

    def _finalize_span(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            set_streaming_span_output(self._span, self.__wrapped__, self._chunks)
        except Exception as e:
            logger.debug("netra.instrumentation.requests: failed to finalize streaming span: %s", e)
        finally:
            self._span.end()

    def __del__(self) -> None:
        self._finalize_span()


def send_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper factory for requests.Session.send."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            request = args[0] if args else kwargs.get("request")
            if request is None:
                return wrapped(*args, **kwargs)
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
                attributes={"http.request.method": method, "url.full": url},
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
                    span.set_attribute("http.response.status_code", response.status_code)
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
            attributes={"http.request.method": method, "url.full": url},
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
                span.set_attribute("http.response.status_code", response.status_code)
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
