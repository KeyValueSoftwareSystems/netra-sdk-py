import logging
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Tuple

from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import remove_url_credentials

from netra.instrumentation.httpx.utils import (
    get_default_span_name,
    set_span_input,
    set_span_output,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)


def send_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrapper factory for httpx.Client.send.

    Args:
        tracer: The tracer to use for the span.

    Returns:
        A wrapper function for httpx.Client.send.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            request = args[0]
            method = request.method
            url = remove_url_credentials(str(request.url))
            span_name = get_default_span_name(method)
        except Exception as e:
            logger.debug("netra.instrumentation.httpx: failed to extract request metadata: %s", e)
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": method,
                "url.full": url,
            },
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

    return wrapper


def async_send_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """
    Wrapper factory for httpx.AsyncClient.send.

    Args:
        tracer: The tracer to use for the span.

    Returns:
        A wrapper function for httpx.AsyncClient.send.
    """

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        try:
            request = args[0]
            method = request.method
            url = remove_url_credentials(str(request.url))
            span_name = get_default_span_name(method)
        except Exception as e:
            logger.debug("netra.instrumentation.httpx: failed to extract request metadata: %s", e)
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": method,
                "url.full": url,
            },
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

    return wrapper
