import logging
from typing import Any, Callable, Dict, Tuple

from opentelemetry.instrumentation.utils import suppress_http_instrumentation
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import remove_url_credentials

from netra.instrumentation.requests.utils import (
    get_default_span_name,
    set_span_input,
    set_span_output,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)


def send_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrapper factory for requests.Session.send.

    Args:
        tracer: The tracer to use for the span.

    Returns:
        A wrapper function for requests.Session.send.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        request = args[0]
        method = (request.method or "").upper()
        url = remove_url_credentials(request.url or "")
        span_name = get_default_span_name(method)

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

                inject(request.headers)

                with suppress_http_instrumentation():
                    response = wrapped(*args, **kwargs)

                if hasattr(response, "status_code"):
                    span.set_attribute("http.response.status_code", response.status_code)
                    set_span_output(span, response)

                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                logger.error("netra.instrumentation.requests: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper
