from __future__ import annotations

import functools
import logging
from timeit import default_timer
from typing import Any, Callable, Dict, Optional

import requests as requests_lib  # type: ignore
from opentelemetry.instrumentation._semconv import (
    _StabilityMode,
)
from opentelemetry.instrumentation.utils import (
    is_http_instrumentation_enabled,
    suppress_http_instrumentation,
)
from opentelemetry.metrics import Histogram
from opentelemetry.propagate import inject
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.span import Span
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.http import ExcludeList, remove_url_credentials

from netra.instrumentation.requests.utils import (
    get_default_span_name,
    record_duration_metrics,
    set_http_status_code_attribute,
    set_span_attributes,
    set_span_input,
    set_span_output,
)

logger = logging.getLogger(__name__)

_RequestHookT = Optional[Callable[[Span, requests_lib.PreparedRequest], None]]
_ResponseHookT = Optional[Callable[[Span, requests_lib.PreparedRequest, requests_lib.Response], None]]


def instrument(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request_hook: _RequestHookT = None,
    response_hook: _ResponseHookT = None,
    excluded_urls: Optional[ExcludeList] = None,
    sem_conv_opt_in_mode: _StabilityMode = _StabilityMode.DEFAULT,
) -> None:
    """Patches requests.Session.send with tracing."""

    wrapped_send = requests_lib.Session.send

    @functools.wraps(wrapped_send)
    def instrumented_send(
        self: requests_lib.Session,
        request: requests_lib.PreparedRequest,
        **kwargs: Any,
    ) -> requests_lib.Response:
        if excluded_urls and excluded_urls.url_disabled(request.url or ""):
            return wrapped_send(self, request, **kwargs)
        if not is_http_instrumentation_enabled():
            return wrapped_send(self, request, **kwargs)
        return trace_request(
            tracer,
            duration_histogram_old,
            duration_histogram_new,
            request,
            wrapped_send,
            self,
            request_hook,
            response_hook,
            sem_conv_opt_in_mode,
            **kwargs,
        )

    instrumented_send.opentelemetry_instrumentation_requests_applied = True  # type: ignore[attr-defined]
    requests_lib.Session.send = instrumented_send


def trace_request(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request: requests_lib.PreparedRequest,
    send_func: Callable[..., requests_lib.Response],
    session: requests_lib.Session,
    request_hook: _RequestHookT,
    response_hook: _ResponseHookT,
    sem_conv_opt_in_mode: _StabilityMode,
    **kwargs: Any,
) -> requests_lib.Response:
    method = (request.method or "").upper()
    url = remove_url_credentials(request.url or "")

    span_attributes: Dict[str, Any] = {}
    metric_labels: Dict[str, Any] = {}
    set_span_attributes(span_attributes, metric_labels, method, url, sem_conv_opt_in_mode)

    with tracer.start_as_current_span(
        get_default_span_name(method), kind=SpanKind.CLIENT, attributes=span_attributes
    ) as span:
        exception = None
        result: Optional[requests_lib.Response] = None

        set_span_input(span, request)

        if callable(request_hook):
            request_hook(span, request)

        # Inject W3C trace context into the outgoing headers.
        inject(request.headers)

        with suppress_http_instrumentation():
            start_time = default_timer()
            try:
                result = send_func(session, request, **kwargs)
            except Exception as exc:
                exception = exc
                result = getattr(exc, "response", None)
            finally:
                elapsed_time = max(default_timer() - start_time, 0)

        if isinstance(result, requests_lib.Response):
            set_http_status_code_attribute(span, result.status_code, metric_labels, sem_conv_opt_in_mode)
            set_span_output(span, result)

            if callable(response_hook):
                response_hook(span, request, result)

        if exception is not None:
            from opentelemetry.instrumentation._semconv import _report_new

            if _report_new(sem_conv_opt_in_mode):
                span.set_attribute(ERROR_TYPE, type(exception).__qualname__)
                metric_labels[ERROR_TYPE] = type(exception).__qualname__
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
        elif isinstance(result, requests_lib.Response) and result.status_code >= 500:
            span.set_status(Status(StatusCode.ERROR, f"HTTP {result.status_code}"))
        else:
            span.set_status(Status(StatusCode.OK))

        record_duration_metrics(
            duration_histogram_old, duration_histogram_new, elapsed_time, metric_labels, sem_conv_opt_in_mode
        )

        if exception is not None:
            raise exception.with_traceback(exception.__traceback__)

    return result


def uninstrument() -> None:
    instr_func = requests_lib.Session.send
    if not getattr(instr_func, "opentelemetry_instrumentation_requests_applied", False):
        return
    requests_lib.Session.send = instr_func.__wrapped__
