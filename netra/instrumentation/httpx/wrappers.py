from __future__ import annotations

import functools
import logging
import types
from timeit import default_timer
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import httpx
from opentelemetry.instrumentation._semconv import (
    _report_new,
    _set_http_network_protocol_version,
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
from opentelemetry.util.http.httplib import set_ip_on_next_http_connection

from netra.instrumentation.httpx.utils import (
    get_default_span_name,
    record_duration_metrics,
    set_http_status_code_attribute,
    set_span_attributes,
    set_span_input,
    set_span_output,
)

logger = logging.getLogger(__name__)

_RequestHookT = Optional[Callable[[Span, httpx.Request], None]]
_ResponseHookT = Optional[Callable[[Span, httpx.Request, httpx.Response], None]]


def instrument(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request_hook: _RequestHookT = None,
    response_hook: _ResponseHookT = None,
    excluded_urls: Optional[ExcludeList] = None,
    sem_conv_opt_in_mode: _StabilityMode = _StabilityMode.DEFAULT,
) -> None:
    """Patches httpx.Client.send and httpx.AsyncClient.send with tracing."""

    wrapped_send = httpx.Client.send

    @functools.wraps(wrapped_send)
    def instrumented_send(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if excluded_urls and excluded_urls.url_disabled(str(request.url)):
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

    instrumented_send.opentelemetry_instrumentation_httpx_applied = True  # type: ignore[attr-defined]
    httpx.Client.send = instrumented_send

    wrapped_async_send = httpx.AsyncClient.send

    @functools.wraps(wrapped_async_send)
    async def instrumented_async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        if excluded_urls and excluded_urls.url_disabled(str(request.url)):
            return await wrapped_async_send(self, request, **kwargs)
        if not is_http_instrumentation_enabled():
            return await wrapped_async_send(self, request, **kwargs)
        return await trace_async_request(
            tracer,
            duration_histogram_old,
            duration_histogram_new,
            request,
            wrapped_async_send,
            self,
            request_hook,
            response_hook,
            sem_conv_opt_in_mode,
            **kwargs,
        )

    instrumented_async_send.opentelemetry_instrumentation_httpx_applied = True  # type: ignore[attr-defined]
    httpx.AsyncClient.send = instrumented_async_send


def trace_request(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request: httpx.Request,
    send_func: Callable[..., httpx.Response],
    client: httpx.Client,
    request_hook: _RequestHookT,
    response_hook: _ResponseHookT,
    sem_conv_opt_in_mode: _StabilityMode,
    **kwargs: Any,
) -> httpx.Response:
    method = request.method
    url = remove_url_credentials(str(request.url))

    span_attributes: Dict[str, Any] = {}
    metric_labels: Dict[str, Any] = {}
    set_span_attributes(span_attributes, metric_labels, method, url, sem_conv_opt_in_mode)

    with (
        tracer.start_as_current_span(
            get_default_span_name(method), kind=SpanKind.CLIENT, attributes=span_attributes
        ) as span,
        set_ip_on_next_http_connection(span),
    ):
        exception = None
        set_span_input(span, request)

        if callable(request_hook):
            request_hook(span, request)

        headers = dict(request.headers)
        inject(headers)
        request.headers.update(headers)

        with suppress_http_instrumentation():
            start_time = default_timer()
            try:
                result = send_func(client, request, **kwargs)
            except Exception as exc:
                exception = exc
                result = getattr(exc, "response", None)
            finally:
                elapsed_time = max(default_timer() - start_time, 0)

        if isinstance(result, httpx.Response):
            span_attributes_response: Dict[str, Any] = {}
            set_http_status_code_attribute(span, result.status_code, metric_labels, sem_conv_opt_in_mode)

            if hasattr(result, "http_version"):
                _set_http_network_protocol_version(metric_labels, result.http_version, sem_conv_opt_in_mode)
                if _report_new(sem_conv_opt_in_mode):
                    _set_http_network_protocol_version(
                        span_attributes_response, result.http_version, sem_conv_opt_in_mode
                    )

            for key, val in span_attributes_response.items():
                span.set_attribute(key, val)

            set_span_output(span, result)

            if callable(response_hook):
                response_hook(span, request, result)

        if exception is not None:
            if _report_new(sem_conv_opt_in_mode):
                span.set_attribute(ERROR_TYPE, type(exception).__qualname__)
                metric_labels[ERROR_TYPE] = type(exception).__qualname__
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
        elif isinstance(result, httpx.Response) and result.status_code >= 500:
            span.set_status(Status(StatusCode.ERROR, f"HTTP {result.status_code}"))
        else:
            span.set_status(Status(StatusCode.OK))

        record_duration_metrics(
            duration_histogram_old, duration_histogram_new, elapsed_time, metric_labels, sem_conv_opt_in_mode
        )

        if exception is not None:
            raise exception.with_traceback(exception.__traceback__)

    return result


async def trace_async_request(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request: httpx.Request,
    send_func: Callable[..., Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    request_hook: _RequestHookT,
    response_hook: _ResponseHookT,
    sem_conv_opt_in_mode: _StabilityMode,
    **kwargs: Any,
) -> httpx.Response:
    method = request.method
    url = remove_url_credentials(str(request.url))

    span_attributes: Dict[str, Any] = {}
    metric_labels: Dict[str, Any] = {}
    set_span_attributes(span_attributes, metric_labels, method, url, sem_conv_opt_in_mode)

    with (
        tracer.start_as_current_span(
            get_default_span_name(method), kind=SpanKind.CLIENT, attributes=span_attributes
        ) as span,
        set_ip_on_next_http_connection(span),
    ):
        exception = None
        set_span_input(span, request)

        if callable(request_hook):
            request_hook(span, request)

        headers = dict(request.headers)
        inject(headers)
        request.headers.update(headers)

        with suppress_http_instrumentation():
            start_time = default_timer()
            try:
                result = await send_func(client, request, **kwargs)
            except Exception as exc:
                exception = exc
                result = getattr(exc, "response", None)
            finally:
                elapsed_time = max(default_timer() - start_time, 0)

        if isinstance(result, httpx.Response):
            span_attributes_response: Dict[str, Any] = {}
            set_http_status_code_attribute(span, result.status_code, metric_labels, sem_conv_opt_in_mode)

            if hasattr(result, "http_version"):
                _set_http_network_protocol_version(metric_labels, result.http_version, sem_conv_opt_in_mode)
                if _report_new(sem_conv_opt_in_mode):
                    _set_http_network_protocol_version(
                        span_attributes_response, result.http_version, sem_conv_opt_in_mode
                    )

            for key, val in span_attributes_response.items():
                span.set_attribute(key, val)

            set_span_output(span, result)

            if callable(response_hook):
                response_hook(span, request, result)

        if exception is not None:
            if _report_new(sem_conv_opt_in_mode):
                span.set_attribute(ERROR_TYPE, type(exception).__qualname__)
                metric_labels[ERROR_TYPE] = type(exception).__qualname__
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
        elif isinstance(result, httpx.Response) and result.status_code >= 500:
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
    _uninstrument_from(httpx.Client)
    _uninstrument_from(httpx.AsyncClient)


def _uninstrument_from(instr_root: Union[type, object], restore_as_bound_func: bool = False) -> None:
    instr_func = getattr(instr_root, "send")
    if not getattr(instr_func, "opentelemetry_instrumentation_httpx_applied", False):
        return
    original = instr_func.__wrapped__
    if restore_as_bound_func:
        original = types.MethodType(original, instr_root)
    setattr(instr_root, "send", original)
