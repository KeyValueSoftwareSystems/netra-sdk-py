from __future__ import annotations

import asyncio
import functools
import logging
import types
from timeit import default_timer
from typing import Any, Awaitable, Callable, Collection, Dict, Optional, Union
from urllib.parse import urlparse

from aiohttp import ClientRequest, ClientResponse, ClientSession
from aiohttp.typedefs import URL
from opentelemetry.instrumentation._semconv import (
    HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
    HTTP_DURATION_HISTOGRAM_BUCKETS_OLD,
    _client_duration_attrs_new,
    _client_duration_attrs_old,
    _filter_semconv_duration_attrs,
    _get_schema_url,
    _OpenTelemetrySemanticConventionStability,
    _OpenTelemetryStabilitySignalType,
    _report_new,
    _report_old,
    _set_http_host_client,
    _set_http_method,
    _set_http_net_peer_name_client,
    _set_http_network_protocol_version,
    _set_http_peer_port_client,
    _set_http_scheme,
    _set_http_url,
    _set_status,
    _StabilityMode,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    is_http_instrumentation_enabled,
    suppress_http_instrumentation,
)
from opentelemetry.metrics import Histogram, get_meter
from opentelemetry.propagate import inject
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.network_attributes import (
    NETWORK_PEER_ADDRESS,
    NETWORK_PEER_PORT,
)
from opentelemetry.semconv.metrics import MetricInstruments
from opentelemetry.semconv.metrics.http_metrics import (
    HTTP_CLIENT_REQUEST_DURATION,
)
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.span import Span
from opentelemetry.util.http import (
    ExcludeList,
    get_excluded_urls,
    parse_excluded_urls,
    remove_url_credentials,
    sanitize_method,
)
from opentelemetry.util.http.httplib import set_ip_on_next_http_connection

from netra.instrumentation.aiohttp.version import __version__

logger = logging.getLogger(__name__)

# Package info for aiohttp instrumentation
_instruments = ("aiohttp >= 3.0.0",)

_excluded_urls_from_env = get_excluded_urls("AIOHTTP_CLIENT")

_RequestHookT = Optional[Callable[[Span, ClientRequest], Awaitable[None]]]
_ResponseHookT = Optional[Callable[[Span, ClientRequest, ClientResponse], Awaitable[None]]]


def _set_http_status_code_attribute(
    span: Span,
    status_code: Union[int, str],
    metric_attributes: Optional[Dict[str, Any]] = None,
    sem_conv_opt_in_mode: _StabilityMode = _StabilityMode.DEFAULT,
) -> None:
    status_code_str = str(status_code)
    try:
        status_code_int = int(status_code)
    except ValueError:
        status_code_int = -1
    if metric_attributes is None:
        metric_attributes = {}
    _set_status(
        span,
        metric_attributes,
        status_code_int,
        status_code_str,
        server_span=False,
        sem_conv_opt_in_mode=sem_conv_opt_in_mode,
    )


def _instrument(
    tracer: Tracer,
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    request_hook: _RequestHookT = None,
    response_hook: _ResponseHookT = None,
    excluded_urls: Optional[ExcludeList] = None,
    sem_conv_opt_in_mode: _StabilityMode = _StabilityMode.DEFAULT,
) -> None:
    """Enables tracing of all aiohttp client requests."""

    wrapped_request = ClientSession._request

    @functools.wraps(wrapped_request)
    async def instrumented_request(self: ClientSession, method: str, url: Any, **kwargs: Any) -> ClientResponse:
        if excluded_urls and excluded_urls.url_disabled(str(url)):
            return await wrapped_request(self, method, url, **kwargs)

        if not is_http_instrumentation_enabled():
            return await wrapped_request(self, method, url, **kwargs)

        span_name = get_default_span_name(method)

        url_str = str(url)
        url = remove_url_credentials(url_str)

        span_attributes: Dict[str, Any] = {}
        _set_http_method(
            span_attributes,
            method,
            sanitize_method(method),
            sem_conv_opt_in_mode,
        )
        _set_http_url(span_attributes, url, sem_conv_opt_in_mode)

        metric_labels: Dict[str, Any] = {}
        _set_http_method(
            metric_labels,
            method,
            sanitize_method(method),
            sem_conv_opt_in_mode,
        )

        try:
            parsed_url = urlparse(url)
            if parsed_url.scheme:
                if _report_old(sem_conv_opt_in_mode):
                    _set_http_scheme(metric_labels, parsed_url.scheme, sem_conv_opt_in_mode)
            if parsed_url.hostname:
                _set_http_host_client(metric_labels, parsed_url.hostname, sem_conv_opt_in_mode)
                _set_http_net_peer_name_client(metric_labels, parsed_url.hostname, sem_conv_opt_in_mode)
                if _report_new(sem_conv_opt_in_mode):
                    _set_http_host_client(
                        span_attributes,
                        parsed_url.hostname,
                        sem_conv_opt_in_mode,
                    )

                    span_attributes[NETWORK_PEER_ADDRESS] = parsed_url.hostname
            if parsed_url.port:
                _set_http_peer_port_client(metric_labels, parsed_url.port, sem_conv_opt_in_mode)
                if _report_new(sem_conv_opt_in_mode):
                    _set_http_peer_port_client(span_attributes, parsed_url.port, sem_conv_opt_in_mode)
                    span_attributes[NETWORK_PEER_PORT] = parsed_url.port
        except ValueError as error:
            logger.error(error)

        with (
            tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=span_attributes) as span,
            set_ip_on_next_http_connection(span),
        ):
            exception = None
            response = None

            headers = kwargs.get("headers", {})
            if headers is None:
                headers = {}
            if hasattr(headers, "items"):
                headers_dict = dict(headers.items())
            else:
                headers_dict = dict(headers)

            inject(headers_dict)
            kwargs["headers"] = headers_dict

            with suppress_http_instrumentation():
                start_time = default_timer()
                try:
                    response = await wrapped_request(self, method, url_str, **kwargs)  # *** PROCEED

                    # Create a ClientRequest object for hooks

                    request_obj = ClientRequest(
                        method=method,
                        url=URL(url_str),
                        headers=headers_dict,
                        data=kwargs.get("data"),
                        params=kwargs.get("params"),
                    )

                    if callable(request_hook):
                        await request_hook(span, request_obj)

                except Exception as exc:  # pylint: disable=W0703
                    exception = exc
                    response = getattr(exc, "response", None)
                finally:
                    elapsed_time = max(default_timer() - start_time, 0)

            if response is not None:
                span_attributes_response: Dict[str, Any] = {}
                _set_http_status_code_attribute(
                    span,
                    response.status,
                    metric_labels,
                    sem_conv_opt_in_mode,
                )

                # Set HTTP version if available
                if hasattr(response, "version") and response.version:
                    version_text = f"{response.version.major}.{response.version.minor}"
                    _set_http_network_protocol_version(metric_labels, version_text, sem_conv_opt_in_mode)
                    if _report_new(sem_conv_opt_in_mode):
                        _set_http_network_protocol_version(
                            span_attributes_response,
                            version_text,
                            sem_conv_opt_in_mode,
                        )

                for key, val in span_attributes_response.items():
                    span.set_attribute(key, val)

                if callable(response_hook) and "request_obj" in locals():
                    await response_hook(span, request_obj, response)

            if exception is not None and _report_new(sem_conv_opt_in_mode):
                span.set_attribute(ERROR_TYPE, type(exception).__qualname__)
                metric_labels[ERROR_TYPE] = type(exception).__qualname__

            if duration_histogram_old is not None:
                duration_attrs_old = _filter_semconv_duration_attrs(
                    metric_labels,
                    _client_duration_attrs_old,
                    _client_duration_attrs_new,
                    _StabilityMode.DEFAULT,
                )
                duration_histogram_old.record(
                    max(round(elapsed_time * 1000), 0),
                    attributes=duration_attrs_old,
                )
            if duration_histogram_new is not None:
                duration_attrs_new = _filter_semconv_duration_attrs(
                    metric_labels,
                    _client_duration_attrs_old,
                    _client_duration_attrs_new,
                    _StabilityMode.HTTP,
                )
                duration_histogram_new.record(elapsed_time, attributes=duration_attrs_new)

            if exception is not None:
                raise exception.with_traceback(exception.__traceback__)

        return response

    # Store the attribute on the function object directly
    setattr(instrumented_request, "opentelemetry_instrumentation_aiohttp_applied", True)
    ClientSession._request = instrumented_request


def _uninstrument() -> None:
    """Disables instrumentation of :code:`aiohttp` client through this module.

    Note that this only works if no other module also patches aiohttp."""
    _uninstrument_from(ClientSession)


def _uninstrument_from(instr_root: Any, restore_as_bound_func: bool = False) -> None:
    for instr_func_name in ("_request",):
        instr_func = getattr(instr_root, instr_func_name)
        if not getattr(
            instr_func,
            "opentelemetry_instrumentation_aiohttp_applied",
            False,
        ):
            continue

        original = instr_func.__wrapped__  # pylint:disable=no-member
        if restore_as_bound_func:
            original = types.MethodType(original, instr_root)
        setattr(instr_root, instr_func_name, original)


def get_default_span_name(method: str) -> str:
    """
    Default implementation for name_callback, returns HTTP {method_name}.
    https://opentelemetry.io/docs/reference/specification/trace/semantic_conventions/http/#name

    Args:
        method: string representing HTTP method
    Returns:
        span name
    """
    method = sanitize_method(method.strip())
    if method == "_OTHER":
        return "HTTP"
    return method


class AioHttpClientInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for aiohttp client
    See `BaseInstrumentor`
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instruments aiohttp client module

        Args:
            **kwargs: Optional arguments
                ``tracer_provider``: a TracerProvider, defaults to global
                ``request_hook``: An optional async callback that is invoked right after a span is created.
                ``response_hook``: An optional async callback which is invoked right before the span is finished processing a response.
                ``excluded_urls``: A string containing a comma-delimited list of regexes used to exclude URLs from tracking
                ``duration_histogram_boundaries``: A list of float values representing the explicit bucket boundaries for the duration histogram.
        """
        semconv_opt_in_mode = _OpenTelemetrySemanticConventionStability._get_opentelemetry_stability_opt_in_mode(
            _OpenTelemetryStabilitySignalType.HTTP,
        )
        schema_url = _get_schema_url(semconv_opt_in_mode)
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=schema_url,
        )
        excluded_urls = kwargs.get("excluded_urls")
        meter_provider = kwargs.get("meter_provider")
        duration_histogram_boundaries = kwargs.get("duration_histogram_boundaries")
        meter = get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=schema_url,
        )
        duration_histogram_old = None
        if _report_old(semconv_opt_in_mode):
            duration_histogram_old = meter.create_histogram(
                name=MetricInstruments.HTTP_CLIENT_DURATION,
                unit="ms",
                description="measures the duration of the outbound HTTP request",
                explicit_bucket_boundaries_advisory=duration_histogram_boundaries
                or HTTP_DURATION_HISTOGRAM_BUCKETS_OLD,
            )
        duration_histogram_new = None
        if _report_new(semconv_opt_in_mode):
            duration_histogram_new = meter.create_histogram(
                name=HTTP_CLIENT_REQUEST_DURATION,
                unit="s",
                description="Duration of HTTP client requests.",
                explicit_bucket_boundaries_advisory=duration_histogram_boundaries
                or HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
            )
        _instrument(
            tracer,
            duration_histogram_old,
            duration_histogram_new,
            request_hook=kwargs.get("request_hook"),
            response_hook=kwargs.get("response_hook"),
            excluded_urls=(_excluded_urls_from_env if excluded_urls is None else parse_excluded_urls(excluded_urls)),
            sem_conv_opt_in_mode=semconv_opt_in_mode,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        _uninstrument()

    @staticmethod
    def uninstrument_session(session: ClientSession) -> None:
        """Disables instrumentation on the session object."""
        _uninstrument_from(session, restore_as_bound_func=True)
