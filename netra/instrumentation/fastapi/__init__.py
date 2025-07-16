from __future__ import annotations

import functools
import logging
import types
from typing import Any, Collection, Dict, Iterable, Optional, Union

import fastapi
import httpx
from fastapi import HTTPException
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.asgi.types import (
    ClientRequestHook,
    ClientResponseHook,
    ServerRequestHook,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.metrics import MeterProvider, get_meter
from opentelemetry.semconv.attributes.http_attributes import HTTP_ROUTE
from opentelemetry.trace import Span, Status, StatusCode, TracerProvider, get_tracer
from opentelemetry.util.http import (
    get_excluded_urls,
    parse_excluded_urls,
    sanitize_method,
)
from starlette.applications import Starlette
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.routing import Match
from starlette.types import ASGIApp

_excluded_urls_from_env = get_excluded_urls("FASTAPI")
_logger = logging.getLogger(__name__)


class SpanAttributeMonitor:
    """Monitor to track span attribute changes and detect HTTP status codes."""

    def __init__(self, span: Span, error_status_codes: set[int], error_messages: dict[Union[int, range], str]):
        self.span = span
        self.error_status_codes = error_status_codes
        self.error_messages = error_messages
        self.original_set_attribute = span.set_attribute
        self.span.set_attribute = self._monitored_set_attribute

    def _monitored_set_attribute(self, key: str, value: Any) -> None:
        """Monitor set_attribute calls to detect HTTP status codes."""
        # Call the original set_attribute method
        self.original_set_attribute(key, value)

        # Check if this is an HTTP status code attribute
        if key == "http.status_code" and isinstance(value, int):
            if value in self.error_status_codes:
                self._record_error_for_span(value)

    def _record_error_for_span(self, status_code: int) -> None:
        """Record an HTTPException for the given span based on the status code."""
        if not self.span or not self.span.is_recording():
            return

        # Get custom error message if available
        error_message = self._get_error_message(status_code)

        # Create and record the HTTPException
        exception = HTTPException(status_code=status_code, detail=error_message)
        self.span.record_exception(exception)

        # Set span status to error for 5xx errors
        if httpx.codes.is_error(status_code):
            self.span.set_status(Status(StatusCode.ERROR, error_message))

        _logger.debug(f"Recorded HTTPException for HTTP {status_code}: {error_message}")

    def _get_error_message(self, status_code: int) -> str:
        """Get the appropriate error message for a status code."""
        # Check for exact status code match
        if status_code in self.error_messages:
            return self.error_messages[status_code]

        # Check for range matches
        for key, message in self.error_messages.items():
            if isinstance(key, range) and status_code in key:
                return message

        # Default messages based on status code ranges
        if 400 <= status_code < 500:
            return f"Client Error: HTTP {status_code}"
        elif 500 <= status_code < 600:
            return f"Server Error: HTTP {status_code}"
        else:
            return f"HTTP {status_code} Error"


class StatusCodeMonitoringMiddleware:
    """Middleware that monitors spans for HTTP status codes and records exceptions."""

    def __init__(
        self,
        app: ASGIApp,
        error_status_codes: Optional[Iterable[int]] = None,
        error_messages: Optional[Dict[Union[int, range], str]] = None,
    ):
        self.app = app
        self.error_status_codes = set(error_status_codes or range(400, 600))
        self.error_messages = error_messages or {}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Get the current span
        from opentelemetry.trace import get_current_span

        span = get_current_span()

        # Set up span monitoring if we have a valid span
        monitor = None
        if span and span.is_recording():
            monitor = SpanAttributeMonitor(span, self.error_status_codes, self.error_messages)

        try:
            await self.app(scope, receive, send)
        finally:
            # Restore original set_attribute method
            if monitor:
                span.set_attribute = monitor.original_set_attribute


class FastAPIInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for FastAPI that records HTTPExceptions for HTTP error status codes.

    This instrumentor follows the OpenTelemetry FastAPI instrumentation pattern while adding
    the capability to monitor http.status_code attributes and automatically record HTTPExceptions
    for error status codes.
    """

    _original_fastapi: Optional[type[fastapi.FastAPI]] = None

    @staticmethod
    def instrument_app(
        app: fastapi.FastAPI,
        server_request_hook: Optional[ServerRequestHook] = None,
        client_request_hook: Optional[ClientRequestHook] = None,
        client_response_hook: Optional[ClientResponseHook] = None,
        tracer_provider: Optional[TracerProvider] = None,
        meter_provider: Optional[MeterProvider] = None,
        excluded_urls: Optional[str] = None,
        http_capture_headers_server_request: Optional[list[str]] = None,
        http_capture_headers_server_response: Optional[list[str]] = None,
        http_capture_headers_sanitize_fields: Optional[list[str]] = None,
        exclude_spans: Optional[list[str]] = None,
        error_status_codes: Optional[Iterable[int]] = None,
        error_messages: Optional[Dict[Union[int, range], str]] = None,
    ) -> None:
        """Instrument an uninstrumented FastAPI application with status code monitoring.

        Args:
            app: The fastapi ASGI application callable to forward requests to.
            server_request_hook: Optional callback which is called with the server span and ASGI
                          scope object for every incoming request.
            client_request_hook: Optional callback which is called with the internal span, and ASGI
                          scope and event which are sent as dictionaries for when the method receive is called.
            client_response_hook: Optional callback which is called with the internal span, and ASGI
                          scope and event which are sent as dictionaries for when the method send is called.
            tracer_provider: The optional tracer provider to use. If omitted
                the current globally configured one is used.
            meter_provider: The optional meter provider to use. If omitted
                the current globally configured one is used.
            excluded_urls: Optional comma delimited string of regexes to match URLs that should not be traced.
            http_capture_headers_server_request: Optional list of HTTP headers to capture from the request.
            http_capture_headers_server_response: Optional list of HTTP headers to capture from the response.
            http_capture_headers_sanitize_fields: Optional list of HTTP headers to sanitize.
            exclude_spans: Optionally exclude HTTP spans from the trace.
            error_status_codes: Optional iterable of status codes to consider as errors. Defaults to range(400, 600).
            error_messages: Optional dictionary mapping status codes or ranges to custom error messages.
        """
        if not hasattr(app, "_is_instrumented_by_opentelemetry"):
            app._is_instrumented_by_opentelemetry = False

        if not getattr(app, "_is_instrumented_by_opentelemetry", False):
            if excluded_urls is None:
                excluded_urls = _excluded_urls_from_env
            else:
                excluded_urls = parse_excluded_urls(excluded_urls)

            tracer = get_tracer(__name__, "1.0.0", tracer_provider)
            meter = get_meter(__name__, "1.0.0", meter_provider)

            # Instead of using `app.add_middleware` we monkey patch `build_middleware_stack` to insert our middleware
            # as the outermost middleware.
            # This follows the OpenTelemetry FastAPI instrumentation pattern.
            def build_middleware_stack(self: Starlette) -> ASGIApp:
                inner_server_error_middleware: ASGIApp = self._original_build_middleware_stack()

                # Add our status code monitoring middleware
                status_code_middleware = StatusCodeMonitoringMiddleware(
                    inner_server_error_middleware,
                    error_status_codes=error_status_codes,
                    error_messages=error_messages,
                )

                # Add OpenTelemetry middleware
                otel_middleware = OpenTelemetryMiddleware(
                    status_code_middleware,
                    excluded_urls=excluded_urls,
                    default_span_details=_get_default_span_details,
                    server_request_hook=server_request_hook,
                    client_request_hook=client_request_hook,
                    client_response_hook=client_response_hook,
                    tracer=tracer,
                    meter=meter,
                    http_capture_headers_server_request=http_capture_headers_server_request,
                    http_capture_headers_server_response=http_capture_headers_server_response,
                    http_capture_headers_sanitize_fields=http_capture_headers_sanitize_fields,
                    exclude_spans=exclude_spans,
                )

                # Wrap in an outer layer of ServerErrorMiddleware
                if isinstance(inner_server_error_middleware, ServerErrorMiddleware):
                    outer_server_error_middleware = ServerErrorMiddleware(
                        app=otel_middleware,
                    )
                else:
                    outer_server_error_middleware = ServerErrorMiddleware(app=otel_middleware)
                return outer_server_error_middleware

            app._original_build_middleware_stack = app.build_middleware_stack
            app.build_middleware_stack = types.MethodType(
                functools.wraps(app.build_middleware_stack)(build_middleware_stack),
                app,
            )

            app._is_instrumented_by_opentelemetry = True
            if app not in _InstrumentedFastAPI._instrumented_fastapi_apps:
                _InstrumentedFastAPI._instrumented_fastapi_apps.add(app)
        else:
            _logger.warning("Attempting to instrument FastAPI app while already instrumented")

    @staticmethod
    def uninstrument_app(app: fastapi.FastAPI) -> None:
        """Remove instrumentation from a FastAPI app."""
        original_build_middleware_stack = getattr(app, "_original_build_middleware_stack", None)
        if original_build_middleware_stack:
            app.build_middleware_stack = original_build_middleware_stack
            del app._original_build_middleware_stack
        app.middleware_stack = app.build_middleware_stack()
        app._is_instrumented_by_opentelemetry = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return ["fastapi"]

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument all FastAPI applications with status code monitoring."""
        self._original_fastapi = fastapi.FastAPI
        _InstrumentedFastAPI._tracer_provider = kwargs.get("tracer_provider")
        _InstrumentedFastAPI._meter_provider = kwargs.get("meter_provider")
        _InstrumentedFastAPI._excluded_urls = kwargs.get("excluded_urls")
        _InstrumentedFastAPI._server_request_hook = kwargs.get("server_request_hook")
        _InstrumentedFastAPI._client_request_hook = kwargs.get("client_request_hook")
        _InstrumentedFastAPI._client_response_hook = kwargs.get("client_response_hook")
        _InstrumentedFastAPI._http_capture_headers_server_request = kwargs.get("http_capture_headers_server_request")
        _InstrumentedFastAPI._http_capture_headers_server_response = kwargs.get("http_capture_headers_server_response")
        _InstrumentedFastAPI._http_capture_headers_sanitize_fields = kwargs.get("http_capture_headers_sanitize_fields")
        _InstrumentedFastAPI._exclude_spans = kwargs.get("exclude_spans")
        _InstrumentedFastAPI._error_status_codes = kwargs.get("error_status_codes")
        _InstrumentedFastAPI._error_messages = kwargs.get("error_messages")
        fastapi.FastAPI = _InstrumentedFastAPI

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from all FastAPI applications."""
        for instance in _InstrumentedFastAPI._instrumented_fastapi_apps:
            self.uninstrument_app(instance)
        _InstrumentedFastAPI._instrumented_fastapi_apps.clear()
        fastapi.FastAPI = self._original_fastapi


class _InstrumentedFastAPI(fastapi.FastAPI):  # type: ignore[misc]
    """FastAPI class with automatic status code monitoring instrumentation."""

    _tracer_provider: Optional[TracerProvider] = None
    _meter_provider: Optional[MeterProvider] = None
    _excluded_urls: Optional[str] = None
    _server_request_hook: Optional[ServerRequestHook] = None
    _client_request_hook: Optional[ClientRequestHook] = None
    _client_response_hook: Optional[ClientResponseHook] = None
    _http_capture_headers_server_request: Optional[list[str]] = None
    _http_capture_headers_server_response: Optional[list[str]] = None
    _http_capture_headers_sanitize_fields: Optional[list[str]] = None
    _exclude_spans: Optional[list[str]] = None
    _error_status_codes: Optional[Iterable[int]] = None
    _error_messages: Optional[Dict[Union[int, range], str]] = None

    _instrumented_fastapi_apps: set[fastapi.FastAPI] = set()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        FastAPIInstrumentor.instrument_app(
            self,
            server_request_hook=self._server_request_hook,
            client_request_hook=self._client_request_hook,
            client_response_hook=self._client_response_hook,
            tracer_provider=self._tracer_provider,
            meter_provider=self._meter_provider,
            excluded_urls=self._excluded_urls,
            http_capture_headers_server_request=self._http_capture_headers_server_request,
            http_capture_headers_server_response=self._http_capture_headers_server_response,
            http_capture_headers_sanitize_fields=self._http_capture_headers_sanitize_fields,
            exclude_spans=self._exclude_spans,
            error_status_codes=self._error_status_codes,
            error_messages=self._error_messages,
        )
        _InstrumentedFastAPI._instrumented_fastapi_apps.add(self)

    def __del__(self) -> None:
        if self in _InstrumentedFastAPI._instrumented_fastapi_apps:
            _InstrumentedFastAPI._instrumented_fastapi_apps.remove(self)


def _get_route_details(scope: dict[str, Any]) -> Optional[str]:
    """
    Function to retrieve Starlette route from scope.

    Args:
        scope: A Starlette scope
    Returns:
        A string containing the route or None
    """
    app = scope["app"]
    route = None

    for starlette_route in app.routes:
        match, _ = starlette_route.matches(scope)
        if match == Match.FULL:
            route = starlette_route.path
            break
        if match == Match.PARTIAL:
            route = starlette_route.path
    return route


def _get_default_span_details(scope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Callback to retrieve span name and attributes from scope.

    Args:
        scope: A Starlette scope
    Returns:
        A tuple of span name and attributes
    """
    route = _get_route_details(scope)
    method = sanitize_method(scope.get("method", "").strip())
    attributes: dict[str, Any] = {}
    if method == "_OTHER":
        method = "HTTP"
    if route:
        attributes[HTTP_ROUTE] = route
    if method and route:  # http
        span_name = f"{method} {route}"
    elif route:  # websocket
        span_name = route
    else:  # fallback
        span_name = method
    return span_name, attributes
