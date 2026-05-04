"""
Unit tests for NetraFastAPIInstrumentor and NetraFastAPIMiddleware.
Tests focusing on core functionality, single-span middleware, and input/output capture.
"""

import json
from typing import Collection
from unittest.mock import AsyncMock, Mock, patch

import fastapi
from fastapi import HTTPException
from opentelemetry.trace import Span, StatusCode
from starlette.routing import Match

from netra.instrumentation.fastapi import (
    NetraFastAPIInstrumentor,
    _InstrumentedFastAPI,
    get_default_span_details,
    get_route_details,
)
from netra.instrumentation.fastapi.middleware import (
    DEFAULT_ERROR_STATUS_CODE_RANGE,
    NetraFastAPIMiddleware,
    _asgi_getter,
)
from netra.instrumentation.fastapi.utils import (
    _SENSITIVE_HEADERS,
    build_request_url,
    get_error_message,
    parse_body,
    sanitize_headers,
    set_span_input,
    set_span_output,
)
from netra.instrumentation.fastapi.version import __version__


class TestNetraFastAPIInstrumentor:
    """Test NetraFastAPIInstrumentor core functionality."""

    def test_initialization(self) -> None:
        """Test NetraFastAPIInstrumentor initialization."""
        instrumentor = NetraFastAPIInstrumentor()

        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self) -> None:
        """Test instrumentation_dependencies returns correct packages."""
        instrumentor = NetraFastAPIInstrumentor()

        dependencies = instrumentor.instrumentation_dependencies()

        assert isinstance(dependencies, Collection)
        assert any("fastapi" in dep for dep in dependencies)

    @patch("netra.instrumentation.fastapi.get_tracer")
    def test_instrument_app_with_default_parameters(self, mock_get_tracer: Mock) -> None:
        """Test instrument_app method with default parameters."""
        app = fastapi.FastAPI()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        NetraFastAPIInstrumentor.instrument_app(app)

        assert app._is_instrumented_by_opentelemetry is True
        assert hasattr(app, "_original_build_middleware_stack")
        mock_get_tracer.assert_called_once()

    @patch("netra.instrumentation.fastapi.get_tracer")
    def test_instrument_app_with_custom_parameters(self, mock_get_tracer: Mock) -> None:
        """Test instrument_app method with custom parameters."""
        app = fastapi.FastAPI()
        mock_tracer_provider = Mock()
        error_status_codes = [404, 500]
        error_messages = {404: "Not Found", 500: "Internal Server Error"}
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        NetraFastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=mock_tracer_provider,
            error_status_codes=error_status_codes,
            error_messages=error_messages,
        )

        assert app._is_instrumented_by_opentelemetry is True
        mock_get_tracer.assert_called_once_with("netra.instrumentation.fastapi", __version__, mock_tracer_provider)

    def test_instrument_app_already_instrumented(self) -> None:
        """Test instrument_app logs warning when app is already instrumented."""
        app = fastapi.FastAPI()
        app._is_instrumented_by_opentelemetry = True

        with patch("netra.instrumentation.fastapi.logger") as mock_logger:
            NetraFastAPIInstrumentor.instrument_app(app)
            mock_logger.warning.assert_called_once()

    def test_uninstrument_app(self) -> None:
        """Test uninstrument_app removes instrumentation."""
        app = fastapi.FastAPI()
        original_build = app.build_middleware_stack
        app._original_build_middleware_stack = original_build
        app._is_instrumented_by_opentelemetry = True

        NetraFastAPIInstrumentor.uninstrument_app(app)

        assert app._is_instrumented_by_opentelemetry is False
        assert not hasattr(app, "_original_build_middleware_stack")
        assert app.build_middleware_stack == original_build

    @patch("netra.instrumentation.fastapi.fastapi.FastAPI")
    def test_instrument_patches_fastapi_class(self, mock_fastapi_class: Mock) -> None:
        """Test _instrument method patches FastAPI class."""
        instrumentor = NetraFastAPIInstrumentor()
        instrumentor._original_fastapi = fastapi.FastAPI

        instrumentor._instrument()

        assert fastapi.FastAPI == _InstrumentedFastAPI

    @patch("netra.instrumentation.fastapi.fastapi.FastAPI")
    def test_uninstrument_restores_original_fastapi(self, mock_fastapi_class: Mock) -> None:
        """Test _uninstrument method restores original FastAPI class."""
        instrumentor = NetraFastAPIInstrumentor()
        original_fastapi = Mock()
        instrumentor._original_fastapi = original_fastapi

        instrumentor._uninstrument()

        assert fastapi.FastAPI == original_fastapi


class TestNetraFastAPIMiddleware:
    """Test NetraFastAPIMiddleware functionality."""

    def test_initialization(self) -> None:
        """Test middleware initialization with defaults."""
        mock_app = Mock()
        mock_tracer = Mock()

        middleware = NetraFastAPIMiddleware(mock_app, mock_tracer)

        assert middleware.app is mock_app
        assert middleware.tracer is mock_tracer
        assert middleware.excluded_urls is None
        assert middleware.error_status_codes == set(DEFAULT_ERROR_STATUS_CODE_RANGE)
        assert middleware.error_messages == {}

    def test_initialization_with_custom_error_codes(self) -> None:
        """Test middleware initialization with custom error status codes."""
        mock_app = Mock()
        mock_tracer = Mock()
        error_codes = [404, 500, 503]
        error_messages = {404: "Not Found"}

        middleware = NetraFastAPIMiddleware(
            mock_app,
            mock_tracer,
            error_status_codes=error_codes,
            error_messages=error_messages,
        )

        assert middleware.error_status_codes == {404, 500, 503}
        assert middleware.error_messages == {404: "Not Found"}

    def test_is_url_excluded_no_exclusion_list(self) -> None:
        """Test URL exclusion when no exclusion list is set."""
        middleware = NetraFastAPIMiddleware(Mock(), Mock())
        scope = {"path": "/health"}

        assert middleware._is_url_excluded(scope) is False

    def test_is_url_excluded_with_exclusion(self) -> None:
        """Test URL exclusion with an active exclusion list."""
        mock_excluded = Mock()
        mock_excluded.url_disabled.return_value = True
        middleware = NetraFastAPIMiddleware(Mock(), Mock(), excluded_urls=mock_excluded)

        assert middleware._is_url_excluded({"path": "/health"}) is True
        mock_excluded.url_disabled.assert_called_once_with("/health")

    def test_build_span_attributes(self) -> None:
        """Test span attribute extraction from ASGI scope."""
        scope = {
            "method": "POST",
            "scheme": "https",
            "path": "/api/users",
            "query_string": b"page=1",
            "http_version": "1.1",
            "server": ("0.0.0.0", 8000),
            "client": ("127.0.0.1", 54321),
            "headers": [(b"user-agent", b"test-client/1.0")],
        }

        attrs = NetraFastAPIMiddleware._build_span_attributes(scope)

        assert attrs["http.method"] == "POST"
        assert attrs["http.scheme"] == "https"
        assert attrs["http.target"] == "/api/users"
        assert attrs["http.flavor"] == "1.1"
        assert attrs["http.host"] == "0.0.0.0:8000"
        assert attrs["net.host.port"] == "8000"
        assert attrs["net.peer.ip"] == "127.0.0.1"
        assert attrs["net.peer.port"] == "54321"
        assert attrs["http.user_agent"] == "test-client/1.0"

    def test_finalize_span_status_ok(self) -> None:
        """Test span finalization with a 200 OK status."""
        mock_span = Mock(spec=Span)
        middleware = NetraFastAPIMiddleware(Mock(), Mock())

        middleware._finalize_span_status(mock_span, 200, None)

        mock_span.set_status.assert_called_once()
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.OK

    def test_finalize_span_status_server_error(self) -> None:
        """Test span finalization with a 500 server error."""
        mock_span = Mock(spec=Span)
        middleware = NetraFastAPIMiddleware(Mock(), Mock())

        middleware._finalize_span_status(mock_span, 500, None)

        mock_span.record_exception.assert_called_once()
        recorded = mock_span.record_exception.call_args[0][0]
        assert isinstance(recorded, HTTPException)
        assert recorded.status_code == 500

        mock_span.set_status.assert_called_once()
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.ERROR

    def test_finalize_span_status_client_error(self) -> None:
        """Test span finalization with a 404 client error."""
        mock_span = Mock(spec=Span)
        middleware = NetraFastAPIMiddleware(Mock(), Mock())

        middleware._finalize_span_status(mock_span, 404, None)

        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.OK

    def test_finalize_span_status_with_exception(self) -> None:
        """Test span finalization when an application exception occurred."""
        mock_span = Mock(spec=Span)
        middleware = NetraFastAPIMiddleware(Mock(), Mock())
        exc = RuntimeError("test error")

        middleware._finalize_span_status(mock_span, 500, exc)

        mock_span.record_exception.assert_called_once_with(exc)
        status = mock_span.set_status.call_args[0][0]
        assert status.status_code == StatusCode.ERROR


class TestInstrumentedFastAPI:
    """Test _InstrumentedFastAPI class functionality."""

    @patch("netra.instrumentation.fastapi.NetraFastAPIInstrumentor.instrument_app")
    def test_initialization(self, mock_instrument_app: Mock) -> None:
        """Test _InstrumentedFastAPI initialization."""
        app = _InstrumentedFastAPI()

        mock_instrument_app.assert_called_once_with(
            app,
            tracer_provider=None,
            excluded_urls=None,
            error_status_codes=None,
            error_messages=None,
        )
        assert app in _InstrumentedFastAPI._instrumented_fastapi_apps

    def test_weakset_allows_gc(self) -> None:
        """Test that _instrumented_fastapi_apps uses WeakSet so apps can be GC'd."""
        import gc
        import weakref

        assert isinstance(_InstrumentedFastAPI._instrumented_fastapi_apps, weakref.WeakSet)

        with patch("netra.instrumentation.fastapi.NetraFastAPIInstrumentor.instrument_app"):
            app = _InstrumentedFastAPI()
            assert app in _InstrumentedFastAPI._instrumented_fastapi_apps
            count_before = len(_InstrumentedFastAPI._instrumented_fastapi_apps)

        del app
        gc.collect()

        assert len(_InstrumentedFastAPI._instrumented_fastapi_apps) < count_before


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_route_details_full_match(self) -> None:
        """Test get_route_details with full route match."""
        mock_route = Mock()
        mock_route.matches.return_value = (Match.FULL, None)
        mock_route.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route]

        scope = {"app": mock_app}

        result = get_route_details(scope)

        assert result == "/test/path"

    def test_get_route_details_partial_match(self) -> None:
        """Test get_route_details with partial route match."""
        mock_route1 = Mock()
        mock_route1.matches.return_value = (Match.NONE, None)
        mock_route1.path = "/other/path"

        mock_route2 = Mock()
        mock_route2.matches.return_value = (Match.PARTIAL, None)
        mock_route2.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route1, mock_route2]

        scope = {"app": mock_app}

        result = get_route_details(scope)

        assert result == "/test/path"

    def test_get_route_details_no_match(self) -> None:
        """Test get_route_details with no route match."""
        mock_route = Mock()
        mock_route.matches.return_value = (Match.NONE, None)
        mock_route.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route]

        scope = {"app": mock_app}

        result = get_route_details(scope)

        assert result is None

    @patch("netra.instrumentation.fastapi.utils.get_route_details")
    @patch("netra.instrumentation.fastapi.utils.sanitize_method")
    def test_get_default_span_details_with_route_and_method(
        self, mock_sanitize_method: Mock, mock_get_route_details: Mock
    ) -> None:
        """Test get_default_span_details with route and method."""
        mock_get_route_details.return_value = "/test/path"
        mock_sanitize_method.return_value = "GET"
        scope = {"method": "GET"}

        span_name, attributes = get_default_span_details(scope)

        assert span_name == "GET /test/path"
        assert attributes["http.route"] == "/test/path"

    @patch("netra.instrumentation.fastapi.utils.get_route_details")
    @patch("netra.instrumentation.fastapi.utils.sanitize_method")
    def test_get_default_span_details_no_route(self, mock_sanitize_method: Mock, mock_get_route_details: Mock) -> None:
        """Test get_default_span_details with no route."""
        mock_get_route_details.return_value = None
        mock_sanitize_method.return_value = "GET"
        scope = {"method": "GET"}

        span_name, attributes = get_default_span_details(scope)

        assert span_name == "GET"
        assert attributes == {}

    @patch("netra.instrumentation.fastapi.utils.get_route_details")
    @patch("netra.instrumentation.fastapi.utils.sanitize_method")
    def test_get_default_span_details_other_method(
        self, mock_sanitize_method: Mock, mock_get_route_details: Mock
    ) -> None:
        """Test get_default_span_details with _OTHER method."""
        mock_get_route_details.return_value = "/test/path"
        mock_sanitize_method.return_value = "_OTHER"
        scope = {"method": "CUSTOM"}

        span_name, attributes = get_default_span_details(scope)

        assert span_name == "HTTP /test/path"
        assert attributes["http.route"] == "/test/path"


class TestHeaderSanitization:
    """Test header sanitization utilities."""

    def test_sanitize_headers_redacts_sensitive(self) -> None:
        """Test that sensitive headers are redacted."""
        raw_headers = [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer secret-token"),
            (b"x-api-key", b"my-api-key"),
            (b"accept", b"*/*"),
        ]

        result = sanitize_headers(raw_headers)

        assert result["content-type"] == "application/json"
        assert result["authorization"] == "[REDACTED]"
        assert result["x-api-key"] == "[REDACTED]"
        assert result["accept"] == "*/*"

    def test_sanitize_headers_all_sensitive_covered(self) -> None:
        """Test that all declared sensitive headers are redacted."""
        for header_name in _SENSITIVE_HEADERS:
            raw_headers = [(header_name.encode(), b"secret-value")]
            result = sanitize_headers(raw_headers)
            assert result[header_name] == "[REDACTED]"


class TestUrlBuilder:
    """Test URL construction from ASGI scope."""

    def test_build_request_url_basic(self) -> None:
        """Test basic URL construction."""
        scope = {
            "scheme": "http",
            "server": ("localhost", 8000),
            "path": "/hello",
            "query_string": b"",
        }

        assert build_request_url(scope) == "http://localhost:8000/hello"

    def test_build_request_url_with_query(self) -> None:
        """Test URL construction with query string."""
        scope = {
            "scheme": "http",
            "server": ("localhost", 8000),
            "path": "/search",
            "query_string": b"q=test&page=1",
        }

        assert build_request_url(scope) == "http://localhost:8000/search?q=test&page=1"

    def test_build_request_url_default_port(self) -> None:
        """Test URL construction with default port omitted."""
        scope = {
            "scheme": "http",
            "server": ("example.com", 80),
            "path": "/",
            "query_string": b"",
        }

        assert build_request_url(scope) == "http://example.com/"

    def test_build_request_url_https_default_port(self) -> None:
        """Test URL construction with HTTPS default port omitted."""
        scope = {
            "scheme": "https",
            "server": ("example.com", 443),
            "path": "/secure",
            "query_string": b"",
        }

        assert build_request_url(scope) == "https://example.com/secure"

    def test_build_request_url_no_server(self) -> None:
        """Test URL construction when no server info is available."""
        scope = {"path": "/fallback", "query_string": b""}

        assert build_request_url(scope) == "/fallback"


class TestBodyParsing:
    """Test body parsing utilities."""

    def test_parse_body_json(self) -> None:
        """Test parsing a JSON body."""
        raw = b'{"key": "value"}'
        result = parse_body(raw)
        assert result == {"key": "value"}

    def test_parse_body_text(self) -> None:
        """Test parsing a plain text body."""
        raw = b"Hello, world!"
        result = parse_body(raw)
        assert result == "Hello, world!"

    def test_parse_body_empty(self) -> None:
        """Test parsing an empty body."""
        assert parse_body(b"") is None

    def test_parse_body_binary(self) -> None:
        """Test parsing binary content."""
        raw = bytes(range(256))
        result = parse_body(raw)
        assert "<binary content:" in result


class TestSpanInputOutput:
    """Test span input/output attribute setting."""

    def test_set_span_input(self) -> None:
        """Test setting the input attribute on a span."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True
        scope = {
            "scheme": "http",
            "server": ("localhost", 8000),
            "path": "/api/test",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer token"),
            ],
        }
        body = b'{"name": "test"}'

        set_span_input(mock_span, scope, body)

        mock_span.set_attribute.assert_called_once()
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "input"
        input_data = json.loads(call_args[0][1])
        assert input_data["url"] == "http://localhost:8000/api/test"
        assert input_data["headers"]["authorization"] == "[REDACTED]"
        assert input_data["body"] == {"name": "test"}

    def test_set_span_input_no_body(self) -> None:
        """Test setting input when there is no request body."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True
        scope = {
            "scheme": "http",
            "server": ("localhost", 8000),
            "path": "/hello",
            "query_string": b"",
            "headers": [],
        }

        set_span_input(mock_span, scope, b"")

        call_args = mock_span.set_attribute.call_args
        input_data = json.loads(call_args[0][1])
        assert "body" not in input_data

    def test_set_span_output(self) -> None:
        """Test setting the output attribute on a span."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True
        headers = [(b"content-type", b"application/json")]
        body = b'{"result": "ok"}'

        set_span_output(mock_span, 200, headers, body)

        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "output"
        output_data = json.loads(call_args[0][1])
        assert output_data["status_code"] == 200
        assert output_data["headers"]["content-type"] == "application/json"
        assert output_data["body"] == {"result": "ok"}

    def test_set_span_output_empty_body(self) -> None:
        """Test setting output when there is no response body."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True

        set_span_output(mock_span, 204, [], b"")

        call_args = mock_span.set_attribute.call_args
        output_data = json.loads(call_args[0][1])
        assert output_data["status_code"] == 204
        assert "body" not in output_data

    def test_set_span_input_not_recording(self) -> None:
        """Test that input is not set when span is not recording."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = False

        set_span_input(mock_span, {}, b"body")

        mock_span.set_attribute.assert_not_called()

    def test_set_span_output_not_recording(self) -> None:
        """Test that output is not set when span is not recording."""
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = False

        set_span_output(mock_span, 200, [], b"body")

        mock_span.set_attribute.assert_not_called()


class TestErrorMessages:
    """Test error message resolution."""

    def test_exact_match(self) -> None:
        """Test exact status code match."""
        result = get_error_message(404, {404: "Custom Not Found"})
        assert result == "Custom Not Found"

    def test_range_match(self) -> None:
        """Test range-based status code match."""
        result = get_error_message(404, {range(400, 500): "Client Error Range"})
        assert result == "Client Error Range"

    def test_default_client_error(self) -> None:
        """Test default message for client errors."""
        result = get_error_message(404, {})
        assert result == "Client Error: HTTP 404"

    def test_default_server_error(self) -> None:
        """Test default message for server errors."""
        result = get_error_message(500, {})
        assert result == "Server Error: HTTP 500"

    def test_default_other_error(self) -> None:
        """Test default message for non-standard error codes."""
        result = get_error_message(302, {})
        assert result == "HTTP 302 Error"


class TestASGIGetter:
    """Test _ASGIGetter for W3C Trace Context-compliant header extraction."""

    def test_get_single_traceparent(self) -> None:
        """Test extracting a single traceparent header."""
        scope = {
            "headers": [
                (b"traceparent", b"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
            ]
        }

        values = _asgi_getter.get(scope, "traceparent")

        assert values == ["00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"]

    def test_get_single_tracestate(self) -> None:
        """Test extracting a single tracestate header."""
        scope = {
            "headers": [
                (b"tracestate", b"congo=t61rcWkgMzE,rojo=00f067aa0ba902b7"),
            ]
        }

        values = _asgi_getter.get(scope, "tracestate")

        assert values == ["congo=t61rcWkgMzE,rojo=00f067aa0ba902b7"]

    def test_get_multiple_tracestate_headers(self) -> None:
        """Test extracting multiple tracestate headers per RFC 7230 Section 3.2.2."""
        scope = {
            "headers": [
                (b"tracestate", b"congo=t61rcWkgMzE"),
                (b"tracestate", b"rojo=00f067aa0ba902b7"),
            ]
        }

        values = _asgi_getter.get(scope, "tracestate")

        assert values is not None
        assert len(values) == 2
        assert "congo=t61rcWkgMzE" in values
        assert "rojo=00f067aa0ba902b7" in values

    def test_get_case_insensitive(self) -> None:
        """Test case-insensitive header lookup per W3C spec."""
        scope = {
            "headers": [
                (b"Traceparent", b"00-abc-def-01"),
            ]
        }

        values = _asgi_getter.get(scope, "traceparent")

        assert values == ["00-abc-def-01"]

    def test_get_missing_header(self) -> None:
        """Test that missing headers return None."""
        scope = {"headers": [(b"content-type", b"text/html")]}

        values = _asgi_getter.get(scope, "traceparent")

        assert values is None

    def test_get_empty_headers(self) -> None:
        """Test that an empty headers list returns None."""
        scope = {"headers": []}

        assert _asgi_getter.get(scope, "traceparent") is None

    def test_get_no_headers_key(self) -> None:
        """Test that a scope with no headers key returns None."""
        scope = {}

        assert _asgi_getter.get(scope, "traceparent") is None

    def test_keys_returns_unique_lowercase(self) -> None:
        """Test that keys() returns unique lower-cased header names."""
        scope = {
            "headers": [
                (b"Traceparent", b"value1"),
                (b"tracestate", b"value2"),
                (b"tracestate", b"value3"),
                (b"Content-Type", b"text/html"),
            ]
        }

        keys = _asgi_getter.keys(scope)

        assert set(keys) == {"traceparent", "tracestate", "content-type"}

    def test_keys_empty_scope(self) -> None:
        """Test keys() with empty scope."""
        assert _asgi_getter.keys({}) == []


class TestW3CTracePropagation:
    """Test W3C Trace Context propagation through the middleware."""

    def test_extract_called_with_asgi_getter(self) -> None:
        """Test that extract() is called with the scope and ASGI getter."""
        import asyncio

        mock_app = AsyncMock()
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=False)
        mock_span.is_recording.return_value = True
        mock_tracer.start_as_current_span.return_value = mock_span

        middleware = NetraFastAPIMiddleware(mock_app, mock_tracer)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "scheme": "http",
            "http_version": "1.1",
            "server": ("localhost", 8000),
            "headers": [
                (b"traceparent", b"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
                (b"tracestate", b"congo=t61rcWkgMzE"),
            ],
        }

        async def mock_send(message: dict) -> None:
            pass

        async def mock_receive() -> dict:
            return {"type": "http.request", "body": b""}

        with (
            patch("netra.instrumentation.fastapi.middleware.extract") as mock_extract,
            patch("netra.instrumentation.fastapi.middleware.context_api") as mock_ctx,
        ):
            mock_extract.return_value = Mock()
            mock_ctx.attach.return_value = Mock()

            asyncio.get_event_loop().run_until_complete(middleware(scope, mock_receive, mock_send))

            mock_extract.assert_called_once_with(carrier=scope, getter=_asgi_getter)

    def test_traceparent_format_parsing(self) -> None:
        """Test that a well-formed traceparent header is correctly parsed by the getter."""
        traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        scope = {"headers": [(b"traceparent", traceparent.encode("latin-1"))]}

        values = _asgi_getter.get(scope, "traceparent")

        assert values is not None
        assert len(values) == 1
        parts = values[0].split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert len(parts[3]) == 2

    def test_traceparent_and_tracestate_together(self) -> None:
        """Test extraction of both traceparent and tracestate headers."""
        scope = {
            "headers": [
                (b"traceparent", b"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
                (b"tracestate", b"rojo=00f067aa0ba902b7,congo=t61rcWkgMzE"),
            ]
        }

        tp = _asgi_getter.get(scope, "traceparent")
        ts = _asgi_getter.get(scope, "tracestate")

        assert tp == ["00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"]
        assert ts == ["rojo=00f067aa0ba902b7,congo=t61rcWkgMzE"]

    def test_non_http_scope_skips_propagation(self) -> None:
        """Test that non-HTTP scopes bypass trace propagation entirely."""
        import asyncio

        mock_app = AsyncMock()
        mock_tracer = Mock()
        middleware = NetraFastAPIMiddleware(mock_app, mock_tracer)

        scope = {"type": "websocket", "headers": []}

        asyncio.get_event_loop().run_until_complete(middleware(scope, AsyncMock(), AsyncMock()))

        mock_app.assert_called_once()
        mock_tracer.start_as_current_span.assert_not_called()
