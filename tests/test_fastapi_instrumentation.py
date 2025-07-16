"""
Unit tests for FastAPIInstrumentor class.
Tests focusing on core functionality, middleware behavior, and span monitoring.
"""

from typing import Collection
from unittest.mock import Mock, patch

import fastapi
from fastapi import HTTPException
from opentelemetry.trace import Span, StatusCode
from starlette.routing import Match

from netra.instrumentation.fastapi import (  # Replace with actual module name
    FastAPIInstrumentor,
    SpanAttributeMonitor,
    StatusCodeMonitoringMiddleware,
    _get_default_span_details,
    _get_route_details,
    _InstrumentedFastAPI,
)


class TestFastAPIInstrumentor:
    """Test FastAPIInstrumentor core functionality."""

    def test_initialization(self):
        """Test FastAPIInstrumentor initialization."""
        # Act
        instrumentor = FastAPIInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = FastAPIInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "fastapi" in dependencies

    @patch("netra.instrumentation.fastapi.get_tracer")
    @patch("netra.instrumentation.fastapi.get_meter")
    @patch("netra.instrumentation.fastapi.OpenTelemetryMiddleware")
    def test_instrument_app_with_default_parameters(self, mock_otel_middleware, mock_get_meter, mock_get_tracer):
        """Test instrument_app method with default parameters."""
        # Arrange
        app = fastapi.FastAPI()
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter

        # Act
        FastAPIInstrumentor.instrument_app(app)

        # Assert
        assert app._is_instrumented_by_opentelemetry is True
        assert hasattr(app, "_original_build_middleware_stack")
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()

    @patch("netra.instrumentation.fastapi.get_tracer")
    @patch("netra.instrumentation.fastapi.get_meter")
    @patch("netra.instrumentation.fastapi.OpenTelemetryMiddleware")
    def test_instrument_app_with_custom_parameters(self, mock_otel_middleware, mock_get_meter, mock_get_tracer):
        """Test instrument_app method with custom parameters."""
        # Arrange
        app = fastapi.FastAPI()
        mock_tracer_provider = Mock()
        mock_meter_provider = Mock()
        error_status_codes = [404, 500]
        error_messages = {404: "Not Found", 500: "Internal Server Error"}
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter

        # Act
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=mock_tracer_provider,
            meter_provider=mock_meter_provider,
            error_status_codes=error_status_codes,
            error_messages=error_messages,
        )

        # Assert
        assert app._is_instrumented_by_opentelemetry is True
        mock_get_tracer.assert_called_once_with("netra.instrumentation.fastapi", "1.0.0", mock_tracer_provider)
        mock_get_meter.assert_called_once_with("netra.instrumentation.fastapi", "1.0.0", mock_meter_provider)

    def test_instrument_app_already_instrumented(self):
        """Test instrument_app logs warning when app is already instrumented."""
        # Arrange
        app = fastapi.FastAPI()
        app._is_instrumented_by_opentelemetry = True

        # Act & Assert
        with patch("netra.instrumentation.fastapi._logger") as mock_logger:
            FastAPIInstrumentor.instrument_app(app)
            mock_logger.warning.assert_called_once()

    def test_uninstrument_app(self):
        """Test uninstrument_app removes instrumentation."""
        # Arrange
        app = fastapi.FastAPI()
        original_build = app.build_middleware_stack
        app._original_build_middleware_stack = original_build
        app._is_instrumented_by_opentelemetry = True

        # Act
        FastAPIInstrumentor.uninstrument_app(app)

        # Assert
        assert app._is_instrumented_by_opentelemetry is False
        assert not hasattr(app, "_original_build_middleware_stack")
        assert app.build_middleware_stack == original_build

    @patch("netra.instrumentation.fastapi.fastapi.FastAPI")
    def test_instrument_patches_fastapi_class(self, mock_fastapi_class):
        """Test _instrument method patches FastAPI class."""
        # Arrange
        instrumentor = FastAPIInstrumentor()
        instrumentor._original_fastapi = fastapi.FastAPI

        # Act
        instrumentor._instrument()

        # Assert
        assert fastapi.FastAPI == _InstrumentedFastAPI

    @patch("netra.instrumentation.fastapi.fastapi.FastAPI")
    def test_uninstrument_restores_original_fastapi(self, mock_fastapi_class):
        """Test _uninstrument method restores original FastAPI class."""
        # Arrange
        instrumentor = FastAPIInstrumentor()
        original_fastapi = Mock()
        instrumentor._original_fastapi = original_fastapi

        # Act
        instrumentor._uninstrument()

        # Assert
        assert fastapi.FastAPI == original_fastapi


class TestSpanAttributeMonitor:
    """Test SpanAttributeMonitor functionality."""

    def test_initialization(self):
        """Test SpanAttributeMonitor initialization."""
        # Arrange
        mock_span = Mock(spec=Span)
        original_set_attribute = mock_span.set_attribute
        error_status_codes = {404, 500}
        error_messages = {404: "Not Found"}

        # Act
        monitor = SpanAttributeMonitor(mock_span, error_status_codes, error_messages)

        # Assert
        assert monitor.span == mock_span
        assert monitor.error_status_codes == error_status_codes
        assert monitor.error_messages == error_messages
        assert monitor.original_set_attribute == original_set_attribute
        # Verify that set_attribute was replaced with monitored version
        assert mock_span.set_attribute == monitor._monitored_set_attribute

    def test_monitored_set_attribute_normal_attribute(self):
        """Test _monitored_set_attribute with normal attribute."""
        # Arrange
        mock_span = Mock(spec=Span)
        original_set_attribute = mock_span.set_attribute
        monitor = SpanAttributeMonitor(mock_span, {404}, {})

        # Act
        monitor._monitored_set_attribute("test.attribute", "value")

        # Assert
        original_set_attribute.assert_called_once_with("test.attribute", "value")

    def test_monitored_set_attribute_http_status_code_error(self):
        """Test _monitored_set_attribute with HTTP status code that triggers error."""
        # Arrange
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True
        original_set_attribute = mock_span.set_attribute
        error_status_codes = {404}
        error_messages = {404: "Not Found"}
        monitor = SpanAttributeMonitor(mock_span, error_status_codes, error_messages)

        # Act
        monitor._monitored_set_attribute("http.status_code", 404)

        # Assert
        original_set_attribute.assert_called_once_with("http.status_code", 404)
        mock_span.record_exception.assert_called_once()
        # Check that HTTPException was recorded
        recorded_exception = mock_span.record_exception.call_args[0][0]
        assert isinstance(recorded_exception, HTTPException)
        assert recorded_exception.status_code == 404
        assert recorded_exception.detail == "Not Found"

    def test_monitored_set_attribute_http_status_code_no_error(self):
        """Test _monitored_set_attribute with HTTP status code that doesn't trigger error."""
        # Arrange
        mock_span = Mock(spec=Span)
        original_set_attribute = mock_span.set_attribute
        monitor = SpanAttributeMonitor(mock_span, {404}, {})

        # Act
        monitor._monitored_set_attribute("http.status_code", 200)

        # Assert
        original_set_attribute.assert_called_once_with("http.status_code", 200)
        mock_span.record_exception.assert_not_called()

    @patch("netra.instrumentation.fastapi.httpx.codes.is_error")
    def test_record_error_for_span_server_error(self, mock_is_error):
        """Test _record_error_for_span with server error status code."""
        # Arrange
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = True
        mock_is_error.return_value = True
        monitor = SpanAttributeMonitor(mock_span, {500}, {})

        # Act
        monitor._record_error_for_span(500)

        # Assert
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR

    def test_record_error_for_span_span_not_recording(self):
        """Test _record_error_for_span when span is not recording."""
        # Arrange
        mock_span = Mock(spec=Span)
        mock_span.is_recording.return_value = False
        monitor = SpanAttributeMonitor(mock_span, {500}, {})

        # Act
        monitor._record_error_for_span(500)

        # Assert
        mock_span.record_exception.assert_not_called()

    def test_get_error_message_exact_match(self):
        """Test _get_error_message with exact status code match."""
        # Arrange
        mock_span = Mock(spec=Span)
        error_messages = {404: "Custom Not Found"}
        monitor = SpanAttributeMonitor(mock_span, {404}, error_messages)

        # Act
        result = monitor._get_error_message(404)

        # Assert
        assert result == "Custom Not Found"

    def test_get_error_message_range_match(self):
        """Test _get_error_message with range match."""
        # Arrange
        mock_span = Mock(spec=Span)
        error_messages = {range(400, 500): "Client Error Range"}
        monitor = SpanAttributeMonitor(mock_span, {404}, error_messages)

        # Act
        result = monitor._get_error_message(404)

        # Assert
        assert result == "Client Error Range"

    def test_get_error_message_default_client_error(self):
        """Test _get_error_message with default client error message."""
        # Arrange
        mock_span = Mock(spec=Span)
        monitor = SpanAttributeMonitor(mock_span, {404}, {})

        # Act
        result = monitor._get_error_message(404)

        # Assert
        assert result == "Client Error: HTTP 404"

    def test_get_error_message_default_server_error(self):
        """Test _get_error_message with default server error message."""
        # Arrange
        mock_span = Mock(spec=Span)
        monitor = SpanAttributeMonitor(mock_span, {500}, {})

        # Act
        result = monitor._get_error_message(500)

        # Assert
        assert result == "Server Error: HTTP 500"


class TestInstrumentedFastAPI:
    """Test _InstrumentedFastAPI class functionality."""

    @patch("netra.instrumentation.fastapi.FastAPIInstrumentor.instrument_app")
    def test_initialization(self, mock_instrument_app):
        """Test _InstrumentedFastAPI initialization."""
        # Act
        app = _InstrumentedFastAPI()

        # Assert
        mock_instrument_app.assert_called_once_with(
            app,
            server_request_hook=None,
            client_request_hook=None,
            client_response_hook=None,
            tracer_provider=None,
            meter_provider=None,
            excluded_urls=None,
            http_capture_headers_server_request=None,
            http_capture_headers_server_response=None,
            http_capture_headers_sanitize_fields=None,
            exclude_spans=None,
            error_status_codes=None,
            error_messages=None,
        )
        assert app in _InstrumentedFastAPI._instrumented_fastapi_apps

    def test_del_removes_from_instrumented_apps(self):
        """Test __del__ removes app from instrumented apps set."""
        # Arrange
        with patch("netra.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"):
            app = _InstrumentedFastAPI()
            assert app in _InstrumentedFastAPI._instrumented_fastapi_apps

        # Act
        app.__del__()

        # Assert
        assert app not in _InstrumentedFastAPI._instrumented_fastapi_apps


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_route_details_full_match(self):
        """Test _get_route_details with full route match."""
        # Arrange
        mock_route = Mock()
        mock_route.matches.return_value = (Match.FULL, None)
        mock_route.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route]

        scope = {"app": mock_app}

        # Act
        result = _get_route_details(scope)

        # Assert
        assert result == "/test/path"

    def test_get_route_details_partial_match(self):
        """Test _get_route_details with partial route match."""
        # Arrange
        mock_route1 = Mock()
        mock_route1.matches.return_value = (Match.NONE, None)
        mock_route1.path = "/other/path"

        mock_route2 = Mock()
        mock_route2.matches.return_value = (Match.PARTIAL, None)
        mock_route2.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route1, mock_route2]

        scope = {"app": mock_app}

        # Act
        result = _get_route_details(scope)

        # Assert
        assert result == "/test/path"

    def test_get_route_details_no_match(self):
        """Test _get_route_details with no route match."""
        # Arrange
        mock_route = Mock()
        mock_route.matches.return_value = (Match.NONE, None)
        mock_route.path = "/test/path"

        mock_app = Mock()
        mock_app.routes = [mock_route]

        scope = {"app": mock_app}

        # Act
        result = _get_route_details(scope)

        # Assert
        assert result is None

    @patch("netra.instrumentation.fastapi._get_route_details")
    @patch("netra.instrumentation.fastapi.sanitize_method")
    def test_get_default_span_details_with_route_and_method(self, mock_sanitize_method, mock_get_route_details):
        """Test _get_default_span_details with route and method."""
        # Arrange
        mock_get_route_details.return_value = "/test/path"
        mock_sanitize_method.return_value = "GET"
        scope = {"method": "GET"}

        # Act
        span_name, attributes = _get_default_span_details(scope)

        # Assert
        assert span_name == "GET /test/path"
        assert attributes["http.route"] == "/test/path"

    @patch("netra.instrumentation.fastapi._get_route_details")
    @patch("netra.instrumentation.fastapi.sanitize_method")
    def test_get_default_span_details_no_route(self, mock_sanitize_method, mock_get_route_details):
        """Test _get_default_span_details with no route."""
        # Arrange
        mock_get_route_details.return_value = None
        mock_sanitize_method.return_value = "GET"
        scope = {"method": "GET"}

        # Act
        span_name, attributes = _get_default_span_details(scope)

        # Assert
        assert span_name == "GET"
        assert attributes == {}

    @patch("netra.instrumentation.fastapi._get_route_details")
    @patch("netra.instrumentation.fastapi.sanitize_method")
    def test_get_default_span_details_other_method(self, mock_sanitize_method, mock_get_route_details):
        """Test _get_default_span_details with _OTHER method."""
        # Arrange
        mock_get_route_details.return_value = "/test/path"
        mock_sanitize_method.return_value = "_OTHER"
        scope = {"method": "CUSTOM"}

        # Act
        span_name, attributes = _get_default_span_details(scope)

        # Assert
        assert span_name == "HTTP /test/path"
        assert attributes["http.route"] == "/test/path"
