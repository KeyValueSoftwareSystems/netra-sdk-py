"""
Unit tests for AioHttpClientInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.aiohttp import AioHttpClientInstrumentor, get_default_span_name


class TestAioHttpClientInstrumentor:
    """Test AioHttpClientInstrumentor core functionality."""

    def test_initialization(self):
        """Test AioHttpClientInstrumentor initialization."""
        # Act
        instrumentor = AioHttpClientInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = AioHttpClientInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "aiohttp >= 3.0.0" in dependencies

    @patch("netra.instrumentation.aiohttp.get_tracer")
    @patch("netra.instrumentation.aiohttp.get_meter")
    @patch("netra.instrumentation.aiohttp._instrument")
    @patch(
        "netra.instrumentation.aiohttp._OpenTelemetrySemanticConventionStability._get_opentelemetry_stability_opt_in_mode"
    )
    @patch("netra.instrumentation.aiohttp._get_schema_url")
    def test_instrument_with_default_parameters(
        self, mock_schema_url, mock_stability_mode, mock_instrument, mock_get_meter, mock_get_tracer
    ):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = AioHttpClientInstrumentor()
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()

        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_schema_url.return_value = "test_schema"
        mock_stability_mode.return_value = Mock()

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()
        mock_instrument.assert_called_once()

    @patch("netra.instrumentation.aiohttp.get_tracer")
    @patch("netra.instrumentation.aiohttp.get_meter")
    @patch("netra.instrumentation.aiohttp._instrument")
    @patch(
        "netra.instrumentation.aiohttp._OpenTelemetrySemanticConventionStability._get_opentelemetry_stability_opt_in_mode"
    )
    @patch("netra.instrumentation.aiohttp._get_schema_url")
    def test_instrument_with_custom_parameters(
        self, mock_schema_url, mock_stability_mode, mock_instrument, mock_get_meter, mock_get_tracer
    ):
        """Test _instrument method with custom parameters."""
        # Arrange
        instrumentor = AioHttpClientInstrumentor()
        mock_tracer_provider = Mock()
        mock_meter_provider = Mock()
        mock_request_hook = Mock()
        mock_response_hook = Mock()
        excluded_urls = "http://example.com"
        duration_boundaries = [0.1, 0.5, 1.0]

        mock_tracer = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()

        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_schema_url.return_value = "test_schema"
        mock_stability_mode.return_value = Mock()

        # Act
        instrumentor._instrument(
            tracer_provider=mock_tracer_provider,
            meter_provider=mock_meter_provider,
            request_hook=mock_request_hook,
            response_hook=mock_response_hook,
            excluded_urls=excluded_urls,
            duration_histogram_boundaries=duration_boundaries,
        )

        # Assert
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()
        mock_instrument.assert_called_once()

        # Verify _instrument was called with correct parameters
        call_args = mock_instrument.call_args
        assert call_args[1]["request_hook"] == mock_request_hook
        assert call_args[1]["response_hook"] == mock_response_hook

    @patch("netra.instrumentation.aiohttp._uninstrument")
    def test_uninstrument(self, mock_uninstrument):
        """Test _uninstrument method calls global uninstrument function."""
        # Arrange
        instrumentor = AioHttpClientInstrumentor()

        # Act
        instrumentor._uninstrument()

        # Assert
        mock_uninstrument.assert_called_once()

    @patch("netra.instrumentation.aiohttp._uninstrument_from")
    def test_uninstrument_session(self, mock_uninstrument_from):
        """Test uninstrument_session static method."""
        # Arrange
        mock_session = Mock()

        # Act
        AioHttpClientInstrumentor.uninstrument_session(mock_session)

        # Assert
        mock_uninstrument_from.assert_called_once_with(mock_session, restore_as_bound_func=True)


class TestUtilityFunctions:
    """Test utility functions in the aiohttp instrumentation module."""

    def test_get_default_span_name(self):
        """Test get_default_span_name function with different HTTP methods."""
        # Test cases
        test_cases = [
            ("GET", "GET"),
            ("POST", "POST"),
            ("PUT", "PUT"),
            ("DELETE", "DELETE"),
            ("PATCH", "PATCH"),
            ("get", "GET"),  # Should handle lowercase
            ("custom", "HTTP"),  # Unknown methods return "HTTP"
        ]

        for method, expected_name in test_cases:
            # Act
            span_name = get_default_span_name(method)

            # Assert
            assert span_name == expected_name
