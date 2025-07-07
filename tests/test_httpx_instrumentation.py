"""
Unit tests for HTTPXInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.httpx import HTTPXInstrumentor, get_default_span_name


class TestHTTPXInstrumentor:
    """Test HTTPXInstrumentor core functionality."""

    def test_initialization(self):
        """Test HTTPXInstrumentor initialization."""
        # Act
        instrumentor = HTTPXInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = HTTPXInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "httpx >= 0.18.0" in dependencies

    @patch("netra.instrumentation.httpx.get_tracer")
    @patch("netra.instrumentation.httpx.get_meter")
    @patch("netra.instrumentation.httpx._instrument")
    def test_instrument_with_default_parameters(self, mock_instrument, mock_get_meter, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = HTTPXInstrumentor()
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()
        mock_instrument.assert_called_once()

    @patch("netra.instrumentation.httpx.get_tracer")
    @patch("netra.instrumentation.httpx.get_meter")
    @patch("netra.instrumentation.httpx._instrument")
    def test_instrument_with_custom_parameters(self, mock_instrument, mock_get_meter, mock_get_tracer):
        """Test _instrument method with custom parameters."""
        # Arrange
        instrumentor = HTTPXInstrumentor()
        mock_tracer_provider = Mock()
        mock_meter_provider = Mock()
        mock_request_hook = Mock()
        mock_response_hook = Mock()
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram

        # Act
        instrumentor._instrument(
            tracer_provider=mock_tracer_provider,
            meter_provider=mock_meter_provider,
            request_hook=mock_request_hook,
            response_hook=mock_response_hook,
            excluded_urls="http://example.com",
        )

        # Assert
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()
        mock_instrument.assert_called_once()

        # Verify the _instrument call includes the hooks
        call_args = mock_instrument.call_args
        assert call_args[1]["request_hook"] == mock_request_hook
        assert call_args[1]["response_hook"] == mock_response_hook

    @patch("netra.instrumentation.httpx._uninstrument")
    def test_uninstrument(self, mock_uninstrument):
        """Test _uninstrument method calls the underlying uninstrument function."""
        # Arrange
        instrumentor = HTTPXInstrumentor()

        # Act
        instrumentor._uninstrument()

        # Assert
        mock_uninstrument.assert_called_once()

    @patch("netra.instrumentation.httpx.get_tracer")
    @patch("netra.instrumentation.httpx.get_meter")
    @patch("netra.instrumentation.httpx._instrument")
    def test_instrument_with_duration_histogram_boundaries(self, mock_instrument, mock_get_meter, mock_get_tracer):
        """Test _instrument method with custom duration histogram boundaries."""
        # Arrange
        instrumentor = HTTPXInstrumentor()
        custom_boundaries = [0.1, 0.5, 1.0, 2.0, 5.0]
        mock_tracer = Mock()
        mock_meter = Mock()
        mock_histogram = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_get_meter.return_value = mock_meter
        mock_meter.create_histogram.return_value = mock_histogram

        # Act
        instrumentor._instrument(duration_histogram_boundaries=custom_boundaries)

        # Assert
        mock_get_tracer.assert_called_once()
        mock_get_meter.assert_called_once()
        mock_instrument.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions in the httpx instrumentation module."""

    def test_get_default_span_name_with_standard_method(self):
        """Test get_default_span_name with standard HTTP method."""
        # Act
        result = get_default_span_name("GET")

        # Assert
        assert result == "GET"

    def test_get_default_span_name_with_lowercase_method(self):
        """Test get_default_span_name with lowercase HTTP method."""
        # Act
        result = get_default_span_name("post")

        # Assert
        assert result == "POST"

    def test_get_default_span_name_with_custom_method(self):
        """Test get_default_span_name with custom HTTP method."""
        # Act
        result = get_default_span_name("PATCH")

        # Assert
        assert result == "PATCH"

    def test_get_default_span_name_with_empty_method(self):
        """Test get_default_span_name with empty method."""
        # Act
        result = get_default_span_name("")

        # Assert
        assert result == "HTTP"
