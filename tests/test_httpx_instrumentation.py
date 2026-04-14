"""
Unit tests for HTTPXInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.httpx import HTTPXInstrumentor
from netra.instrumentation.httpx.utils import get_default_span_name


class TestHTTPXInstrumentor:
    """Test HTTPXInstrumentor core functionality."""

    def test_initialization(self):
        """Test HTTPXInstrumentor initialization."""
        instrumentor = HTTPXInstrumentor()

        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        instrumentor = HTTPXInstrumentor()

        dependencies = instrumentor.instrumentation_dependencies()

        assert isinstance(dependencies, Collection)
        assert "httpx >= 0.18.0" in dependencies

    @patch("netra.instrumentation.httpx.get_tracer")
    @patch("netra.instrumentation.httpx.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap, mock_get_tracer):
        """Test _instrument method with default parameters."""
        instrumentor = HTTPXInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        assert mock_wrap.call_count == 2

    @patch("netra.instrumentation.httpx.get_tracer")
    @patch("netra.instrumentation.httpx.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        instrumentor = HTTPXInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        mock_get_tracer.assert_called_once()
        assert mock_wrap.call_count == 2

    @patch("netra.instrumentation.httpx.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method calls unwrap for both sync and async clients."""
        instrumentor = HTTPXInstrumentor()

        instrumentor._uninstrument()

        assert mock_unwrap.call_count == 2


class TestUtilityFunctions:
    """Test utility functions in the httpx instrumentation module."""

    def test_get_default_span_name_with_standard_method(self):
        """Test get_default_span_name with standard HTTP method."""
        result = get_default_span_name("GET")

        assert result == "GET"

    def test_get_default_span_name_with_lowercase_method(self):
        """Test get_default_span_name with lowercase HTTP method."""
        result = get_default_span_name("post")

        assert result == "POST"

    def test_get_default_span_name_with_custom_method(self):
        """Test get_default_span_name with custom HTTP method."""
        result = get_default_span_name("PATCH")

        assert result == "PATCH"

    def test_get_default_span_name_with_empty_method(self):
        """Test get_default_span_name with empty method."""
        result = get_default_span_name("")

        assert result == "HTTP"
