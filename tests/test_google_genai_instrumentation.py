"""
Unit tests for NetraGoogleGenAiInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.google_genai import NetraGoogleGenAiInstrumentor


class TestNetraGoogleGenAiInstrumentor:
    """Test NetraGoogleGenAiInstrumentor core functionality."""

    def test_initialization(self):
        """Test NetraGoogleGenAiInstrumentor initialization."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        dependencies = instrumentor.instrumentation_dependencies()

        assert isinstance(dependencies, Collection)
        assert "google-genai >= 0.1.0" in dependencies

    @patch("netra.instrumentation.google_genai.get_tracer")
    @patch("netra.instrumentation.google_genai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        instrumentor = NetraGoogleGenAiInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.google_genai.get_tracer")
    @patch("netra.instrumentation.google_genai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        instrumentor = NetraGoogleGenAiInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.google_genai", mock_get_tracer.call_args[0][1], mock_tracer_provider
        )
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.google_genai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        instrumentor._uninstrument()

        assert mock_unwrap.call_count == 8
