"""
Unit tests for GoogleGenAiInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.google_genai import (
    GoogleGenAiInstrumentor,
    is_async_streaming_response,
    is_streaming_response,
    should_send_prompts,
)


class TestGoogleGenAiInstrumentor:
    """Test GoogleGenAiInstrumentor core functionality."""

    def test_initialization(self):
        """Test GoogleGenAiInstrumentor initialization."""
        # Act
        instrumentor = GoogleGenAiInstrumentor(exception_logger=None)

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_initialization_with_exception_logger(self):
        """Test GoogleGenAiInstrumentor initialization with custom exception logger."""
        # Arrange
        mock_logger = Mock()

        # Act
        instrumentor = GoogleGenAiInstrumentor(exception_logger=mock_logger)

        # Assert
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = GoogleGenAiInstrumentor(exception_logger=None)

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "google-genai >= 0.1.0" in dependencies

    @patch("netra.instrumentation.google_genai.get_tracer")
    @patch("netra.instrumentation.google_genai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = GoogleGenAiInstrumentor(exception_logger=None)
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        # Should wrap all methods defined in WRAPPED_METHODS (8 methods)
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.google_genai.get_tracer")
    @patch("netra.instrumentation.google_genai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        # Arrange
        instrumentor = GoogleGenAiInstrumentor(exception_logger=None)
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # Assert
        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.google_genai", mock_get_tracer.call_args[0][1], mock_tracer_provider  # version
        )
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.google_genai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        # Arrange
        instrumentor = GoogleGenAiInstrumentor(exception_logger=None)

        # Act
        instrumentor._uninstrument()

        # Assert
        # Should unwrap all methods defined in WRAPPED_METHODS (8 methods)
        assert mock_unwrap.call_count == 8


class TestUtilityFunctions:
    """Test utility functions in the google_genai instrumentation module."""

    @patch.dict("os.environ", {"TRACELOOP_TRACE_CONTENT": "true"})
    def test_should_send_prompts_true_from_env(self):
        """Test should_send_prompts returns True when environment variable is set."""
        # Act
        result = should_send_prompts()

        # Assert
        assert result is True

    def test_should_send_prompts_default(self):
        """Test should_send_prompts returns default value when no environment variable is set."""
        # Act
        result = should_send_prompts()

        # Assert
        assert result is True  # Default behavior when no env var is set

    def test_is_streaming_response_with_generator(self):
        """Test is_streaming_response returns True for generator objects."""

        # Arrange
        def sample_generator():
            yield 1
            yield 2

        generator = sample_generator()

        # Act
        result = is_streaming_response(generator)

        # Assert
        assert result is True

    def test_is_streaming_response_with_non_generator(self):
        """Test is_streaming_response returns False for non-generator objects."""
        # Act
        result = is_streaming_response("not a generator")

        # Assert
        assert result is False

    def test_is_async_streaming_response_with_async_generator(self):
        """Test is_async_streaming_response returns True for async generator objects."""

        # Arrange
        async def sample_async_generator():
            yield 1
            yield 2

        async_generator = sample_async_generator()

        # Act
        result = is_async_streaming_response(async_generator)

        # Assert
        assert result is True

        # Cleanup
        async_generator.aclose()

    def test_is_async_streaming_response_with_non_async_generator(self):
        """Test is_async_streaming_response returns False for non-async generator objects."""
        # Act
        result = is_async_streaming_response("not an async generator")

        # Assert
        assert result is False
