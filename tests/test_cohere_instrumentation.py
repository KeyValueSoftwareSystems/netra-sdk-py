"""
Unit tests for CohereInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

from netra.instrumentation.cohere import CohereInstrumentor, _llm_request_type_by_method, should_send_prompts


class TestCohereInstrumentor:
    """Test CohereInstrumentor core functionality."""

    def test_initialization(self):
        """Test CohereInstrumentor initialization."""
        # Act
        instrumentor = CohereInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_initialization_with_exception_logger(self):
        """Test CohereInstrumentor initialization with custom exception logger."""
        # Arrange
        mock_logger = Mock()

        # Act
        instrumentor = CohereInstrumentor(exception_logger=mock_logger)

        # Assert
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = CohereInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "cohere >=4.2.7, <6" in dependencies

    @patch("netra.instrumentation.cohere.get_tracer")
    @patch("netra.instrumentation.cohere.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = CohereInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        # Should wrap all methods defined in WRAPPED_METHODS (5 methods)
        assert mock_wrap_function.call_count == 5

    @patch("netra.instrumentation.cohere.get_tracer")
    @patch("netra.instrumentation.cohere.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        # Arrange
        instrumentor = CohereInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # Assert
        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.cohere", mock_get_tracer.call_args[0][1], mock_tracer_provider  # version
        )
        assert mock_wrap_function.call_count == 5

    @patch("netra.instrumentation.cohere.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        # Arrange
        instrumentor = CohereInstrumentor()

        # Act
        instrumentor._uninstrument()

        # Assert
        # Should unwrap all methods defined in WRAPPED_METHODS (5 methods)
        assert mock_unwrap.call_count == 5


class TestUtilityFunctions:
    """Test utility functions in the cohere instrumentation module."""

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

    def test_llm_request_type_by_method(self):
        """Test _llm_request_type_by_method returns correct request types."""
        # Arrange & Act & Assert
        from opentelemetry.semconv_ai import LLMRequestTypeValues

        test_cases = [
            ("chat", LLMRequestTypeValues.CHAT),
            ("chat_stream", LLMRequestTypeValues.CHAT),
            ("rerank", LLMRequestTypeValues.RERANK),
            ("unknown_method", LLMRequestTypeValues.UNKNOWN),
            (None, LLMRequestTypeValues.UNKNOWN),
        ]

        for method, expected_type in test_cases:
            result = _llm_request_type_by_method(method)
            assert result == expected_type
