"""
Unit tests for MistralAiInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import Mock, patch

import pytest

# Skip tests if mistralai is not installed
pytest.importorskip("mistralai")

from netra.instrumentation.mistralai import MistralAiInstrumentor, _llm_request_type_by_method, should_send_prompts


class TestMistralAiInstrumentor:
    """Test MistralAiInstrumentor core functionality."""

    def test_initialization(self):
        """Test MistralAiInstrumentor initialization."""
        # Act
        instrumentor = MistralAiInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_initialization_with_exception_logger(self):
        """Test MistralAiInstrumentor initialization with custom exception logger."""
        # Arrange
        mock_logger = Mock()

        # Act
        instrumentor = MistralAiInstrumentor(exception_logger=mock_logger)

        # Assert
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = MistralAiInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "mistralai >= 1.0.0" in dependencies

    @patch("netra.instrumentation.mistralai.get_tracer")
    @patch("netra.instrumentation.mistralai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = MistralAiInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        # Should wrap all methods defined in WRAPPED_METHODS (5 methods)
        assert mock_wrap_function.call_count == 5

    @patch("netra.instrumentation.mistralai.get_tracer")
    @patch("netra.instrumentation.mistralai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        # Arrange
        instrumentor = MistralAiInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # Assert
        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.mistralai", mock_get_tracer.call_args[0][1], mock_tracer_provider  # version
        )
        assert mock_wrap_function.call_count == 5

    @patch("netra.instrumentation.mistralai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        # Arrange
        instrumentor = MistralAiInstrumentor()

        # Act
        instrumentor._uninstrument()

        # Assert
        # Should unwrap all methods defined in WRAPPED_METHODS (5 methods)
        assert mock_unwrap.call_count == 5


class TestUtilityFunctions:
    """Test utility functions in the mistralai instrumentation module."""

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

    def test_llm_request_type_by_method_chat_methods(self):
        """Test _llm_request_type_by_method returns CHAT for chat methods."""
        # Arrange & Act & Assert
        from opentelemetry.semconv_ai import LLMRequestTypeValues

        chat_methods = ["complete", "complete_async", "stream", "stream_async"]
        for method in chat_methods:
            result = _llm_request_type_by_method(method)
            assert result == LLMRequestTypeValues.CHAT

    def test_llm_request_type_by_method_unknown(self):
        """Test _llm_request_type_by_method returns UNKNOWN for unknown methods."""
        # Arrange & Act & Assert
        from opentelemetry.semconv_ai import LLMRequestTypeValues

        test_cases = ["unknown_method", "create", None]
        for method in test_cases:
            result = _llm_request_type_by_method(method)
            assert result == LLMRequestTypeValues.UNKNOWN
