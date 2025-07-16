"""
Unit tests for NetraOpenAIInstrumentor class.
Focuses on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import MagicMock, Mock, patch

from netra.instrumentation.openai import (
    NetraOpenAIInstrumentor,
    is_streaming_response,
    should_suppress_instrumentation,
)


class TestNetraOpenAIInstrumentor:
    """Test NetraOpenAIInstrumentor core functionality."""

    def test_initialization(self):
        """Test NetraOpenAIInstrumentor initialization."""
        # Act
        instrumentor = NetraOpenAIInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = NetraOpenAIInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "openai >= 1.0.0" in dependencies

    @patch("netra.instrumentation.openai.get_tracer")
    @patch("netra.instrumentation.openai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = NetraOpenAIInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        # Should wrap all methods (chat, completion, embeddings, responses)
        assert mock_wrap_function.call_count >= 6  # At least 6 methods are wrapped

    @patch("netra.instrumentation.openai.get_tracer")
    @patch("netra.instrumentation.openai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        # Arrange
        instrumentor = NetraOpenAIInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Act
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # Assert
        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.openai", mock_get_tracer.call_args[0][1], mock_tracer_provider  # version
        )
        assert mock_wrap_function.call_count >= 6

    @patch("netra.instrumentation.openai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        # Arrange
        instrumentor = NetraOpenAIInstrumentor()

        # Act
        instrumentor._uninstrument()

        # Assert
        # Should unwrap all methods (chat, completion, embeddings, responses)
        assert mock_unwrap.call_count >= 6


class TestWrappers:
    """Test wrapper functionality in the OpenAI instrumentation module."""

    def test_chat_wrapper_non_streaming(self):
        """Test chat_wrapper for non-streaming requests."""
        from netra.instrumentation.openai.wrappers import chat_wrapper

        # Arrange
        mock_tracer = Mock()
        mock_span_context = MagicMock()
        mock_span_context.__enter__.return_value
        mock_tracer.start_as_current_span.return_value = mock_span_context

        wrapped = Mock(return_value={"id": "test-id", "choices": [{"message": {"content": "test"}}]})
        instance = Mock()
        args = ()
        kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False}

        wrapper = chat_wrapper(mock_tracer)

        # Act
        result = wrapper(wrapped, instance, args, kwargs)

        # Assert
        wrapped.assert_called_once_with(*args, **kwargs)
        mock_tracer.start_as_current_span.assert_called_once()
        assert result == wrapped.return_value

    @patch("netra.instrumentation.openai.wrappers.StreamingWrapper")
    def test_chat_wrapper_streaming(self, mock_streaming_wrapper_class):
        """Test chat_wrapper for streaming requests."""
        from netra.instrumentation.openai.wrappers import chat_wrapper

        # Arrange
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span

        def generator():
            yield {"id": "test-id", "choices": [{"delta": {"content": "Hello"}}]}
            yield {"id": "test-id", "choices": [{"delta": {"content": " world"}}]}

        wrapped = Mock(return_value=generator())
        instance = Mock()
        args = ()
        kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}

        # Mock the StreamingWrapper to return a simple object
        mock_wrapper_instance = Mock()
        mock_streaming_wrapper_class.return_value = mock_wrapper_instance

        wrapper = chat_wrapper(mock_tracer)

        # Act
        result = wrapper(wrapped, instance, args, kwargs)

        # Assert
        wrapped.assert_called_once_with(*args, **kwargs)
        mock_tracer.start_span.assert_called_once()
        mock_streaming_wrapper_class.assert_called_once()
        assert result == mock_wrapper_instance


class TestUtilityFunctions:
    """Test utility functions in the openai instrumentation module."""

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

    @patch("netra.instrumentation.openai.context_api.get_value")
    def test_should_suppress_instrumentation_true(self, mock_get_value):
        """Test should_suppress_instrumentation returns True when suppression is enabled."""
        # Arrange
        mock_get_value.return_value = True

        # Act
        result = should_suppress_instrumentation()

        # Assert
        assert result is True

    @patch("netra.instrumentation.openai.context_api.get_value")
    def test_should_suppress_instrumentation_false(self, mock_get_value):
        """Test should_suppress_instrumentation returns False when suppression is disabled."""
        # Arrange
        mock_get_value.return_value = False

        # Act
        result = should_suppress_instrumentation()

        # Assert
        assert result is False
