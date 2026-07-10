"""
Unit tests for NetraOpenAIInstrumentor class.
Focuses on core functionality and happy path scenarios.
"""

from typing import Collection
from unittest.mock import MagicMock, Mock, patch

from netra.instrumentation.openai import NetraOpenAIInstrumentor
from netra.instrumentation.openai.utils import should_suppress_instrumentation


class TestNetraOpenAIInstrumentor:
    """Test NetraOpenAIInstrumentor core functionality."""

    def test_initialization(self):
        """Test NetraOpenAIInstrumentor initialization."""
        instrumentor = NetraOpenAIInstrumentor()

        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        instrumentor = NetraOpenAIInstrumentor()

        dependencies = instrumentor.instrumentation_dependencies()

        assert isinstance(dependencies, Collection)
        assert "openai >= 1.0.0" in dependencies

    @patch("netra.instrumentation.openai.get_tracer")
    @patch("netra.instrumentation.openai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        instrumentor = NetraOpenAIInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        # chat x2, embeddings x2, responses x2
        assert mock_wrap_function.call_count == 6

    @patch("netra.instrumentation.openai.get_tracer")
    @patch("netra.instrumentation.openai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        instrumentor = NetraOpenAIInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.openai", mock_get_tracer.call_args[0][1], mock_tracer_provider
        )
        assert mock_wrap_function.call_count == 6

    @patch("netra.instrumentation.openai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all OpenAI methods it targets."""
        instrumentor = NetraOpenAIInstrumentor()

        instrumentor._uninstrument()

        # chat x2, completions x2, embeddings x2, responses x2
        assert mock_unwrap.call_count == 8


class TestWrappers:
    """Test wrapper functionality in the OpenAI instrumentation module."""

    @patch("netra.instrumentation.openai.wrappers.record_span_timing")
    def test_chat_wrapper_non_streaming(self, mock_record_timing):
        """Test chat_wrapper for non-streaming requests starts a span and returns the wrapped result."""
        from netra.instrumentation.openai.wrappers import chat_wrapper

        mock_tracer = Mock()
        mock_span_context = MagicMock()
        mock_span_context.__enter__.return_value = Mock()
        mock_tracer.start_as_current_span.return_value = mock_span_context

        wrapped = Mock(return_value={"id": "test-id", "choices": [{"message": {"content": "test"}}]})
        instance = Mock()
        args = ()
        kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False}

        wrapper = chat_wrapper(mock_tracer)

        result = wrapper(wrapped, instance, args, kwargs)

        wrapped.assert_called_once_with(*args, **kwargs)
        mock_tracer.start_as_current_span.assert_called_once()
        assert result == wrapped.return_value

    @patch("netra.instrumentation.openai.wrappers.StreamingWrapper")
    def test_chat_wrapper_streaming(self, mock_streaming_wrapper_class):
        """Test chat_wrapper for streaming requests wraps the response in StreamingWrapper."""
        from netra.instrumentation.openai.wrappers import chat_wrapper

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

        mock_wrapper_instance = Mock()
        mock_streaming_wrapper_class.return_value = mock_wrapper_instance

        wrapper = chat_wrapper(mock_tracer)

        result = wrapper(wrapped, instance, args, kwargs)

        wrapped.assert_called_once_with(*args, **kwargs)
        mock_tracer.start_span.assert_called_once()
        mock_streaming_wrapper_class.assert_called_once()
        assert result == mock_wrapper_instance


class TestUtilityFunctions:
    """Test utility functions in the openai instrumentation module."""

    @patch("netra.instrumentation.openai.utils.context_api.get_value")
    def test_should_suppress_instrumentation_true(self, mock_get_value):
        """Test should_suppress_instrumentation returns True when suppression is enabled."""
        mock_get_value.return_value = True

        result = should_suppress_instrumentation()

        assert result is True

    @patch("netra.instrumentation.openai.utils.context_api.get_value")
    def test_should_suppress_instrumentation_false(self, mock_get_value):
        """Test should_suppress_instrumentation returns False when suppression is disabled."""
        mock_get_value.return_value = False

        result = should_suppress_instrumentation()

        assert result is False
