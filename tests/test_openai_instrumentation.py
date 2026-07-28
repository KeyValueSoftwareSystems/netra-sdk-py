"""
Unit tests for NetraOpenAIInstrumentor class.
Focuses on core functionality and happy path scenarios.
"""

import json
from typing import Collection
from unittest.mock import MagicMock, Mock, patch

from opentelemetry.semconv_ai import SpanAttributes

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
        attrs = self._capture(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 8},
            }
        )

        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 100
        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 20
        assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 120
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 64
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 30
        assert attrs[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 8

    def test_responses_api_cache_write(self):
        """Responses API usage (input_tokens_details) also captures cache_write_tokens."""
        from opentelemetry.semconv_ai import SpanAttributes

        attrs = self._capture(
            {
                "input_tokens": 200,
                "output_tokens": 40,
                "total_tokens": 240,
                "input_tokens_details": {"cached_tokens": 128, "cache_write_tokens": 50},
                "output_tokens_details": {"reasoning_tokens": 12},
            }
        )

        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 200
        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 40
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 128
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 50
        assert attrs[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 12

    def test_zero_valued_cache_write_is_recorded(self):
        """A cache_write_tokens of 0 must be recorded, not dropped as falsy."""
        from opentelemetry.semconv_ai import SpanAttributes

        attrs = self._capture(
            {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            }
        )

        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 0
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 0
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 0

    def test_missing_cache_write_is_omitted(self):
        """When cache_write_tokens is absent, the creation attribute is not set."""
        from opentelemetry.semconv_ai import SpanAttributes

        attrs = self._capture(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        )

        assert SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 4


class TestResponseMessageAttributesToolCalls:
    """Test _set_response_message_attributes handling of tool call responses."""

    @staticmethod
    def _capture(response_dict):
        from netra.instrumentation.openai.utils import _set_response_message_attributes

        span = Mock()
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        _set_response_message_attributes(span, response_dict)
        return captured

    def test_tool_calls_response_no_empty_assistant_entry(self):
        """When content is null and tool_calls are present, no empty assistant entry should be emitted."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        attrs = self._capture(response)

        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]) == {
            "name": "get_weather",
            "arguments": '{"location":"London"}',
        }
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_call_id"] == "call_abc123"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.finish_reason"] == "tool_calls"
        assert f"{SpanAttributes.LLM_COMPLETIONS}.2.role" not in attrs

    def test_tool_calls_response_with_content_keeps_assistant_entry(self):
        """When content is non-empty alongside tool_calls, both assistant text and tool calls are emitted."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that for you.",
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":"test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        attrs = self._capture(response)

        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Let me check that for you."
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.role"] == "assistant"
        assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.content"]) == {
            "name": "lookup",
            "arguments": '{"q":"test"}',
        }
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.tool_call_id"] == "call_xyz"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.2.finish_reason"] == "tool_calls"

    def test_normal_text_response_still_works(self):
        """A regular text response (no tool_calls) still emits the assistant entry."""
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ]
        }

        attrs = self._capture(response)

        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "Hello there!"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.finish_reason"] == "stop"

    def test_multiple_tool_calls_in_single_response(self):
        """Multiple tool_calls in one message produce sequential indexed entries without an empty leader."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "get_time", "arguments": '{"tz":"EST"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        attrs = self._capture(response)

        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_call_id"] == "call_1"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.tool_call_id"] == "call_2"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.2.finish_reason"] == "tool_calls"

    def test_delta_tool_calls_no_empty_entry(self):
        """Delta branch (streaming) also skips empty content when tool_calls present."""
        response = {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_stream",
                                "function": {"name": "search", "arguments": '{"q":"test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        attrs = self._capture(response)

        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert "search" in attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_call_id"] == "call_stream"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.1.finish_reason"] == "tool_calls"


class TestChatCompletionInputToolCalls:
    """Test _set_chat_completion_input handling of tool-related messages."""

    @staticmethod
    def _capture(messages):
        from netra.instrumentation.openai.utils import _set_chat_completion_input

        span = Mock()
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        _set_chat_completion_input(span, messages)
        return captured

    def test_pydantic_model_message_is_captured(self):
        """Non-dict messages with model_dump() are converted and captured."""
        pydantic_msg = Mock()
        pydantic_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello!",
        }

        messages = [
            {"role": "user", "content": "Hi"},
            pydantic_msg,
        ]

        attrs = self._capture(messages)

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Hi"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Hello!"

    def test_assistant_tool_calls_are_serialized(self):
        """Assistant messages with tool_calls produce indexed entries with name/arguments JSON."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": '{"temp": 22}'},
        ]

        attrs = self._capture(messages)

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "What's the weather?"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
        assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"]) == {
            "name": "get_weather",
            "arguments": '{"location":"London"}',
        }
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "call_abc"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "tool"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == '{"temp": 22}'
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.tool_call_id"] == "call_abc"

    def test_tool_call_id_captured_on_tool_messages(self):
        """Tool messages have their tool_call_id captured as a span attribute."""
        messages = [
            {"role": "tool", "tool_call_id": "call_xyz", "content": "result data"},
        ]

        attrs = self._capture(messages)

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "tool"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "result data"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.tool_call_id"] == "call_xyz"

    def test_pydantic_assistant_with_tool_calls(self):
        """A Pydantic ChatCompletionMessage with tool_calls is correctly converted and serialized."""
        func_mock = Mock()
        func_mock.name = "get_weather"
        func_mock.arguments = '{"location":"Paris"}'
        tc_mock = Mock()
        tc_mock.id = "call_pydantic"
        tc_mock.function = func_mock

        pydantic_msg = Mock()
        pydantic_msg.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_pydantic",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
                }
            ],
        }

        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            pydantic_msg,
            {"role": "tool", "tool_call_id": "call_pydantic", "content": '{"temp": 20}'},
        ]

        attrs = self._capture(messages)

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
        assert json.loads(attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"]) == {
            "name": "get_weather",
            "arguments": '{"location":"Paris"}',
        }
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.tool_call_id"] == "call_pydantic"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "tool"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.tool_call_id"] == "call_pydantic"

    def test_none_content_message_is_skipped(self):
        """A message with content=None and no tool_calls is skipped entirely — no blank entry emitted."""
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "actual question"},
        ]

        attrs = self._capture(messages)

        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" in attrs
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "actual question"
        assert f"{SpanAttributes.LLM_PROMPTS}.1.role" not in attrs

    def test_contiguous_indices_with_tool_call_conversation(self):
        """Full tool-call conversation produces contiguous prompt indices with no gaps."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Weather in London?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 15}'},
        ]

        attrs = self._capture(messages)

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == "Hi!"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "assistant"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.3.tool_call_id"] == "call_1"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.4.role"] == "tool"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.4.tool_call_id"] == "call_1"
