"""
Unit tests for NetraOpenAIInstrumentor class.
Focuses on core functionality and happy path scenarios.
"""

import json
from typing import Collection
from unittest.mock import MagicMock, Mock, patch

from netra.instrumentation.libraries.openai import NetraOpenAIInstrumentor
from netra.instrumentation.libraries.openai.utils import should_suppress_instrumentation


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

    @patch("netra.instrumentation.libraries.openai.get_tracer")
    @patch("netra.instrumentation.libraries.openai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        instrumentor = NetraOpenAIInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        # chat x2, embeddings x2, responses x2
        assert mock_wrap_function.call_count == 6

    @patch("netra.instrumentation.libraries.openai.get_tracer")
    @patch("netra.instrumentation.libraries.openai.wrap_function_wrapper")
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

    @patch("netra.instrumentation.libraries.openai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all OpenAI methods it targets."""
        instrumentor = NetraOpenAIInstrumentor()

        instrumentor._uninstrument()

        # chat x2, completions x2, embeddings x2, responses x2
        assert mock_unwrap.call_count == 8


class TestWrappers:
    """Test wrapper functionality in the OpenAI instrumentation module."""

    @patch("netra.instrumentation.libraries.openai.wrappers.record_span_timing")
    def test_chat_wrapper_non_streaming(self, mock_record_timing):
        """Test chat_wrapper for non-streaming requests starts a span and returns the wrapped result."""
        from netra.instrumentation.libraries.openai.wrappers import chat_wrapper

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

    @patch("netra.instrumentation.libraries.openai.wrappers.StreamingWrapper")
    def test_chat_wrapper_streaming(self, mock_streaming_wrapper_class):
        """Test chat_wrapper for streaming requests wraps the response in StreamingWrapper."""
        from netra.instrumentation.libraries.openai.wrappers import chat_wrapper

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

    @patch("netra.instrumentation.libraries.openai.utils.context_api.get_value")
    def test_should_suppress_instrumentation_true(self, mock_get_value):
        """Test should_suppress_instrumentation returns True when suppression is enabled."""
        mock_get_value.return_value = True

        result = should_suppress_instrumentation()

        assert result is True

    @patch("netra.instrumentation.libraries.openai.utils.context_api.get_value")
    def test_should_suppress_instrumentation_false(self, mock_get_value):
        """Test should_suppress_instrumentation returns False when suppression is disabled."""
        mock_get_value.return_value = False

        result = should_suppress_instrumentation()

        assert result is False


class TestUsageAttributes:
    """Test _set_usage_attributes token capture across Chat and Responses shapes."""

    @staticmethod
    def _capture(usage):
        """Run _set_usage_attributes against a recording span and return {attr: value}."""
        from netra.instrumentation.libraries.openai.utils import _set_usage_attributes

        span = Mock()
        span.is_recording.return_value = True
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        _set_usage_attributes(span, usage)
        return captured

    def test_chat_completions_cache_read_and_write(self):
        """Chat Completions usage maps cached/cache_write tokens to read/creation attributes."""
        from opentelemetry.semconv_ai import SpanAttributes

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
        from netra.instrumentation.libraries.openai.utils import _set_response_message_attributes

        span = Mock()
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        _set_response_message_attributes(span, response_dict)
        return captured

    def _output(self, response_dict):
        return json.loads(self._capture(response_dict)["output"])

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

        output = self._output(response)

        assert len(output) == 1
        assert output[0]["role"] == "assistant"
        assert json.loads(output[0]["content"]) == {"name": "get_weather", "arguments": '{"location":"London"}'}
        assert output[0]["tool_call_id"] == "call_abc123"
        assert output[0]["finish_reason"] == "tool_calls"

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

        output = self._output(response)

        assert output[0] == {"role": "assistant", "content": "Let me check that for you."}
        assert json.loads(output[1]["content"]) == {"name": "lookup", "arguments": '{"q":"test"}'}
        assert output[1]["tool_call_id"] == "call_xyz"
        assert output[1]["finish_reason"] == "tool_calls"

    def test_normal_text_response_still_works(self):
        """A regular text response (no tool_calls) emits the assistant entry with its finish_reason."""
        response = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ]
        }

        assert self._output(response) == [{"role": "assistant", "content": "Hello there!", "finish_reason": "stop"}]

    def test_multiple_tool_calls_in_single_response(self):
        """Multiple tool_calls in one message produce sequential entries without an empty leader."""
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

        output = self._output(response)

        assert [entry["tool_call_id"] for entry in output] == ["call_1", "call_2"]
        assert all(entry["role"] == "assistant" for entry in output)
        assert "finish_reason" not in output[0]
        assert output[1]["finish_reason"] == "tool_calls"

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

        output = self._output(response)

        assert len(output) == 1
        assert "search" in output[0]["content"]
        assert output[0]["tool_call_id"] == "call_stream"
        assert output[0]["finish_reason"] == "tool_calls"

    def test_legacy_text_completion_is_captured(self):
        """Legacy Completions API choices carry their text in ``choices[].text``."""
        response = {"choices": [{"text": "Once upon a time", "finish_reason": "length"}]}

        assert self._output(response) == [
            {"role": "assistant", "content": "Once upon a time", "finish_reason": "length"}
        ]

    def test_response_without_messages_writes_no_output(self):
        """A response with no output messages (e.g. embeddings) leaves ``output`` untouched."""
        assert "output" not in self._capture({"data": [{"embedding": [0.1, 0.2]}], "usage": {}})


class TestChatCompletionInputToolCalls:
    """Test _set_chat_completion_input handling of tool-related messages."""

    @staticmethod
    def _capture(messages):
        from netra.instrumentation.libraries.openai.utils import _set_chat_completion_input

        span = Mock()
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        _set_chat_completion_input(span, messages)
        return captured

    def _input(self, messages):
        return json.loads(self._capture(messages)["input"])

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

        assert self._input(messages) == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

    def test_assistant_tool_calls_are_serialized(self):
        """Assistant messages with tool_calls produce entries with name/arguments JSON."""
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

        entries = self._input(messages)

        assert entries[0] == {"role": "user", "content": "What's the weather?"}
        assert entries[1]["role"] == "assistant"
        assert json.loads(entries[1]["content"]) == {"name": "get_weather", "arguments": '{"location":"London"}'}
        assert entries[1]["tool_call_id"] == "call_abc"
        assert entries[2] == {"role": "tool", "content": '{"temp": 22}', "tool_call_id": "call_abc"}

    def test_tool_call_id_captured_on_tool_messages(self):
        """Tool messages keep their tool_call_id in the input entry."""
        messages = [
            {"role": "tool", "tool_call_id": "call_xyz", "content": "result data"},
        ]

        assert self._input(messages) == [{"role": "tool", "content": "result data", "tool_call_id": "call_xyz"}]

    def test_pydantic_assistant_with_tool_calls(self):
        """A Pydantic ChatCompletionMessage with tool_calls is correctly converted and serialized."""
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

        entries = self._input(messages)

        assert [entry["role"] for entry in entries] == ["user", "assistant", "tool"]
        assert json.loads(entries[1]["content"]) == {"name": "get_weather", "arguments": '{"location":"Paris"}'}
        assert entries[1]["tool_call_id"] == "call_pydantic"
        assert entries[2]["tool_call_id"] == "call_pydantic"

    def test_none_content_message_is_skipped(self):
        """A message with content=None and no tool_calls is skipped entirely — no blank entry emitted."""
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "actual question"},
        ]

        assert self._input(messages) == [{"role": "user", "content": "actual question"}]

    def test_tool_call_conversation_keeps_message_order(self):
        """A full tool-call conversation is recorded in order with no gaps."""
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

        entries = self._input(messages)

        assert [entry["role"] for entry in entries] == ["user", "assistant", "user", "assistant", "tool"]
        assert entries[1]["content"] == "Hi!"
        assert entries[3]["tool_call_id"] == "call_1"
        assert entries[4]["tool_call_id"] == "call_1"

    def test_no_recordable_messages_writes_no_input(self):
        """Messages with no content or tool calls leave ``input`` untouched."""
        assert "input" not in self._capture([{"role": "user", "content": None}])
