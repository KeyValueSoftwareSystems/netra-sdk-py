"""
Unit tests for NetraOpenAIInstrumentor class.
Focuses on core functionality and happy path scenarios.
"""

import json
from typing import Collection
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.semconv_ai import SpanAttributes

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
        # chat x2, embeddings x2, responses x2, decisions x2
        assert mock_wrap_function.call_count == 8

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
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.libraries.openai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all OpenAI methods it targets."""
        instrumentor = NetraOpenAIInstrumentor()

        instrumentor._uninstrument()

        # chat x2, completions x2, embeddings x2, responses x2, decisions x2
        assert mock_unwrap.call_count == 10


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


TTL_30M_ATTRIBUTE = "gen_ai.usage.cache_creation_input_tokens_30m"
TTL_12H_ATTRIBUTE = "gen_ai.usage.cache_creation_input_tokens_12h"
RESPONSES_USAGE_WITH_TTL_SPLIT = {
    "input_tokens": 12000,
    "output_tokens": 500,
    "total_tokens": 12500,
    "input_tokens_details": {
        "cached_tokens": 4000,
        "cache_write_tokens": 7000,
        "cache_write_tokens_details": {"ttl_30m_tokens": 3000, "ttl_12h_tokens": 4000},
    },
}


class TestCacheWriteTtlAttributes:
    """Test capture of OpenAI's per-TTL cache-write split (GPT-5.6+)."""

    @staticmethod
    def _capture(usage):
        return TestUsageAttributes._capture(usage)

    def test_responses_api_cache_write_ttl_split_sets_tier_attributes(self):
        attrs = self._capture(RESPONSES_USAGE_WITH_TTL_SPLIT)

        assert attrs[TTL_30M_ATTRIBUTE] == 3000
        assert attrs[TTL_12H_ATTRIBUTE] == 4000
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 7000

    def test_chat_completions_cache_write_ttl_split_sets_tier_attributes(self):
        attrs = self._capture(
            {
                "prompt_tokens": 12000,
                "completion_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 4000,
                    "cache_write_tokens": 7000,
                    "cache_write_tokens_details": {"ttl_30m_tokens": 7000, "ttl_12h_tokens": 0},
                },
            }
        )

        assert attrs[TTL_30M_ATTRIBUTE] == 7000
        assert attrs[TTL_12H_ATTRIBUTE] == 0

    def test_zero_valued_ttl_tiers_are_recorded(self):
        attrs = self._capture(
            {
                "input_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_write_tokens_details": {"ttl_30m_tokens": 0, "ttl_12h_tokens": 0},
                },
            }
        )

        assert attrs[TTL_30M_ATTRIBUTE] == 0
        assert attrs[TTL_12H_ATTRIBUTE] == 0

    def test_missing_ttl_breakdown_sets_no_tier_attributes(self):
        attrs = self._capture(
            {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 6}}
        )

        assert TTL_30M_ATTRIBUTE not in attrs
        assert TTL_12H_ATTRIBUTE not in attrs
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 6

    @pytest.mark.parametrize("breakdown", [None, "3000", [3000, 4000]])
    def test_non_dict_ttl_breakdown_is_ignored(self, breakdown):
        attrs = self._capture(
            {
                "input_tokens": 10,
                "input_tokens_details": {"cache_write_tokens": 6, "cache_write_tokens_details": breakdown},
            }
        )

        assert TTL_30M_ATTRIBUTE not in attrs
        assert TTL_12H_ATTRIBUTE not in attrs
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 6

    def test_openai_sdk_model_dump_preserves_ttl_breakdown(self):
        """The openai package does not type cache_write_tokens_details; model_dump must still keep it."""
        responses_types = pytest.importorskip("openai.types.responses")
        from netra.instrumentation.libraries.openai.utils import model_as_dict, set_response_attributes

        usage = responses_types.ResponseUsage.model_validate(
            {
                **RESPONSES_USAGE_WITH_TTL_SPLIT,
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        )
        span = Mock()
        span.is_recording.return_value = True
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)

        set_response_attributes(span, {"model": "gpt-5.6-terra", "usage": model_as_dict(usage)})

        assert captured[TTL_30M_ATTRIBUTE] == 3000
        assert captured[TTL_12H_ATTRIBUTE] == 4000

    def test_streaming_response_completed_carries_ttl_split(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from netra.instrumentation.libraries.openai.wrappers import StreamingWrapper

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        span = provider.get_tracer(__name__).start_span("openai.response")
        completed_event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": "gpt-5.6-terra",
                "output": [],
                "usage": RESPONSES_USAGE_WITH_TTL_SPLIT,
            },
        }

        class FakeStream:
            """Stands in for openai.Stream, which (unlike a list iterator) accepts the proxy's attributes."""

            def __init__(self, events):
                self._events = iter(events)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._events)

        list(StreamingWrapper(span, FakeStream([completed_event]), {"model": "gpt-5.6-terra", "input": "hi"}))

        (finished,) = exporter.get_finished_spans()
        assert finished.attributes[TTL_30M_ATTRIBUTE] == 3000
        assert finished.attributes[TTL_12H_ATTRIBUTE] == 4000


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
        from netra.instrumentation.libraries.openai.utils import _set_chat_completion_input

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


class TestRequestToolDefinitions:
    """Test set_request_attributes capture of the ``tools`` and ``tool_choice`` request args."""

    TOOLS_ATTRIBUTE = "gen_ai.request.tools"
    WEATHER_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

    @staticmethod
    def _capture(kwargs, operation_type="chat"):
        from netra.instrumentation.libraries.openai.utils import set_request_attributes

        span = Mock()
        span.is_recording.return_value = True
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        set_request_attributes(span, kwargs, operation_type)
        return captured

    def test_chat_completions_nested_function_tool_is_captured(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": self.WEATHER_SCHEMA},
            }
        ]

        attrs = self._capture({"model": "gpt-4o-mini", "messages": [], "tools": tools})

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [
            {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": self.WEATHER_SCHEMA}
        ]

    def test_tools_are_recorded_as_single_attribute_not_indexed_keys(self):
        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": self.WEATHER_SCHEMA}},
            {"type": "function", "function": {"name": "get_time"}},
        ]

        attrs = self._capture({"messages": [], "tools": tools})

        assert [key for key in attrs if "tools" in key or "functions" in key] == [self.TOOLS_ATTRIBUTE]

    def test_responses_flat_function_tool_and_builtin_tool_are_captured(self):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": self.WEATHER_SCHEMA,
            },
            {"type": "web_search_preview"},
        ]

        attrs = self._capture({"model": "gpt-4o-mini", "input": "hi", "tools": tools}, operation_type="response")

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": self.WEATHER_SCHEMA,
            },
            {"type": "web_search_preview"},
        ]

    def test_chat_completions_custom_tool_records_name_and_description(self):
        tools = [{"type": "custom", "custom": {"name": "run_sql", "description": "Run a query"}}]

        attrs = self._capture({"messages": [], "tools": tools})

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [
            {"type": "custom", "name": "run_sql", "description": "Run a query"}
        ]

    def test_model_object_tool_is_converted_via_model_dump(self):
        tool = Mock()
        tool.model_dump.return_value = {"type": "function", "name": "get_weather", "parameters": self.WEATHER_SCHEMA}

        attrs = self._capture({"input": "hi", "tools": [tool]}, operation_type="response")

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [
            {"type": "function", "name": "get_weather", "parameters": self.WEATHER_SCHEMA}
        ]

    def test_tool_without_type_is_skipped(self):
        tools = [{"name": "untyped"}, {"type": "file_search"}]

        attrs = self._capture({"input": "hi", "tools": tools}, operation_type="response")

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [{"type": "file_search"}]

    def test_only_untyped_tools_records_no_tools_attribute(self):
        attrs = self._capture({"input": "hi", "tools": [{"name": "untyped"}]}, operation_type="response")

        assert self.TOOLS_ATTRIBUTE not in attrs

    def test_unserialisable_parameters_do_not_raise(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"enum": {"a"}}}}]

        attrs = self._capture({"messages": [], "tools": tools})

        assert json.loads(attrs[self.TOOLS_ATTRIBUTE]) == [
            {"type": "function", "name": "f", "parameters": {"enum": "{'a'}"}}
        ]

    def test_no_tools_records_no_tool_attributes(self):
        attrs = self._capture({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})

        assert self.TOOLS_ATTRIBUTE not in attrs
        assert "llm.request.tool_choice" not in attrs

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ("auto", "auto"),
            ("required", "required"),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                '{"type": "function", "function": {"name": "get_weather"}}',
            ),
            ({"type": "function", "name": "get_weather"}, '{"type": "function", "name": "get_weather"}'),
        ],
    )
    def test_tool_choice_is_captured(self, tool_choice, expected):
        attrs = self._capture({"messages": [], "tool_choice": tool_choice})

        assert attrs["llm.request.tool_choice"] == expected

    def test_tool_choice_sentinel_is_not_recorded(self):
        # Stands in for openai's NOT_GIVEN, which callers forward explicitly when unset.
        not_given = object()

        attrs = self._capture({"messages": [], "tools": not_given, "tool_choice": not_given})

        assert "llm.request.tool_choice" not in attrs
        assert self.TOOLS_ATTRIBUTE not in attrs


class TestToolAndCacheTierCaptureNeverFailsTheCall:
    """A failure while capturing tools or cache-write tiers must not abort or fail the user's OpenAI call."""

    @staticmethod
    def _circular_schema():
        schema: dict = {"type": "object"}
        schema["properties"] = {"self": schema}
        return schema

    @staticmethod
    def _run_wrapper(wrapper_factory, kwargs, response, span=None):
        """Invoke a non-streaming wrapper against a recording span; return (result, wrapped, captured attrs)."""
        span = span or Mock()
        span.is_recording.return_value = True
        captured: dict = {}
        if span.set_attribute.side_effect is None:
            span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        tracer = Mock()
        span_context = MagicMock()
        span_context.__enter__.return_value = span
        tracer.start_as_current_span.return_value = span_context
        wrapped = Mock(return_value=response)

        with patch("netra.instrumentation.libraries.openai.wrappers.record_span_timing"):
            result = wrapper_factory(tracer)(wrapped, Mock(), (), kwargs)

        wrapped.assert_called_once_with(**kwargs)
        return result, captured

    def test_chat_call_succeeds_when_tool_schema_cannot_be_serialised(self):
        from netra.instrumentation.libraries.openai.wrappers import chat_wrapper

        response = {"model": "gpt-4o-mini", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": self._circular_schema()}}],
            "tool_choice": "auto",
        }

        result, attrs = self._run_wrapper(chat_wrapper, kwargs, response)

        assert result is response
        assert "gen_ai.request.tools" not in attrs
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "ok"

    def test_response_call_succeeds_when_tool_choice_cannot_be_serialised(self):
        from netra.instrumentation.libraries.openai.wrappers import responses_wrapper

        tool_choice: dict = {"type": "function", "name": "f"}
        tool_choice["self"] = tool_choice
        response = {"model": "gpt-4o-mini", "output_text": "ok"}
        kwargs = {"model": "gpt-4o-mini", "input": "hi", "tool_choice": tool_choice}

        result, attrs = self._run_wrapper(responses_wrapper, kwargs, response)

        assert result is response
        assert "llm.request.tool_choice" not in attrs
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "ok"

    def test_response_call_succeeds_when_tool_model_dump_raises(self):
        from netra.instrumentation.libraries.openai.wrappers import responses_wrapper

        tool = Mock()
        tool.model_dump.side_effect = ValueError("cannot dump")
        response = {"model": "gpt-4o-mini", "output_text": "ok"}
        kwargs = {"model": "gpt-4o-mini", "input": "hi", "tools": [tool]}

        result, attrs = self._run_wrapper(responses_wrapper, kwargs, response)

        assert result is response
        assert "gen_ai.request.tools" not in attrs

    def test_response_is_returned_when_cache_tier_capture_fails(self):
        from netra.instrumentation.libraries.openai.wrappers import responses_wrapper

        captured: dict = {}

        def set_attribute(key, value):
            if key == TTL_30M_ATTRIBUTE:
                raise RuntimeError("exporter rejected attribute")
            captured[key] = value

        span = Mock()
        span.set_attribute.side_effect = set_attribute
        response = {"model": "gpt-5.6-terra", "output_text": "ok", "usage": RESPONSES_USAGE_WITH_TTL_SPLIT}

        result, _ = self._run_wrapper(responses_wrapper, {"model": "gpt-5.6-terra", "input": "hi"}, response, span)

        assert result is response
        assert captured[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 7000
        assert captured[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 12500
        assert captured[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "ok"


DECISION_USAGE = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 150},
    "output_tokens": 12,
    "output_tokens_details": {"reasoning_tokens": 4},
    "total_tokens": 1212,
}

DECISION_ANSWERS = [
    {"type": "predicate", "name": "is_spam", "probability": 0.08},
    {
        "type": "choice",
        "name": "category",
        "choice": "billing",
        "confidence": 0.91,
        "probabilities": [{"value": "billing", "probability": 0.91}, {"value": "tech", "probability": 0.09}],
    },
    {
        "type": "score",
        "name": "urgency",
        "score": 2.4,
        "confidence": 0.7,
        "probabilities": [{"label": "low", "value": 1, "probability": 0.2}],
    },
    {"type": "refusal", "name": None},
]

DECISION_QUESTIONS = [
    {"type": "predicate", "name": "is_spam", "instructions": "Is this message spam?"},
    {
        "type": "choice",
        "name": "category",
        "instructions": "Which team should handle this?",
        "choices": [{"value": "billing"}, {"value": "tech", "description": "Technical issues"}],
    },
    {
        "type": "score",
        "name": "urgency",
        "instructions": "How urgent is this?",
        "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}],
    },
]


def _recording_span():
    """Return a mock recording span plus the dict its set_attribute calls write into."""
    span = Mock()
    span.is_recording.return_value = True
    captured: dict = {}
    span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
    return span, captured


class TestDecisionsInstrumentor:
    """Registration of the Decisions API wrappers on NetraOpenAIInstrumentor."""

    @patch("netra.instrumentation.libraries.openai.get_tracer")
    @patch("netra.instrumentation.libraries.openai.wrap_function_wrapper")
    def test_decisions_create_is_wrapped_sync_and_async(self, mock_wrap_function, mock_get_tracer):
        NetraOpenAIInstrumentor()._instrument()

        targets = [(c.args[0], c.args[1]) for c in mock_wrap_function.call_args_list]
        assert ("openai.resources.decisions", "Decisions.create") in targets
        assert ("openai.resources.decisions", "AsyncDecisions.create") in targets

    @patch("netra.instrumentation.libraries.openai.logger")
    @patch("netra.instrumentation.libraries.openai.get_tracer")
    @patch("netra.instrumentation.libraries.openai.wrap_function_wrapper")
    def test_openai_without_decisions_is_skipped_without_error(self, mock_wrap_function, mock_get_tracer, mock_logger):
        """Older openai releases have no decisions module; that must not be logged as a failure."""

        def wrap(module, name, wrapper):
            if module == "openai.resources.decisions":
                raise ModuleNotFoundError("No module named 'openai.resources.decisions'")

        mock_wrap_function.side_effect = wrap

        NetraOpenAIInstrumentor()._instrument()

        mock_logger.error.assert_not_called()
        # The other six targets were still wrapped.
        assert mock_wrap_function.call_count == 7

    @patch("netra.instrumentation.libraries.openai.unwrap")
    def test_uninstrument_unwraps_decision_classes(self, mock_unwrap):
        """unwrap() needs the owning class path; ``(module, "Class.method")`` would be a silent no-op."""
        NetraOpenAIInstrumentor()._uninstrument()

        targets = [(c.args[0], c.args[1]) for c in mock_unwrap.call_args_list]
        assert ("openai.resources.decisions.Decisions", "create") in targets
        assert ("openai.resources.decisions.AsyncDecisions", "create") in targets

    @patch("netra.instrumentation.libraries.openai.logger")
    @patch("netra.instrumentation.libraries.openai.unwrap")
    def test_uninstrument_without_decisions_is_quiet(self, mock_unwrap, mock_logger):
        def unwrap(obj, attr):
            if str(obj).startswith("openai.resources.decisions"):
                raise ImportError("No module named 'openai.resources.decisions'")

        mock_unwrap.side_effect = unwrap

        NetraOpenAIInstrumentor()._uninstrument()

        assert not any("ecision" in str(c) for c in mock_logger.error.call_args_list)


class TestDecisionRequestAttributes:
    """set_request_attributes for operation_type="decision"."""

    @staticmethod
    def _capture(kwargs):
        from netra.instrumentation.libraries.openai.utils import set_request_attributes

        span, captured = _recording_span()
        set_request_attributes(span, kwargs, "decision")
        return captured

    def test_model_and_request_type_are_recorded(self):
        attrs = self._capture({"model": "gpt-6-luna", "input": "hi", "questions": DECISION_QUESTIONS})

        assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "decision"
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-6-luna"

    def test_string_input_is_one_user_prompt(self):
        attrs = self._capture({"model": "gpt-6-luna", "input": "Refund me now!", "questions": DECISION_QUESTIONS})

        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Refund me now!"
        assert f"{SpanAttributes.LLM_PROMPTS}.1.role" not in attrs

    def test_questions_are_recorded_as_json_array(self):
        attrs = self._capture({"model": "gpt-6-luna", "input": "hi", "questions": DECISION_QUESTIONS})

        assert json.loads(attrs["gen_ai.decision.questions"]) == DECISION_QUESTIONS

    def test_mixed_input_items_are_recorded_in_order(self):
        input_items = [
            {"role": "system", "content": "You triage support tickets."},
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "input_text", "text": "Card charged twice"}],
            },
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"id": 7}'},
            {"type": "function_call_output", "call_id": "c1", "output": "2 charges found"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "grep", "input": "charge"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "ok"}]},
            {"role": "assistant", "content": "Looking into it."},
        ]

        attrs = self._capture({"model": "gpt-6-luna", "input": input_items, "questions": DECISION_QUESTIONS})

        p = SpanAttributes.LLM_PROMPTS
        assert [attrs[f"{p}.{i}.role"] for i in range(7)] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ]
        assert attrs[f"{p}.0.content"] == "You triage support tickets."
        assert json.loads(attrs[f"{p}.1.content"]) == [{"type": "input_text", "text": "Card charged twice"}]
        assert json.loads(attrs[f"{p}.2.content"]) == {"name": "lookup", "arguments": '{"id": 7}'}
        assert attrs[f"{p}.2.tool_call_id"] == "c1"
        assert attrs[f"{p}.3.content"] == "2 charges found"
        assert attrs[f"{p}.3.tool_call_id"] == "c1"
        assert json.loads(attrs[f"{p}.4.content"]) == {"name": "grep", "arguments": "charge"}
        assert attrs[f"{p}.5.tool_call_id"] == "c2"
        assert f"{p}.0.tool_call_id" not in attrs
        assert attrs[f"{p}.6.content"] == "Looking into it."

    def test_one_shot_iterables_are_not_consumed(self):
        """The SDK accepts any iterable; reading a generator here would empty the real request."""
        input_items = (m for m in [{"role": "user", "content": "hi"}])
        questions = (q for q in DECISION_QUESTIONS)

        attrs = self._capture({"model": "gpt-6-luna", "input": input_items, "questions": questions})

        assert len(list(input_items)) == 1
        assert len(list(questions)) == len(DECISION_QUESTIONS)
        assert f"{SpanAttributes.LLM_PROMPTS}.0.role" not in attrs
        assert "gen_ai.decision.questions" not in attrs

    def test_unserialisable_question_does_not_raise(self):
        question: dict = {"type": "predicate", "instructions": "x"}
        question["self"] = question

        attrs = self._capture({"model": "gpt-6-luna", "input": "hi", "questions": [question]})

        assert "gen_ai.decision.questions" not in attrs
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "hi"


class TestDecisionResponseAttributes:
    """set_response_attributes for a Decisions API response."""

    @staticmethod
    def _capture(response_dict):
        from netra.instrumentation.libraries.openai.utils import set_response_attributes

        span, captured = _recording_span()
        set_response_attributes(span, response_dict)
        return captured

    def test_answers_are_recorded_as_attribute_and_completion(self):
        attrs = self._capture({"model": "gpt-6-luna", "answers": DECISION_ANSWERS, "usage": DECISION_USAGE})

        assert json.loads(attrs["gen_ai.decision.answers"]) == DECISION_ANSWERS
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
        assert json.loads(attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]) == DECISION_ANSWERS
        assert attrs[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-6-luna"

    def test_usage_is_recorded(self):
        attrs = self._capture({"model": "gpt-6-luna", "answers": DECISION_ANSWERS, "usage": DECISION_USAGE})

        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 1200
        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 12
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 1000
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 150
        assert attrs[SpanAttributes.LLM_USAGE_REASONING_TOKENS] == 4
        assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 1212

    def test_non_decision_response_records_no_answers(self):
        attrs = self._capture({"model": "gpt-4o-mini", "output_text": "ok"})

        assert "gen_ai.decision.answers" not in attrs
        assert attrs[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == "ok"


class TestDecisionWrappers:
    """decisions_wrapper / adecisions_wrapper span lifecycle."""

    KWARGS = {"model": "gpt-6-luna", "input": "Card charged twice", "questions": DECISION_QUESTIONS}
    RESPONSE = {"model": "gpt-6-luna", "answers": DECISION_ANSWERS, "usage": DECISION_USAGE}

    @staticmethod
    def _tracer():
        span, captured = _recording_span()
        tracer = Mock()
        span_context = MagicMock()
        span_context.__enter__.return_value = span
        tracer.start_as_current_span.return_value = span_context
        return tracer, span, captured

    def test_sync_wrapper_records_request_and_response(self):
        from netra.instrumentation.libraries.openai.wrappers import DECISION_SPAN_NAME, decisions_wrapper

        tracer, span, attrs = self._tracer()
        wrapped = Mock(return_value=self.RESPONSE)

        with patch("netra.instrumentation.libraries.openai.wrappers.record_span_timing"):
            result = decisions_wrapper(tracer)(wrapped, Mock(), (), dict(self.KWARGS))

        assert result is self.RESPONSE
        wrapped.assert_called_once_with(**self.KWARGS)
        assert tracer.start_as_current_span.call_args.args[0] == DECISION_SPAN_NAME == "openai.decision"
        assert tracer.start_as_current_span.call_args.kwargs["attributes"] == {"llm.request.type": "decision"}
        assert attrs[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == "Card charged twice"
        assert json.loads(attrs["gen_ai.decision.answers"]) == DECISION_ANSWERS

    def test_async_wrapper_records_request_and_response(self):
        import asyncio

        from netra.instrumentation.libraries.openai.wrappers import adecisions_wrapper

        tracer, span, attrs = self._tracer()

        async def wrapped(*args, **kwargs):
            return self.RESPONSE

        with patch("netra.instrumentation.libraries.openai.wrappers.record_span_timing"):
            result = asyncio.run(adecisions_wrapper(tracer)(wrapped, Mock(), (), dict(self.KWARGS)))

        assert result is self.RESPONSE
        assert attrs[SpanAttributes.LLM_REQUEST_TYPE] == "decision"
        assert json.loads(attrs["gen_ai.decision.answers"]) == DECISION_ANSWERS

    def test_api_error_marks_span_and_reraises(self):
        from opentelemetry.trace.status import StatusCode

        from netra.instrumentation.libraries.openai.wrappers import decisions_wrapper

        tracer, span, _ = self._tracer()
        wrapped = Mock(side_effect=RuntimeError("400 bad questions"))

        with pytest.raises(RuntimeError, match="400 bad questions"):
            decisions_wrapper(tracer)(wrapped, Mock(), (), dict(self.KWARGS))

        assert span.set_status.call_args.args[0].status_code == StatusCode.ERROR

    @patch("netra.instrumentation.libraries.openai.wrappers.should_suppress_instrumentation", return_value=True)
    def test_suppressed_instrumentation_skips_span(self, _):
        from netra.instrumentation.libraries.openai.wrappers import decisions_wrapper

        tracer, _, _ = self._tracer()
        wrapped = Mock(return_value=self.RESPONSE)

        assert decisions_wrapper(tracer)(wrapped, Mock(), (), dict(self.KWARGS)) is self.RESPONSE
        tracer.start_as_current_span.assert_not_called()


class TestDecisionsAgainstRealSdk:
    """End-to-end through the real openai Decisions resource (skipped on openai releases without it)."""

    def test_sync_and_async_create_emit_decision_spans(self):
        import asyncio

        decisions_module = pytest.importorskip("openai.resources.decisions")
        from openai.types.decision import Decision
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        decision = Decision.model_validate(
            {"model": "gpt-6-luna", "answers": DECISION_ANSWERS, "usage": DECISION_USAGE}
        )

        async def apost(*args, **kwargs):
            return decision

        # The resources bind ``_post`` to ``client.post`` at construction, so the HTTP layer is the client mock.
        sync_client, async_client = Mock(), Mock()
        sync_client.post.return_value = decision
        async_client.post.side_effect = apost
        sync_resource = decisions_module.Decisions(sync_client)
        async_resource = decisions_module.AsyncDecisions(async_client)
        instrumentor = NetraOpenAIInstrumentor()
        instrumentor._instrument(tracer_provider=provider)
        try:
            assert sync_resource.create(**TestDecisionWrappers.KWARGS) is decision
            assert asyncio.run(async_resource.create(**TestDecisionWrappers.KWARGS)) is decision
        finally:
            instrumentor._uninstrument()

        assert sync_client.post.call_args.args[0] == "/decisions"

        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["openai.decision", "openai.decision"]
        for span in spans:
            assert span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "decision"
            assert span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-6-luna"
            assert span.attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS] == 150
            answers = json.loads(span.attributes["gen_ai.decision.answers"])
            assert [a["type"] for a in answers] == ["predicate", "choice", "score", "refusal"]
        # Uninstrument really removed the wrappers.
        assert not hasattr(decisions_module.Decisions.create, "__wrapped__")
