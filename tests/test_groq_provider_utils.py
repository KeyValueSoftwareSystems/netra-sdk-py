import random
import unittest
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.groq.utils import (
    _set_chat_input,
    _set_response_message_attributes,
    _set_usage_attributes,
    set_request_attributes,
    set_response_attributes,
)

from .fixtures.base_provider_utils import BaseProviderUtils, MockMessageObject


class TestGroqProviderUtils(unittest.TestCase, BaseProviderUtils):
    set_request_attributes_method = staticmethod(set_request_attributes)
    set_response_attributes_method = staticmethod(set_response_attributes)
    _set_chat_input_method = staticmethod(_set_chat_input)
    _set_response_message_attributes_method = staticmethod(_set_response_message_attributes)
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)

    def test_set_chat_input_check(self):
        messages_object = [
            MockMessageObject(role="system", content="Initialize core instructions."),
            MockMessageObject(role="user", content="Explain quantum computing simply."),
            MockMessageObject(role="assistant", content="Quantum computing uses qubits..."),
        ]
        prompt_dummy = "Test message"
        self._set_chat_input_check(messages_object, prompt_dummy)

        messages_dict = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                # Missing 'role' key completely -> tests the fallback to "user"
                "content": "This should default to a user role."
            },
            {
                "role": "assistant",
                # Non-string content -> tests the str(content) conversion block
                "content": ["Nested list content", 12345],
            },
        ]
        prompt_dummy = None
        self._set_chat_input_check(messages_dict, prompt_dummy)

    def test_set_request_attributes(self):
        OP_TYPE = "OP_TYPE"
        mock_span = MagicMock()
        samples = random.sample(list(self.ATTRIBUTE_MAPPINGS.keys()), k=random.randint(1, len(self.ATTRIBUTE_MAPPINGS)))
        kwargs = {sample: "mock" for sample in samples}
        # picking a random sample from kwargs

        kwargs["messages"] = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Test"}]

        kwargs["prompt"] = "Test Prompt"

        self.set_request_attributes_method(mock_span, kwargs, OP_TYPE)

        for key, value in kwargs.items():
            if key in self.ATTRIBUTE_MAPPINGS:
                mock_span.set_attribute.assert_any_call(self.ATTRIBUTE_MAPPINGS[key], value)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_REQUEST_TYPE, OP_TYPE)

        self._set_chat_input_check(kwargs["messages"], kwargs["prompt"])

    def test_set_response_message_attributes(self):
        # Test Case 1: Standard Complete Response (Unary Block)
        unary_success_data = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "The capital of France is Paris."},
                    "finish_reason": "stop",
                }
            ]
        }

        # Test Case 2: Streaming Chunk Response (Delta Block)
        streaming_success_data = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Par"},
                    "finish_reason": None,  # Streams often pass null finish reasons mid-flight
                }
            ]
        }

        # Test Case 3: Multiple Choices Response (n > 1)
        # Tests that message_index tracks and increases across separate array objects
        multiple_choices_data = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Option A text"}, "finish_reason": "stop"},
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Option B alternative text"},
                    "finish_reason": "length",
                },
            ]
        }

        max_length_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "This sentence is cut off mid-way because the"},
                    "finish_reason": "length",
                }
            ]
        }

        tool_call_response = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "Kochi"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        multi_stream_response = {
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Running option one"}, "finish_reason": None},
                {
                    "index": 1,
                    "delta": {"role": "assistant", "content": "Alternative route processing"},
                    "finish_reason": "stop",
                },
            ]
        }

        cases = [
            unary_success_data,
            streaming_success_data,
            multiple_choices_data,
            max_length_response,
            tool_call_response,
            multi_stream_response,
        ]

        for case in cases:
            self._set_response_message_attributes_check(case)
