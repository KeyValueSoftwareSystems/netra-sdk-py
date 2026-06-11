import itertools
import random
from collections.abc import Iterable
from typing import Any, Callable
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes


class MockMessageObject:
    def __init__(self, role: str, content: any):
        self.role = role
        self.content = content


class BaseProviderUtils:
    """Base class to be inherited by test cases that are going to test `utils.py` from `netra/instrumentation/openai/`, `netra/instrumentation/groq/` etc."""

    ALIASES = {
        "prompt_tokens": ["prompt_tokens", "input_tokens"],
        "completion_tokens": ["completion_tokens", "output_tokens"],
        "prompt_tokens_details": ["prompt_tokens_details", "input_tokens_details"],
    }

    P_TOKEN = 100
    C_TOKEN = 50
    P_DETAIL = 10
    T_TOKEN = 160

    ATTRIBUTE_MAPPINGS = {
        "model": SpanAttributes.LLM_REQUEST_MODEL,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_completion_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "max_tokens_to_sample": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "reasoning_effort": SpanAttributes.LLM_REQUEST_REASONING_EFFORT,
        "frequency_penalty": SpanAttributes.LLM_FREQUENCY_PENALTY,
        "presence_penalty": SpanAttributes.LLM_PRESENCE_PENALTY,
        "stop": SpanAttributes.LLM_CHAT_STOP_SEQUENCES,
        "stream": SpanAttributes.LLM_IS_STREAMING,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    }

    set_request_attributes_method: Callable = None
    set_response_attributes_method: Callable = None
    _set_usage_attributes_method: Callable = None
    _set_chat_input_method: Callable = None
    _set_response_message_attributes_method: Callable = None

    def __build_input_data(self):
        keys_groups = [
            self.ALIASES["prompt_tokens"],
            self.ALIASES["completion_tokens"],
            self.ALIASES["prompt_tokens_details"],
        ]

        for p_token, c_token, p_detail in itertools.product(*keys_groups):
            data = dict()

            data[p_token] = self.P_TOKEN
            data[c_token] = self.C_TOKEN
            data[p_detail] = {"cached_tokens": self.P_DETAIL}
            data["total_tokens"] = self.T_TOKEN

            yield data

    def __build_no_details_data(self):
        keys_group = [self.ALIASES["prompt_tokens"], self.ALIASES["completion_tokens"]]

        for p_token, c_token in itertools.product(*keys_group):
            data = dict()

            data[p_token] = self.P_TOKEN
            data[c_token] = self.C_TOKEN
            data["total_tokens"] = self.T_TOKEN

            yield data

    def test_set_usage_attributes(self):
        for dummy_data in self.__build_input_data():
            mock_span = MagicMock()
            self._set_usage_attributes_method(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, self.P_DETAIL)

    def test_set_usage_attributes_no_prompt_tokens_details(self):
        for dummy_data in self.__build_no_details_data():
            mock_span = MagicMock()
            self._set_usage_attributes_method(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)

    def test_empty_dict(self):
        mock_span = MagicMock()
        self._set_usage_attributes_method(mock_span, dict())

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertEqual(len(called_keys), 0)

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

        self.__set_chat_input_check(kwargs["messages"], kwargs["prompt"])

    def __set_chat_input_check(self, messages: list[str], prompt: str):
        mock_span = MagicMock()

        self._set_chat_input_method(mock_span, messages, prompt)

        for i, message in enumerate(messages):
            if isinstance(message, MockMessageObject):
                mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.role)
                mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.{i}.content", message.content)
            else:
                mock_span.set_attribute.assert_any_call(
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message["role"] if "role" in message else "user"
                )
                mock_span.set_attribute.assert_any_call(
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content", str(message["content"])
                )

    def test_set_chat_input_check(self):
        messages_object = [
            MockMessageObject(role="system", content="Initialize core instructions."),
            MockMessageObject(role="user", content="Explain quantum computing simply."),
            MockMessageObject(role="assistant", content="Quantum computing uses qubits..."),
        ]
        prompt_dummy = "Test message"
        self.__set_chat_input_check(messages_object, prompt_dummy)

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
        self.__set_chat_input_check(messages_dict, prompt_dummy)

    def __set_response_message_attributes_check(self, response_dict: dict[str, Any]):
        mock_span = MagicMock()
        self._set_response_message_attributes_method(mock_span, response_dict)

        if choices := response_dict.get("choices"):
            self.assertTrue(isinstance(choices, Iterable))

            message_index = 0
            for choice in choices:
                message = None
                if _message := choice.get("message"):
                    message = _message
                elif delta := choice.get("delta"):
                    message = delta

                if message is not None:
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", message.get("role", "assistant")
                    )
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", message.get("content", "")
                    )

                    message_index += 1

                if finish_reason := choice.get("finish_reason"):
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.finish_reason", finish_reason
                    )

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
            self.__set_response_message_attributes_check(case)

    def test_set_response_attributes(self):
        mock_span_1 = MagicMock()
        mock_span_1.is_recording = lambda: False
        self.set_response_attributes_method(mock_span_1, dict())
        self.assertEqual(0, mock_span_1.set_attribute.call_count)

        mock_span_2 = MagicMock()
        mock_span_1.is_recording = lambda: True
        self.set_response_attributes_method(mock_span_2, {"model": "test_model_name"})
        mock_span_2.set_attribute.assert_any_call(SpanAttributes.LLM_RESPONSE_MODEL, "test_model_name")
