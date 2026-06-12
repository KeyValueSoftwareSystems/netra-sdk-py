import itertools
from typing import Callable
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

    def _build_input_data(self):
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

    def _build_no_details_data(self):
        keys_group = [self.ALIASES["prompt_tokens"], self.ALIASES["completion_tokens"]]

        for p_token, c_token in itertools.product(*keys_group):
            data = dict()

            data[p_token] = self.P_TOKEN
            data[c_token] = self.C_TOKEN
            data["total_tokens"] = self.T_TOKEN

            yield data

    def test_set_usage_attributes(self):
        """Tests _set_usage_attributes"""
        for dummy_data in self._build_input_data():
            mock_span = MagicMock()
            self._set_usage_attributes_method(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, self.P_DETAIL)

    def test_set_usage_attributes_no_prompt_tokens_details(self):
        """Tests _set_usage_attributes without prompt token details"""
        for dummy_data in self._build_no_details_data():
            mock_span = MagicMock()
            self._set_usage_attributes_method(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)

    def test_empty_dict(self):
        """Tests _set_usage_attributes with empty dictionary"""
        mock_span = MagicMock()
        self._set_usage_attributes_method(mock_span, dict())

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertEqual(len(called_keys), 0)
