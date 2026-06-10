import itertools
import unittest
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.groq.utils import _set_usage_attributes


class TestGroqProviderUtils(unittest.TestCase):
    """Tests `_set_usage_attributes` from `groq.utils`"""

    ALIASES = {
        "prompt_tokens": ["prompt_tokens", "input_tokens"],
        "completion_tokens": ["completion_tokens", "output_tokens"],
        "prompt_tokens_details": ["prompt_tokens_details", "input_tokens_details"],
    }

    P_TOKEN = 100
    C_TOKEN = 50
    P_DETAIL = 10
    T_TOKEN = 160

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
            _set_usage_attributes(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, self.P_DETAIL)

    def test_set_usage_attributes_no_prompt_tokens_details(self):
        for dummy_data in self.__build_no_details_data():
            mock_span = MagicMock()
            _set_usage_attributes(mock_span, dummy_data)

            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, self.P_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, self.C_TOKEN)
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, self.T_TOKEN)

    def test_empty_dict(self):
        mock_span = MagicMock()
        _set_usage_attributes(mock_span, dict())

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertEqual(len(called_keys), 0)
