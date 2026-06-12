import unittest
from unittest.mock import MagicMock

from netra.processors.span_io_processor import SpanIOProcessor


class TestSpanIOProcessor(unittest.TestCase):
    """Test cases for `SpanIOProcessor` as per `4.3` in PRD"""

    _USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    _USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    _USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    _USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"

    def _get_mocks(self, original_value: str, desired_value: str):
        mock_span = MagicMock()
        mock_original_set_attribute = MagicMock()
        mock_span.set_attribute = mock_original_set_attribute
        # creating fake span

        SpanIOProcessor._wrap_set_attribute(mock_span)
        # adding set_attribute method to span

        mock_span.set_attribute(original_value, 100)
        # setting value

        mock_original_set_attribute.assert_called_with(desired_value, 100)

    def test_alias_to_prompt_tokens(self):
        self._get_mocks(self._USAGE_INPUT_TOKENS, self._USAGE_PROMPT_TOKENS)

    def test_alias_to_completion_tokens(self):
        self._get_mocks(self._USAGE_OUTPUT_TOKENS, self._USAGE_COMPLETION_TOKENS)

    def test_pass_through_prompt_tokens(self):
        self._get_mocks(self._USAGE_PROMPT_TOKENS, self._USAGE_PROMPT_TOKENS)

    def test_pass_through_completion_tokens(self):
        self._get_mocks(self._USAGE_COMPLETION_TOKENS, self._USAGE_COMPLETION_TOKENS)
