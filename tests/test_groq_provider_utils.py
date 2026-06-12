import unittest

from netra.instrumentation.groq.utils import (
    _set_chat_input,
    _set_response_message_attributes,
    _set_usage_attributes,
    set_request_attributes,
    set_response_attributes,
)

from .fixtures.base_provider_utils import BaseProviderUtils


class TestGroqProviderUtils(unittest.TestCase, BaseProviderUtils):
    set_request_attributes_method = staticmethod(set_request_attributes)
    set_response_attributes_method = staticmethod(set_response_attributes)
    _set_chat_input_method = staticmethod(_set_chat_input)
    _set_response_message_attributes_method = staticmethod(_set_response_message_attributes)
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)
