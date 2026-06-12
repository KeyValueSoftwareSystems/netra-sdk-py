import unittest

from netra.instrumentation.groq.utils import _set_usage_attributes

from .fixtures.base_provider_utils import BaseProviderUtils


class TestGroqProviderUtils(unittest.TestCase, BaseProviderUtils):
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)
