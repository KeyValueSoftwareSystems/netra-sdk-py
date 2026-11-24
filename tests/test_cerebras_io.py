"""Tests for the structured ``input``/``output`` built by the Cerebras instrumentation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from opentelemetry.semconv_ai import LLMRequestTypeValues

from netra.instrumentation.libraries.cerebras.utils import set_request_attributes, set_response_attributes


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def test_chat_input_keeps_roles():
    """Chat messages become entries with their roles."""
    span = _span()
    messages = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]

    set_request_attributes(span, LLMRequestTypeValues.CHAT, {"model": "llama3.1-8b", "messages": messages})

    assert json.loads(span.attributes["input"]) == messages


def test_completion_prompt_becomes_user_entry():
    """A completion prompt becomes a single user entry."""
    span = _span()

    set_request_attributes(span, LLMRequestTypeValues.COMPLETION, {"prompt": "Once upon"})

    assert json.loads(span.attributes["input"]) == [{"role": "user", "content": "Once upon"}]


def test_chat_output_from_response_object():
    """A response object's choice becomes an assistant entry carrying the finish reason."""
    span = _span()
    response_dict = {
        "id": "r1",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
    }

    set_response_attributes(span, SimpleNamespace(model_dump=lambda: response_dict))

    assert json.loads(span.attributes["output"]) == [{"role": "assistant", "content": "Hi", "finish_reason": "stop"}]
