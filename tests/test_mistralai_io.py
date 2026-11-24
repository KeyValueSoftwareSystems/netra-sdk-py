"""Tests for the structured ``input``/``output`` built by the MistralAI instrumentation."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from opentelemetry.semconv_ai import LLMRequestTypeValues

MODULE = "netra.instrumentation.libraries.mistralai"


@pytest.fixture
def mistral():
    """Import the instrumentation with the ``mistralai`` types it needs stubbed out."""
    sdk, models = ModuleType("mistralai"), ModuleType("mistralai.models")
    sdk.AssistantMessage = sdk.ChatCompletionChoice = sdk.UsageInfo = object
    models.ChatCompletionResponse = object
    with patch.dict(sys.modules, {"mistralai": sdk, "mistralai.models": models}):
        for name in [name for name in sys.modules if name.startswith(MODULE)]:
            del sys.modules[name]
        yield importlib.import_module(MODULE)


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def test_chat_input_keeps_roles_and_encodes_chunk_content(mistral):
    """Message objects keep their role and chunk-list content is JSON-encoded."""
    span = _span()
    chunks = [{"type": "text", "text": "Hi"}]
    messages = [SimpleNamespace(role="system", content="Be brief"), {"role": "user", "content": chunks}]

    mistral._set_input_attributes(span, LLMRequestTypeValues.CHAT, {}, {"messages": messages})

    assert json.loads(span.attributes["input"]) == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": json.dumps(chunks)},
    ]


def test_chat_output_encodes_chunk_content_with_finish_reason(mistral):
    """A choice with chunk-list content becomes one assistant entry carrying the finish reason."""
    span = _span()
    chunks = [{"type": "text", "text": "Hello"}]
    response = {"id": "r1", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": chunks}}]}

    mistral._set_response_attributes(span, LLMRequestTypeValues.CHAT, response)

    assert json.loads(span.attributes["output"]) == [
        {"role": "assistant", "content": json.dumps(chunks), "finish_reason": "stop"}
    ]
