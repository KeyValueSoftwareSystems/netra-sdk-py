"""Tests for the structured ``input``/``output`` built by the Groq instrumentation."""

import json
from unittest.mock import Mock

from netra.instrumentation.libraries.groq.utils import set_request_attributes, set_response_attributes

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
TOOL_CALL_ENTRY = {
    "role": "assistant",
    "content": json.dumps({"name": "get_weather", "arguments": '{"city": "Paris"}'}),
    "tool_call_id": "c1",
}


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def test_chat_input_includes_tool_calls_and_results():
    """Assistant tool calls and tool results become tool-call and tool-result entries."""
    span = _span()
    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        {"role": "tool", "content": "20C", "tool_call_id": "c1"},
    ]

    set_request_attributes(span, {"model": "llama-3", "messages": messages}, "chat")

    assert json.loads(span.attributes["input"]) == [
        {"role": "user", "content": "Weather in Paris?"},
        TOOL_CALL_ENTRY,
        {"role": "tool", "content": "20C", "tool_call_id": "c1"},
    ]


def test_chat_output_tool_call_carries_finish_reason():
    """A tool-call choice becomes a tool-call entry carrying the finish reason."""
    span = _span()
    response = {
        "model": "llama-3",
        "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [TOOL_CALL]}}],
    }

    set_response_attributes(span, response)

    assert json.loads(span.attributes["output"]) == [{**TOOL_CALL_ENTRY, "finish_reason": "tool_calls"}]
