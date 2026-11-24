"""Tests for the structured ``input``/``output`` built by the Agno instrumentation."""

import json
from types import SimpleNamespace

from netra.instrumentation.libraries.agno.utils import format_messages_as_input, format_response_as_output

TOOL_CALLS = [
    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
    {"id": "c2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
]
TOOL_CALL_ENTRIES = [
    {
        "role": "assistant",
        "content": json.dumps({"name": "get_weather", "arguments": '{"city": "Paris"}'}),
        "tool_call_id": "c1",
    },
    {"role": "assistant", "content": json.dumps({"name": "get_time", "arguments": "{}"}), "tool_call_id": "c2"},
]


def test_input_assistant_tool_calls_become_tool_call_entries():
    """An assistant message carrying only tool calls becomes one entry per call."""
    messages = [
        SimpleNamespace(role="system", content="Be brief"),
        SimpleNamespace(role="user", content="Weather and time in Paris?"),
        SimpleNamespace(role="assistant", content=None, tool_calls=TOOL_CALLS),
    ]

    assert json.loads(format_messages_as_input(messages)) == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Weather and time in Paris?"},
        *TOOL_CALL_ENTRIES,
    ]


def test_output_tool_calls_become_tool_call_entries():
    """A response with tool calls and no content becomes one entry per call."""
    response = SimpleNamespace(content=None, tool_calls=TOOL_CALLS)

    assert json.loads(format_response_as_output(response)) == TOOL_CALL_ENTRIES


def test_output_text_becomes_assistant_entry():
    """A response with text content becomes a single assistant entry."""
    response = SimpleNamespace(content="Sunny", tool_calls=None)

    assert json.loads(format_response_as_output(response)) == [{"role": "assistant", "content": "Sunny"}]
