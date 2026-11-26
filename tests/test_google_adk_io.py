"""Tests for the structured ``input``/``output`` built by the Google ADK instrumentation."""

import json
from types import SimpleNamespace

from netra.instrumentation.libraries.google_adk.utils import (
    extract_llm_request_attributes,
    extract_llm_response_attributes,
)


def _call(call_id, name, args):
    return {"role": "assistant", "content": json.dumps({"name": name, "arguments": args}), "tool_call_id": call_id}


def test_request_function_parts_become_tool_entries():
    """Function call and response parts become tool-call and tool-result entries."""
    request = {
        "model": "gemini-2.0-flash",
        "config": {"system_instruction": "Be brief"},
        "contents": [
            {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
            {
                "role": "model",
                "parts": [{"function_call": {"id": "c1", "name": "get_weather", "args": {"city": "Paris"}}}],
            },
            {
                "role": "user",
                "parts": [{"function_response": {"id": "c1", "name": "get_weather", "response": {"temp": 20}}}],
            },
            {"role": "model", "parts": [{"text": "It is 20C."}]},
        ],
    }

    assert json.loads(extract_llm_request_attributes(request)["input"]) == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Weather in Paris?"},
        _call("c1", "get_weather", {"city": "Paris"}),
        {"role": "tool", "content": json.dumps({"temp": 20}), "tool_call_id": "c1"},
        {"role": "assistant", "content": "It is 20C."},
    ]


def test_response_text_and_function_calls_become_separate_entries():
    """Response text becomes one assistant entry followed by one entry per function call."""
    parts = [
        SimpleNamespace(text="Checking both.", function_call=None),
        SimpleNamespace(text=None, function_call=SimpleNamespace(id="c1", name="get_weather", args={"city": "Paris"})),
        SimpleNamespace(text=None, function_call=SimpleNamespace(id="c2", name="get_weather", args={"city": "Rome"})),
    ]
    event = SimpleNamespace(content=SimpleNamespace(parts=parts))

    assert json.loads(extract_llm_response_attributes(event, [])["output"]) == [
        {"role": "assistant", "content": "Checking both."},
        _call("c1", "get_weather", {"city": "Paris"}),
        _call("c2", "get_weather", {"city": "Rome"}),
    ]
