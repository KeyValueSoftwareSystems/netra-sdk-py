"""Tests for the structured ``input``/``output`` built by the DSPy instrumentation."""

import json
from types import SimpleNamespace

from netra.instrumentation.libraries.dspy.utils import extract_llm_input_messages, extract_llm_output_messages


def test_messages_become_input_entries():
    """LM call messages become entries with their roles."""
    messages = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]

    assert json.loads(dict(extract_llm_input_messages({"messages": messages}))["input"]) == messages


def test_prompt_becomes_user_entry():
    """A string prompt becomes a single user entry."""
    assert json.loads(dict(extract_llm_input_messages({"prompt": "Hi"}))["input"]) == [
        {"role": "user", "content": "Hi"}
    ]


def test_string_completions_become_assistant_entries():
    """Each string completion becomes an assistant entry."""
    output = dict(extract_llm_output_messages(["Hello", "Hi"]))["output"]

    assert json.loads(output) == [{"role": "assistant", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]


def test_response_choices_become_assistant_entries():
    """A response object's choices become assistant entries carrying the finish reason."""
    response_dict = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}]}
    response = SimpleNamespace(choices=[], model_dump=lambda: response_dict)

    assert json.loads(dict(extract_llm_output_messages(response))["output"]) == [
        {"role": "assistant", "content": "Hello", "finish_reason": "stop"}
    ]
