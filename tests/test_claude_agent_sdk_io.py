"""Tests for the structured ``input``/``output`` built by the Claude Agent SDK instrumentation."""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

MODULE = "netra.instrumentation.libraries.claude_agent_sdk"
SDK_TYPES = (
    "AssistantMessage",
    "ClaudeAgentOptions",
    "ResultMessage",
    "SystemMessage",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "UserMessage",
)


@pytest.fixture
def utils():
    """Import the instrumentation utils with the ``claude_agent_sdk`` types stubbed out."""
    sdk = ModuleType("claude_agent_sdk")
    for name in SDK_TYPES:
        setattr(sdk, name, type(name, (SimpleNamespace,), {}))
    with patch.dict(sys.modules, {"claude_agent_sdk": sdk}):
        for name in [name for name in sys.modules if name.startswith(MODULE)]:
            del sys.modules[name]
        yield importlib.import_module(f"{MODULE}.utils")


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def _result(**fields):
    defaults = dict(
        structured_output=None,
        result=None,
        num_turns=None,
        session_id=None,
        stop_reason=None,
        is_error=False,
        permission_denials=None,
        errors=None,
    )
    return SimpleNamespace(**{**defaults, **fields})


def test_system_prompt_and_prompt_become_input_entries(utils):
    """The options' system prompt and the user prompt become system and user entries."""
    span = _span()
    options = utils.ClaudeAgentOptions(
        model="claude-sonnet",
        system_prompt="Be brief",
        permission_mode=None,
        max_turns=None,
        cwd=None,
        session_id=None,
        continue_conversation=None,
    )

    utils.set_request_attributes(span, "Hi", options)

    assert json.loads(span.attributes["input"]) == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Hi"},
    ]


def test_result_text_becomes_assistant_entry(utils):
    """The final result text becomes a single assistant entry."""
    span = _span()

    utils.set_result_message_attributes(Mock(), span, _result(result="Done"))

    assert json.loads(span.attributes["output"]) == [{"role": "assistant", "content": "Done"}]


def test_structured_output_takes_precedence_over_result(utils):
    """Structured output is JSON-encoded and used instead of the result text."""
    span = _span()

    utils.set_result_message_attributes(Mock(), span, _result(result="Done", structured_output={"answer": 42}))

    assert json.loads(span.attributes["output"]) == [{"role": "assistant", "content": json.dumps({"answer": 42})}]
