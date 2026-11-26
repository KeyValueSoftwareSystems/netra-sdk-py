"""Tests for the structured ``input``/``output`` built by the pydantic_ai and pydantic_ai_slim instrumentations."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class UserPromptNode(SimpleNamespace):
    pass


class ModelRequestNode(SimpleNamespace):
    pass


class CallToolsNode(SimpleNamespace):
    pass


@pytest.fixture(params=["pydantic_ai", "pydantic_ai_slim"])
def utils(request):
    """Return the utils module of each pydantic_ai instrumentation variant."""
    return importlib.import_module(f"netra.instrumentation.libraries.{request.param}.utils")


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def test_user_prompt_node_becomes_user_entry(utils):
    """The user prompt becomes a single user entry."""
    span = _span()

    utils.set_node_attributes(span, UserPromptNode(user_prompt="Weather in Paris?"))

    assert json.loads(span.attributes["input"]) == [{"role": "user", "content": "Weather in Paris?"}]


def test_model_request_parts_map_part_kinds_to_roles(utils):
    """Request parts map their part kind to a role, and tool returns keep their call id."""
    span = _span()
    parts = [
        SimpleNamespace(part_kind="system-prompt", content="Be brief"),
        SimpleNamespace(part_kind="user-prompt", content="Weather in Paris?"),
        SimpleNamespace(part_kind="tool-return", content="20C", tool_name="get_weather", tool_call_id="c1"),
    ]

    utils.set_node_attributes(span, ModelRequestNode(request=SimpleNamespace(parts=parts)))

    assert json.loads(span.attributes["input"]) == [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "tool", "content": "20C", "tool_call_id": "c1"},
    ]


def test_call_tools_text_part_becomes_assistant_entry(utils):
    """A text part of the model response becomes an assistant entry."""
    span = _span()
    response = SimpleNamespace(parts=[SimpleNamespace(part_kind="text", content="Sunny")])

    utils.set_node_attributes(span, CallToolsNode(model_response=response))

    assert json.loads(span.attributes["output"]) == [{"role": "assistant", "content": "Sunny"}]


def test_run_result_output_becomes_assistant_entry(utils):
    """A run result's output becomes a single assistant entry."""
    span = _span()

    utils.set_pydantic_response_attributes(span, SimpleNamespace(output="Done"))

    assert json.loads(span.attributes["output"]) == [{"role": "assistant", "content": "Done"}]
