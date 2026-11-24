"""Tests for the structured ``input``/``output`` built by the ElevenLabs instrumentation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from netra.instrumentation.libraries.elevenlabs.utils import set_request_attributes, set_response_attributes


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def test_text_becomes_input_entry():
    """Text to convert becomes a single input entry."""
    span = _span()

    set_request_attributes(span, {"text": "Hello there", "voice_id": "v1"})

    assert json.loads(span.attributes["input"]) == [{"role": "Input", "content": "Hello there"}]


def test_prompt_becomes_input_entry():
    """A generation prompt becomes a single input entry."""
    span = _span()

    set_request_attributes(span, {"prompt": "Calm piano"})

    assert json.loads(span.attributes["input"]) == [{"role": "Input", "content": "Calm piano"}]


def test_transcribed_text_becomes_output_entry():
    """A response's text becomes a single output entry."""
    span = _span()

    set_response_attributes(span, SimpleNamespace(text="Hello there"))

    assert json.loads(span.attributes["output"]) == [{"role": "Output", "content": "Hello there"}]
