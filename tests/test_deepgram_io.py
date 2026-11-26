"""Tests for the structured ``input``/``output`` built by the Deepgram instrumentation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from netra.instrumentation.libraries.deepgram.utils import set_request_attributes, set_response_attributes


def _span():
    span = Mock()
    span.attributes = {}
    span.set_attribute.side_effect = span.attributes.__setitem__
    return span


def _alternatives(transcript):
    return SimpleNamespace(alternatives=[SimpleNamespace(transcript=transcript)])


def test_speak_text_becomes_input_entry():
    """Text to speak becomes a single input entry."""
    span = _span()

    set_request_attributes(span, {"text": "Hello there"})

    assert json.loads(span.attributes["input"]) == [{"role": "Input", "content": "Hello there"}]


def test_http_transcript_becomes_output_entry():
    """The first transcript of an HTTP response becomes the output entry."""
    span = _span()
    response = SimpleNamespace(results=SimpleNamespace(channels=[_alternatives("Hello there")]))

    set_response_attributes(span, response)

    assert json.loads(span.attributes["output"]) == [{"role": "Transcribed Text", "content": "Hello there"}]


def test_websocket_results_transcript_becomes_output_entry():
    """The transcript of a WebSocket ``Results`` event becomes the output entry."""
    span = _span()

    set_response_attributes(span, SimpleNamespace(type="Results", channel=_alternatives("Hi")))

    assert json.loads(span.attributes["output"]) == [{"role": "Transcribed Text", "content": "Hi"}]
