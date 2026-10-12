"""Tests for SpanIOProcessor folding legacy indexed gen_ai message keys into input/output."""

import json

from opentelemetry.sdk.trace import TracerProvider

from netra.processors.instrumentation_span_processor import InstrumentationSpanProcessor
from netra.processors.span_io_processor import SpanIOProcessor


def _start_span():
    # Same processor order as the Tracer setup: InstrumentationSpanProcessor first.
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    provider.add_span_processor(SpanIOProcessor())
    return provider.get_tracer("test").start_span("third_party.chat")


def test_indexed_keys_are_folded_into_input_and_output_and_not_stored():
    span = _start_span()
    span.set_attribute("gen_ai.prompt.0.role", "user")
    span.set_attribute("gen_ai.prompt.0.content", "hi")
    span.set_attributes({"gen_ai.completion.0.role": "assistant", "gen_ai.completion.0.content": "hello"})
    span.set_attribute("gen_ai.completion.0.finish_reason", "stop")

    attrs = dict(span.attributes or {})
    assert json.loads(attrs["input"]) == [{"role": "user", "content": "hi"}]
    assert json.loads(attrs["output"]) == [{"role": "assistant", "content": "hello", "finish_reason": "stop"}]
    assert not any(key.startswith(("gen_ai.prompt", "gen_ai.completion")) for key in attrs)
    span.end()


def test_plural_indexed_keys_are_folded_in_index_order():
    span = _start_span()
    span.set_attribute("gen_ai.prompts.1.content", "second")
    span.set_attribute("gen_ai.prompts.0.content", "first")

    assert [entry["content"] for entry in json.loads(span.attributes["input"])] == ["first", "second"]
    span.end()


def test_indexed_keys_do_not_overwrite_input_already_written():
    span = _start_span()
    span.set_attribute("input", "already")
    span.set_attribute("gen_ai.prompt.0.content", "hi")

    attrs = dict(span.attributes or {})
    assert attrs["input"] == "already"
    assert "gen_ai.prompt.0.content" not in attrs
    span.end()


def test_non_message_gen_ai_keys_pass_through():
    span = _start_span()
    span.set_attribute("gen_ai.request.model", "gpt-4")

    assert span.attributes["gen_ai.request.model"] == "gpt-4"
    span.end()
