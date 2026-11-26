"""Tests for SpanIOProcessor folding legacy indexed gen_ai message keys into input/output."""

import json

from opentelemetry.sdk.trace import TracerProvider

from netra.processors import span_io_processor
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
    span.end()

    attrs = dict(span.attributes or {})
    assert json.loads(attrs["input"]) == [{"role": "user", "content": "hi"}]
    assert json.loads(attrs["output"]) == [{"role": "assistant", "content": "hello", "finish_reason": "stop"}]
    assert not any(key.startswith(("gen_ai.prompt", "gen_ai.completion")) for key in attrs)


def test_messages_are_built_once_when_the_span_ends(monkeypatch):
    calls = []
    build = span_io_processor._build_messages
    monkeypatch.setattr(span_io_processor, "_build_messages", lambda index_map: calls.append(1) or build(index_map))
    span = _start_span()
    for index in range(20):
        span.set_attributes({f"gen_ai.prompt.{index}.role": "user", f"gen_ai.prompt.{index}.content": f"m{index}"})

    assert span.attributes["input"] == ""
    span.end()

    assert len(calls) == 1
    assert len(json.loads(span.attributes["input"])) == 20


def test_plural_indexed_keys_are_folded_in_index_order():
    span = _start_span()
    span.set_attribute("gen_ai.prompts.1.content", "second")
    span.set_attribute("gen_ai.prompts.0.content", "first")
    span.end()

    assert [entry["content"] for entry in json.loads(span.attributes["input"])] == ["first", "second"]


def test_indexed_keys_do_not_overwrite_input_already_written():
    span = _start_span()
    span.set_attribute("input", "already")
    span.set_attribute("gen_ai.prompt.0.content", "hi")
    span.end()

    attrs = dict(span.attributes or {})
    assert attrs["input"] == "already"
    assert "gen_ai.prompt.0.content" not in attrs


def test_input_written_after_indexed_keys_wins():
    span = _start_span()
    span.set_attribute("gen_ai.prompt.0.content", "hi")
    span.set_attribute("input", "explicit")
    span.end()

    assert span.attributes["input"] == "explicit"


def test_non_message_gen_ai_keys_pass_through():
    span = _start_span()
    span.set_attribute("gen_ai.request.model", "gpt-4")

    assert span.attributes["gen_ai.request.model"] == "gpt-4"
    span.end()


def test_messages_are_ordered_by_index_and_missing_indexes_skipped():
    span = _start_span()
    span.set_attribute("gen_ai.prompt.10.content", "last")
    span.set_attribute("gen_ai.prompt.0.content", "first")
    span.set_attribute("gen_ai.prompt.2.content", "middle")
    span.end()

    assert json.loads(span.attributes["input"]) == [{"content": "first"}, {"content": "middle"}, {"content": "last"}]


def test_indexed_keys_passed_at_span_creation_are_folded():
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    provider.add_span_processor(SpanIOProcessor())
    span = provider.get_tracer("test").start_span(
        "third_party.chat",
        attributes={"gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "hi", "gen_ai.request.model": "gpt-4"},
    )
    span.end()

    attrs = dict(span.attributes or {})
    assert json.loads(attrs["input"]) == [{"role": "user", "content": "hi"}]
    assert attrs["gen_ai.request.model"] == "gpt-4"
    assert not any(key.startswith("gen_ai.prompt") for key in attrs)


def test_folded_tool_call_fields_become_tool_call_entries():
    span = _start_span()
    span.set_attributes(
        {
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.tool_calls.0.id": "call_1",
            "gen_ai.completion.0.tool_calls.0.name": "lookup",
            "gen_ai.completion.0.tool_calls.0.arguments": '{"id": 42}',
        }
    )
    span.end()

    assert json.loads(span.attributes["output"]) == [
        {
            "role": "assistant",
            "content": json.dumps({"name": "lookup", "arguments": '{"id": 42}'}),
            "tool_call_id": "call_1",
        }
    ]
