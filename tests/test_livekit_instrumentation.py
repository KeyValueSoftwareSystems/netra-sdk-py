"""Tests for Netra's LiveKit voice-agent instrumentation.

The suite exercises the package through the real OpenTelemetry SDK rather than
mocks: spans are created from a tracer whose instrumentation scope is
``livekit-agents`` — the only thing the processor gates on — so no
``livekit-agents`` install is required.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from wrapt import ObjectProxy

from netra.exporters.filtering_span_exporter import FilteringSpanExporter
from netra.instrumentation import livekit as livekit_instrumentation
from netra.instrumentation.instruments import DEFAULT_INSTRUMENTS_FOR_ROOT
from netra.instrumentation.livekit import NetraLiveKitInstrumentor
from netra.instrumentation.livekit.provider_binding import _ShieldedTracerProvider
from netra.instrumentation.livekit.session_root_processor import SessionRootSpanProcessor
from netra.instrumentation.livekit.trace_processor import SpanMappingProcessor
from netra.instrumentation.livekit.utils import (
    LIVEKIT_SCOPE_NAME,
    MAX_CONVERSATION_MESSAGES_PER_SIDE,
    NETRA_CONVERSATION_TRUNCATED,
    NETRA_SPAN_TYPE,
    ConversationSide,
    content_of_choice_event,
    content_of_event,
    conversation_from_attributes,
    is_zero_usage,
    messages_for_parent,
    messages_from_chat_ctx,
    netra_span_type_for,
    role_of_choice_event,
    tts_pricing_attributes_from,
)
from netra.processors.llm_trace_identifier_span_processor import LlmTraceIdentifierSpanProcessor
from netra.processors.root_instrument_filter_processor import RootInstrumentFilterProcessor
from netra.processors.root_span_processor import RootSpanProcessor
from netra.span_wrapper import SpanType

pytestmark = pytest.mark.unit

# The shipped root allow-list, as instrument-name strings. ``livekit`` is in it,
# which is what keeps LiveKit's own spans out of the candidate registry.
DEFAULT_ROOT_INSTRUMENT_NAMES = {instrument.value for instrument in DEFAULT_INSTRUMENTS_FOR_ROOT}


class _Harness:
    """A tracer provider carrying the LiveKit processor and an in-memory exporter."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        # Registration order mirrors production: the exporting processor is
        # installed by netra/tracer.py first, the LiveKit one by _instrument().
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.provider.add_span_processor(SpanMappingProcessor())
        self.livekit_tracer = self.provider.get_tracer(LIVEKIT_SCOPE_NAME)

    def tracer(self, scope_name: str) -> Any:
        """Return a tracer for some other instrumentation scope."""
        return self.provider.get_tracer(scope_name)

    def finished(self, name: str) -> ReadableSpan:
        """Return the single finished span called *name*."""
        matches = [span for span in self.exporter.get_finished_spans() if span.name == name]
        assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
        return matches[0]

    def attributes(self, name: str) -> Dict[str, Any]:
        """Return the exported attributes of the finished span called *name*."""
        return dict(self.finished(name).attributes or {})


@pytest.fixture
def harness() -> _Harness:
    return _Harness()


def _record(harness: _Harness, span_name: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Run one LiveKit span through the processor and return its exported attributes."""
    span = harness.livekit_tracer.start_span(span_name)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    span.end()
    return harness.attributes(span_name)


def _messages(attributes: Dict[str, Any], side: str) -> List[Tuple[str, str]]:
    """Read the indexed ``gen_ai.<side>.N.role/content`` pairs back, in index order."""
    indices = sorted(
        int(key.split(".")[2]) for key in attributes if key.startswith(f"gen_ai.{side}.") and key.endswith(".content")
    )
    return [
        (attributes[f"gen_ai.{side}.{index}.role"], attributes[f"gen_ai.{side}.{index}.content"]) for index in indices
    ]


class TestSpanTypeMapping:
    @pytest.mark.parametrize(
        "span_name,expected",
        [
            ("agent_turn", SpanType.AGENT.value),
            ("llm_node", SpanType.GENERATION.value),
            ("llm_request", SpanType.GENERATION.value),
            ("function_tool", SpanType.TOOL.value),
            ("tts_request", SpanType.GENERATION.value),
            ("agent_session", SpanType.SPAN.value),
            ("something_unmapped", SpanType.SPAN.value),
            (None, SpanType.SPAN.value),
            ("", SpanType.SPAN.value),
        ],
    )
    def test_returns_span_type_for_name(self, span_name: Optional[str], expected: str) -> None:
        assert netra_span_type_for(span_name) == expected

    def test_span_type_is_stamped_on_livekit_spans(self, harness: _Harness) -> None:
        assert _record(harness, "agent_turn", {})["netra.span.type"] == SpanType.AGENT.value

    def test_job_entrypoint_is_marked_as_a_workflow(self, harness: _Harness) -> None:
        assert _record(harness, "job_entrypoint", {})["netra.entity.type"] == "workflow"

    def test_non_entity_spans_carry_no_entity_marker(self, harness: _Harness) -> None:
        assert "netra.entity.type" not in _record(harness, "agent_turn", {})

    @pytest.mark.parametrize(
        "span_name,expected",
        [("agent_session", "session"), ("agent_turn", "span"), ("user_turn", "span")],
    )
    def test_audio_type_is_stamped_on_interaction_spans(self, harness: _Harness, span_name: str, expected: str) -> None:
        assert _record(harness, span_name, {})["netra.audio.type"] == expected

    def test_audio_type_is_absent_from_other_spans(self, harness: _Harness) -> None:
        assert "netra.audio.type" not in _record(harness, "llm_node", {})

    def test_spans_from_other_scopes_are_left_untouched(self, harness: _Harness) -> None:
        span = harness.tracer("openai").start_span("agent_turn")
        span.set_attribute("lk.function_tool.name", "lookup")
        span.end()

        attributes = harness.attributes("agent_turn")
        assert "netra.span.type" not in attributes
        assert "netra.tool.name" not in attributes
        assert attributes["lk.function_tool.name"] == "lookup"


class TestAttributeMirroring:
    @pytest.mark.parametrize(
        "lk_key,netra_key",
        [
            ("lk.function_tool.name", "netra.tool.name"),
            ("lk.function_tool.arguments", "input"),
            ("lk.function_tool.output", "output"),
            ("lk.response.ttft", "netra.latency.ttft"),
            ("lk.response.ttfb", "netra.latency.ttfb"),
            ("lk.e2e_latency", "netra.latency.e2e"),
            ("lk.end_of_turn_delay", "netra.latency.end_of_turn_delay"),
            ("lk.transcript_confidence", "netra.stt.confidence"),
            ("lk.interrupted", "netra.turn.interrupted"),
        ],
    )
    def test_mapped_attribute_is_mirrored_and_original_preserved(
        self, harness: _Harness, lk_key: str, netra_key: str
    ) -> None:
        attributes = _record(harness, "function_tool", {lk_key: 0.25})

        assert attributes[netra_key] == 0.25
        assert attributes[lk_key] == 0.25, "the original lk.* attribute must be preserved"

    def test_unmapped_attribute_is_written_through_unchanged(self, harness: _Harness) -> None:
        attributes = _record(harness, "agent_turn", {"lk.speech_id": "sp_1"})

        assert attributes["lk.speech_id"] == "sp_1"

    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_value_is_not_mirrored(self, harness: _Harness, empty: Any) -> None:
        span = harness.livekit_tracer.start_span("function_tool")
        span.set_attribute("lk.function_tool.name", empty)
        span.end()

        assert "netra.tool.name" not in harness.attributes("function_tool")

    def test_set_attributes_plural_goes_through_the_mapping(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("user_turn")
        span.set_attributes({"lk.user_transcript": "book me a table", "lk.e2e_latency": 1.5})
        span.end()

        attributes = harness.attributes("user_turn")
        assert attributes["netra.latency.e2e"] == 1.5
        assert _messages(attributes, "completion") == [("user", "book me a table")]

    def test_empty_set_attributes_is_a_no_op(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("agent_turn")
        span.set_attributes({})
        span.end()

        assert "gen_ai.prompt.0.content" not in harness.attributes("agent_turn")


class TestUsageAttributes:
    def test_reported_usage_is_marked_as_framework_sourced(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"gen_ai.usage.input_tokens": 120})

        assert attributes["gen_ai.usage.input_tokens"] == 120
        assert attributes["netra.usage.source"] == "framework"

    def test_zero_usage_is_dropped_rather_than_claimed(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"gen_ai.usage.input_tokens": 0})

        assert "gen_ai.usage.input_tokens" not in attributes
        assert "netra.usage.source" not in attributes

    @pytest.mark.parametrize(
        "value,expected",
        [(0, True), (0.0, True), (1, False), (-1, False), (True, False), (False, False), (None, False), ("0", False)],
    )
    def test_is_zero_usage_only_matches_numeric_zero(self, value: Any, expected: bool) -> None:
        assert is_zero_usage(value) is expected


class TestConversationMapping:
    def test_agent_turn_reads_as_system_then_user_then_reply(self, harness: _Harness) -> None:
        attributes = _record(
            harness,
            "agent_turn",
            {
                "lk.instructions": "You are a helpful agent.",
                "lk.user_input": "what is the weather?",
                "lk.response.text": "It is sunny.",
            },
        )

        assert _messages(attributes, "prompt") == [
            ("system", "You are a helpful agent."),
            ("user", "what is the weather?"),
        ]
        assert _messages(attributes, "completion") == [("assistant", "It is sunny.")]

    def test_tts_input_text_is_attributed_to_the_assistant(self, harness: _Harness) -> None:
        attributes = _record(harness, "tts_request", {"lk.input_text": "It is sunny."})

        assert _messages(attributes, "prompt") == [("assistant", "It is sunny.")]

    def test_user_transcript_is_the_callers_words(self, harness: _Harness) -> None:
        attributes = _record(harness, "user_turn", {"lk.user_transcript": "hello there"})

        assert _messages(attributes, "completion") == [("user", "hello there")]

    def test_chat_ctx_is_expanded_into_indexed_prompts(self, harness: _Harness) -> None:
        chat_ctx = json.dumps(
            {
                "items": [
                    {"type": "message", "role": "system", "content": ["Be brief."]},
                    {"type": "agent_config_update", "role": "system", "content": ["ignored"]},
                    {"type": "message", "role": "user", "content": ["hi"]},
                ]
            }
        )
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": chat_ctx})

        assert _messages(attributes, "prompt") == [("system", "Be brief."), ("user", "hi")]
        assert attributes["lk.chat_ctx"] == chat_ctx, "the original blob must be preserved"

    def test_sources_on_one_span_share_the_index_counters(self, harness: _Harness) -> None:
        chat_ctx = json.dumps({"items": [{"type": "message", "role": "system", "content": ["Be brief."]}]})
        attributes = _record(
            harness,
            "agent_turn",
            {"lk.chat_ctx": chat_ctx, "lk.user_input": "hi", "lk.response.text": "hello"},
        )

        assert _messages(attributes, "prompt") == [("system", "Be brief."), ("user", "hi")]
        assert _messages(attributes, "completion") == [("assistant", "hello")]

    def test_conversation_content_does_not_write_input_directly(self, harness: _Harness) -> None:
        attributes = _record(harness, "agent_turn", {"lk.user_input": "hi"})

        assert "input" not in attributes, "input is assembled downstream by SpanIOProcessor"


# A conversation long enough that expanding all of it would overflow the OTel
# default attribute budget (2 attributes per message vs OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT
# = 128), so the tests below fail on an unbounded recorder rather than merely
# asserting the cap's arithmetic.
_OVERFLOWING_MESSAGE_COUNT = MAX_CONVERSATION_MESSAGES_PER_SIDE * 4
_DEFAULT_SPAN_ATTRIBUTE_LIMIT = 128


def _chat_ctx(num_messages: int) -> str:
    """A serialised ChatContext of *num_messages* alternating turns."""
    items = [
        {
            "type": "message",
            "role": "user" if index % 2 == 0 else "assistant",
            "content": [f"turn {index}"],
        }
        for index in range(num_messages)
    ]
    return json.dumps({"items": items})


class TestConversationIsBounded:
    """The conversation must not grow past the span's bounded attribute capacity.

    OTel's ``BoundedAttributes`` evicts the *oldest* attribute on overflow, so an
    unbounded sequence deletes the markers stamped in ``on_start`` rather than
    dropping its own tail. See ``MAX_CONVERSATION_MESSAGES_PER_SIDE``.
    """

    def test_long_chat_ctx_does_not_evict_the_netra_markers(self, harness: _Harness) -> None:
        # The regression: the unbounded expansion pushed the span past the
        # 128-attribute default, and the markers — written first, in on_start —
        # were the first things evicted.
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": _chat_ctx(_OVERFLOWING_MESSAGE_COUNT)})

        assert attributes[NETRA_SPAN_TYPE] == SpanType.GENERATION.value
        assert attributes["lk.chat_ctx"], "the original blob must still be preserved"
        assert len(attributes) < _DEFAULT_SPAN_ATTRIBUTE_LIMIT

    def test_long_chat_ctx_keeps_the_newest_turns(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": _chat_ctx(_OVERFLOWING_MESSAGE_COUNT)})
        prompts = _messages(attributes, "prompt")

        assert len(prompts) == MAX_CONVERSATION_MESSAGES_PER_SIDE
        # The tail of the conversation survives; its opening is dropped.
        first_kept = _OVERFLOWING_MESSAGE_COUNT - MAX_CONVERSATION_MESSAGES_PER_SIDE
        assert prompts[0][1] == f"turn {first_kept}"
        assert prompts[-1][1] == f"turn {_OVERFLOWING_MESSAGE_COUNT - 1}"

    def test_truncation_is_marked_on_the_span(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": _chat_ctx(_OVERFLOWING_MESSAGE_COUNT)})

        assert attributes[NETRA_CONVERSATION_TRUNCATED] is True

    def test_a_short_conversation_is_not_marked_truncated(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": _chat_ctx(4)})

        assert len(_messages(attributes, "prompt")) == 4
        assert NETRA_CONVERSATION_TRUNCATED not in attributes

    def test_a_conversation_exactly_at_the_cap_is_not_marked(self, harness: _Harness) -> None:
        attributes = _record(harness, "llm_node", {"lk.chat_ctx": _chat_ctx(MAX_CONVERSATION_MESSAGES_PER_SIDE)})

        assert len(_messages(attributes, "prompt")) == MAX_CONVERSATION_MESSAGES_PER_SIDE
        assert NETRA_CONVERSATION_TRUNCATED not in attributes

    def test_unbounded_events_are_capped_too(self, harness: _Harness) -> None:
        # livekit-agents emits one conversation event per context item on
        # llm_request (llm/llm.py -> _chat_ctx_to_otel_events), so the event path
        # grows with the call exactly like lk.chat_ctx does.
        span = harness.livekit_tracer.start_span("llm_request")
        for index in range(_OVERFLOWING_MESSAGE_COUNT):
            span.add_event("gen_ai.user.message", {"content": f"turn {index}"})
        span.end()
        attributes = harness.attributes("llm_request")

        assert len(_messages(attributes, "prompt")) == MAX_CONVERSATION_MESSAGES_PER_SIDE
        assert attributes[NETRA_CONVERSATION_TRUNCATED] is True
        assert attributes[NETRA_SPAN_TYPE] == SpanType.GENERATION.value

    def test_the_reply_survives_a_saturated_prompt_side(self, harness: _Harness) -> None:
        # The two sides count independently, so a long context can never crowd
        # out the completion — the single most important value on the span.
        attributes = _record(
            harness,
            "llm_node",
            {"lk.chat_ctx": _chat_ctx(_OVERFLOWING_MESSAGE_COUNT), "lk.response.text": "THE REPLY"},
        )

        assert _messages(attributes, "completion") == [("assistant", "THE REPLY")]

    def test_propagated_child_content_is_capped(self, harness: _Harness) -> None:
        # A provider span's own indexed prompts also grow with the conversation,
        # and llm_request_run inherits them wholesale.
        parent = harness.livekit_tracer.start_span("llm_request_run")
        child_attributes: Dict[str, Any] = {}
        for index in range(_OVERFLOWING_MESSAGE_COUNT):
            child_attributes[f"gen_ai.prompt.{index}.role"] = "user"
            child_attributes[f"gen_ai.prompt.{index}.content"] = f"turn {index}"
        with trace.use_span(parent, end_on_exit=False):
            child = harness.tracer("openai").start_span("openai.chat", attributes=child_attributes)
            child.end()
        parent.end()
        attributes = harness.attributes("llm_request_run")

        assert len(_messages(attributes, "prompt")) == MAX_CONVERSATION_MESSAGES_PER_SIDE
        assert attributes[NETRA_CONVERSATION_TRUNCATED] is True


class TestConversationEvents:
    @pytest.mark.parametrize(
        "event_name,role",
        [
            ("gen_ai.system.message", "system"),
            ("gen_ai.user.message", "user"),
            ("gen_ai.assistant.message", "assistant"),
            ("gen_ai.tool.message", "tool"),
        ],
    )
    def test_conversation_event_becomes_an_indexed_prompt(self, harness: _Harness, event_name: str, role: str) -> None:
        span = harness.livekit_tracer.start_span("llm_request")
        span.add_event(event_name, {"content": "some text"})
        span.end()

        assert _messages(harness.attributes("llm_request"), "prompt") == [(role, "some text")]

    def test_choice_event_becomes_an_indexed_completion(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("llm_request")
        span.add_event("gen_ai.choice", {"role": "assistant", "content": "the reply"})
        span.end()

        assert _messages(harness.attributes("llm_request"), "completion") == [("assistant", "the reply")]

    def test_event_is_still_recorded_on_the_span(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("llm_request")
        span.add_event("gen_ai.user.message", {"content": "hi"})
        span.end()

        events = harness.finished("llm_request").events
        assert [event.name for event in events] == ["gen_ai.user.message"]

    def test_unrelated_event_is_recorded_but_not_mapped(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("llm_request")
        span.add_event("some.other.event", {"content": "hi"})
        span.end()

        attributes = harness.attributes("llm_request")
        assert "gen_ai.prompt.0.content" not in attributes
        assert [event.name for event in harness.finished("llm_request").events] == ["some.other.event"]

    def test_event_without_content_contributes_no_message(self, harness: _Harness) -> None:
        span = harness.livekit_tracer.start_span("llm_request")
        span.add_event("gen_ai.user.message", {"role": "user"})
        span.end()

        assert "gen_ai.prompt.0.content" not in harness.attributes("llm_request")


class TestTtsPricing:
    def test_priceable_fields_are_lifted_out_of_the_metrics_blob(self, harness: _Harness) -> None:
        metrics = json.dumps({"characters_count": 42, "metadata": {"model_name": "cartesia/sonic-3"}})
        attributes = _record(harness, "tts_request", {"lk.tts_metrics": metrics})

        assert attributes["gen_ai.request.model"] == "cartesia/sonic-3"
        assert attributes["gen_ai.usage.prompt.character_count"] == 42
        assert attributes["lk.tts_metrics"] == metrics, "the original blob must be preserved"

    def test_lifted_character_count_is_marked_framework_sourced(self, harness: _Harness) -> None:
        metrics = json.dumps({"characters_count": 42, "metadata": {"model_name": "cartesia/sonic-3"}})
        attributes = _record(harness, "tts_request", {"lk.tts_metrics": metrics})

        assert attributes["netra.usage.source"] == "framework"

    @pytest.mark.parametrize(
        "payload,expected_model,expected_count",
        [
            ({"characters_count": 7, "metadata": {"model_name": "sonic"}}, "sonic", 7),
            ({"characters_count": 7.9, "metadata": {"model_name": "sonic"}}, "sonic", 7),
            ({"characters_count": 0, "metadata": {"model_name": "sonic"}}, "sonic", None),
            ({"characters_count": -3, "metadata": {"model_name": "sonic"}}, "sonic", None),
            ({"characters_count": True, "metadata": {"model_name": "sonic"}}, "sonic", None),
            ({"characters_count": 7}, None, 7),
            ({"characters_count": 7, "metadata": {"model_name": ""}}, None, 7),
            ({"metadata": {"model_name": "sonic"}}, "sonic", None),
            ("not json", None, None),
            (None, None, None),
            ([], None, None),
        ],
    )
    def test_extraction_tolerates_every_shape(
        self, payload: Any, expected_model: Optional[str], expected_count: Optional[int]
    ) -> None:
        pricing = tts_pricing_attributes_from(payload)

        assert pricing.model == expected_model
        assert pricing.character_count == expected_count

    def test_accepts_a_json_string_and_a_dict_identically(self) -> None:
        payload = {"characters_count": 9, "metadata": {"model_name": "sonic"}}

        assert tts_pricing_attributes_from(payload) == tts_pricing_attributes_from(json.dumps(payload))


class TestChildToParentPropagation:
    def _child_under(self, harness: _Harness, parent_name: str, scope: str, attributes: Dict[str, Any]) -> None:
        parent = harness.livekit_tracer.start_span(parent_name)
        with trace.use_span(parent, end_on_exit=False):
            child = harness.tracer(scope).start_span("provider.call")
            for key, value in attributes.items():
                child.set_attribute(key, value)
            child.end()
        parent.end()

    def test_provider_span_content_is_lifted_onto_llm_request_run(self, harness: _Harness) -> None:
        self._child_under(
            harness,
            "llm_request_run",
            "openai",
            {
                "gen_ai.prompt.0.role": "user",
                "gen_ai.prompt.0.content": "hi",
                "gen_ai.completion.0.role": "assistant",
                "gen_ai.completion.0.content": "hello",
            },
        )

        attributes = harness.attributes("llm_request_run")
        assert _messages(attributes, "prompt") == [("user", "hi")]
        assert _messages(attributes, "completion") == [("assistant", "hello")]

    def test_non_llm_child_input_is_not_copied_up_as_a_message(self, harness: _Harness) -> None:
        self._child_under(harness, "llm_request_run", "httpx", {"input": "POST https://api.example.com/v1/chat"})

        attributes = harness.attributes("llm_request_run")
        assert "gen_ai.prompt.0.content" not in attributes

    def test_llm_child_raw_io_is_copied_when_it_has_no_indexed_pairs(self, harness: _Harness) -> None:
        self._child_under(
            harness,
            "llm_request_run",
            "openai",
            {"gen_ai.request.model": "gpt-4o", "input": "hi", "output": "hello"},
        )

        attributes = harness.attributes("llm_request_run")
        assert _messages(attributes, "prompt") == [("user", "hi")]
        assert _messages(attributes, "completion") == [("assistant", "hello")]

    def test_tts_node_inherits_from_its_tts_request_child(self, harness: _Harness) -> None:
        parent = harness.livekit_tracer.start_span("tts_node")
        with trace.use_span(parent, end_on_exit=False):
            child = harness.livekit_tracer.start_span("tts_request")
            child.set_attribute("lk.input_text", "It is sunny.")
            child.end()
        parent.end()

        assert _messages(harness.attributes("tts_node"), "prompt") == [("assistant", "It is sunny.")]

    def test_spans_not_awaiting_child_content_are_unaffected(self, harness: _Harness) -> None:
        self._child_under(
            harness,
            "agent_turn",
            "openai",
            {"gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "hi"},
        )

        assert "gen_ai.prompt.0.content" not in harness.attributes("agent_turn")

    def test_content_arriving_after_the_parent_ended_is_dropped(self, harness: _Harness) -> None:
        parent = harness.livekit_tracer.start_span("llm_request_run")
        with trace.use_span(parent, end_on_exit=False):
            child = harness.tracer("openai").start_span("provider.call")
            child.set_attribute("gen_ai.prompt.0.role", "user")
            child.set_attribute("gen_ai.prompt.0.content", "hi")
        parent.end()
        child.end()

        assert "gen_ai.prompt.0.content" not in harness.attributes("llm_request_run")


class TestConversationReading:
    def test_indexed_pairs_are_ordered_numerically_not_lexically(self) -> None:
        conversation = conversation_from_attributes(
            {
                "gen_ai.prompt.10.role": "user",
                "gen_ai.prompt.10.content": "eleventh",
                "gen_ai.prompt.2.role": "user",
                "gen_ai.prompt.2.content": "third",
            }
        )

        assert conversation.prompts == [("user", "third"), ("user", "eleventh")]

    def test_plural_prompts_form_is_accepted(self) -> None:
        conversation = conversation_from_attributes({"gen_ai.prompts.0.role": "user", "gen_ai.prompts.0.content": "hi"})

        assert conversation.prompts == [("user", "hi")]

    def test_role_without_content_is_not_a_message(self) -> None:
        conversation = conversation_from_attributes({"gen_ai.prompt.0.role": "user"})

        assert conversation.prompts == []

    def test_content_without_role_keeps_the_text(self) -> None:
        conversation = conversation_from_attributes({"gen_ai.prompt.0.content": "hi"})

        assert conversation.prompts == [("", "hi")]

    def test_raw_input_is_suppressed_when_indexed_pairs_exist(self) -> None:
        conversation = conversation_from_attributes(
            {"gen_ai.prompt.0.role": "user", "gen_ai.prompt.0.content": "hi", "input": "hi"}
        )

        assert conversation.raw_input is None

    def test_raw_io_is_kept_when_there_are_no_indexed_pairs(self) -> None:
        conversation = conversation_from_attributes({"input": "hi", "output": "hello"})

        assert (conversation.raw_input, conversation.raw_output) == ("hi", "hello")

    @pytest.mark.parametrize("attributes,expected", [({"gen_ai.request.model": "x"}, True), ({"input": "hi"}, False)])
    def test_gen_ai_authorship_is_detected(self, attributes: Dict[str, Any], expected: bool) -> None:
        assert conversation_from_attributes(attributes).carries_gen_ai is expected

    def test_empty_attributes_yield_an_empty_conversation(self) -> None:
        conversation = conversation_from_attributes(None)

        assert conversation.prompts == []
        assert conversation.completions == []
        assert conversation.carries_gen_ai is False

    def test_raw_io_is_dropped_when_not_allowed(self) -> None:
        conversation = conversation_from_attributes({"input": "hi", "output": "hello"})

        assert messages_for_parent(conversation, allow_raw_io=False) == []

    def test_raw_io_takes_fallback_roles_when_allowed(self) -> None:
        conversation = conversation_from_attributes({"input": "hi", "output": "hello"})

        messages = messages_for_parent(conversation, allow_raw_io=True)
        assert [(message.side, message.role, message.content) for message in messages] == [
            (ConversationSide.PROMPT, "user", "hi"),
            (ConversationSide.COMPLETION, "assistant", "hello"),
        ]


class TestChatContextParsing:
    def test_string_and_dict_payloads_agree(self) -> None:
        payload = {"items": [{"type": "message", "role": "user", "content": ["hi"]}]}

        assert messages_from_chat_ctx(payload) == messages_from_chat_ctx(json.dumps(payload)) == [("user", "hi")]

    def test_multiple_text_parts_are_joined_by_newline(self) -> None:
        payload = {"items": [{"type": "message", "role": "user", "content": ["one", "two"]}]}

        assert messages_from_chat_ctx(payload) == [("user", "one\ntwo")]

    def test_non_text_content_parts_are_skipped(self) -> None:
        payload = {"items": [{"type": "message", "role": "user", "content": [{"type": "image"}, "caption"]}]}

        assert messages_from_chat_ctx(payload) == [("user", "caption")]

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            None,
            [],
            {"items": "not a list"},
            {},
            {"items": [{"type": "message", "role": "", "content": ["hi"]}]},
            {"items": [{"type": "message", "role": "user", "content": []}]},
            {"items": [{"type": "message", "role": "user"}]},
            {"items": ["not a mapping"]},
        ],
    )
    def test_malformed_payloads_yield_no_messages(self, payload: Any) -> None:
        assert messages_from_chat_ctx(payload) == []


class TestEventPayloadParsing:
    def test_choice_content_is_preferred_over_tool_calls(self) -> None:
        assert content_of_choice_event({"content": "reply", "tool_calls": ['{"name": "x"}']}) == "reply"

    def test_tool_only_reply_falls_back_to_the_tool_calls(self) -> None:
        attributes = {"tool_calls": ['{"name": "lookup"}', '{"name": "book"}']}

        assert content_of_choice_event(attributes) == '[{"name": "lookup"}, {"name": "book"}]'

    def test_tool_call_fallback_is_valid_json(self) -> None:
        rendered = content_of_choice_event({"tool_calls": ['{"name": "lookup"}']})

        assert json.loads(str(rendered)) == [{"name": "lookup"}]

    @pytest.mark.parametrize("attributes", [None, {}, {"content": ""}, {"tool_calls": []}, {"tool_calls": ""}])
    def test_empty_choice_events_carry_no_content(self, attributes: Any) -> None:
        assert content_of_choice_event(attributes) is None

    @pytest.mark.parametrize(
        "attributes,expected",
        [({"role": "tool"}, "tool"), ({}, "assistant"), ({"role": ""}, "assistant"), (None, "assistant")],
    )
    def test_choice_role_defaults_to_assistant(self, attributes: Any, expected: str) -> None:
        assert role_of_choice_event(attributes) == expected

    @pytest.mark.parametrize("attributes", [None, {}, {"content": ""}, {"content": None}])
    def test_event_without_content_returns_none(self, attributes: Any) -> None:
        assert content_of_event(attributes) is None

    def test_non_string_event_content_is_stringified(self) -> None:
        assert content_of_event({"content": 42}) == "42"


class TestShieldedTracerProvider:
    def test_livekit_added_processors_are_refused(self) -> None:
        delegate = TracerProvider()
        shield = _ShieldedTracerProvider(delegate)
        exporter = InMemorySpanExporter()

        shield.add_span_processor(SimpleSpanProcessor(exporter))
        delegate.get_tracer("x").start_span("s").end()

        assert exporter.get_finished_spans() == ()

    def test_shutdown_is_absorbed(self) -> None:
        delegate = TracerProvider()
        exporter = InMemorySpanExporter()
        delegate.add_span_processor(SimpleSpanProcessor(exporter))

        _ShieldedTracerProvider(delegate).shutdown()
        delegate.get_tracer("x").start_span("s").end()

        assert len(exporter.get_finished_spans()) == 1, "the delegate's pipeline must survive LiveKit's teardown"

    def test_get_tracer_delegates(self) -> None:
        delegate = TracerProvider()

        assert _ShieldedTracerProvider(delegate).get_tracer("x") is delegate.get_tracer("x")

    def test_resource_is_the_delegates(self) -> None:
        delegate = TracerProvider(resource=Resource.create({"service.name": "voice-agent"}))

        assert _ShieldedTracerProvider(delegate).resource is delegate.resource

    def test_resource_falls_back_when_the_delegate_has_none(self) -> None:
        class _ApiOnlyProvider:
            pass

        shield = _ShieldedTracerProvider(_ApiOnlyProvider())  # type: ignore[arg-type]

        assert shield.resource == Resource.get_empty()

    def test_force_flush_propagates(self) -> None:
        calls: List[int] = []

        class _RecordingProvider:
            def force_flush(self, timeout_millis: int = 30000) -> bool:
                calls.append(timeout_millis)
                return True

        assert _ShieldedTracerProvider(_RecordingProvider()).force_flush(500) is True  # type: ignore[arg-type]
        assert calls == [500]

    def test_force_flush_tolerates_a_provider_without_one(self) -> None:
        class _ApiOnlyProvider:
            pass

        assert _ShieldedTracerProvider(_ApiOnlyProvider()).force_flush() is True  # type: ignore[arg-type]


class TestSessionStartHook:
    def test_start_result_is_returned_untouched(self) -> None:
        from netra.instrumentation.livekit.wrappers import wrap_start

        async def fake_start(**kwargs: Any) -> str:
            return "started"

        result = asyncio.run(wrap_start(fake_start, object(), (), {"room": None}))

        assert result == "started"

    def test_start_exceptions_propagate_unchanged(self) -> None:
        from netra.instrumentation.livekit.wrappers import wrap_start

        async def failing_start(**kwargs: Any) -> None:
            raise RuntimeError("livekit blew up")

        with pytest.raises(RuntimeError, match="livekit blew up"):
            asyncio.run(wrap_start(failing_start, object(), (), {}))


@pytest.fixture
def fake_agent_session(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stand-in ``livekit.agents.voice.agent_session`` module.

    The hook is installed and removed by module path, so the wrap/unwrap round
    trip can be exercised without a livekit-agents install — the shape of the
    module tree is the only thing that matters to it.
    """
    import sys
    from types import ModuleType

    class AgentSession:
        async def start(self, agent: Any = None, **kwargs: Any) -> str:
            return "started"

    modules: Dict[str, ModuleType] = {}
    for path in ("livekit", "livekit.agents", "livekit.agents.voice", "livekit.agents.voice.agent_session"):
        modules[path] = ModuleType(path)
    modules["livekit.agents.voice.agent_session"].AgentSession = AgentSession  # type: ignore[attr-defined]
    modules["livekit.agents.voice"].agent_session = modules["livekit.agents.voice.agent_session"]  # type: ignore[attr-defined]
    modules["livekit.agents"].voice = modules["livekit.agents.voice"]  # type: ignore[attr-defined]
    modules["livekit"].agents = modules["livekit.agents"]  # type: ignore[attr-defined]
    for path, module in modules.items():
        monkeypatch.setitem(sys.modules, path, module)

    # The hook guard is a module global; leaving it set would leak into later tests.
    monkeypatch.setattr(livekit_instrumentation, "_session_hook_installed", False)
    yield AgentSession
    monkeypatch.setattr(livekit_instrumentation, "_session_hook_installed", False)


class TestSessionHookLifecycle:
    @staticmethod
    def _is_wrapped(agent_session: Any) -> bool:
        return isinstance(agent_session.start, ObjectProxy)

    def test_hook_wraps_agent_session_start(self, fake_agent_session: Any) -> None:
        livekit_instrumentation._install_session_hook()

        assert self._is_wrapped(fake_agent_session)

    def test_uninstrument_actually_removes_the_wrapper(self, fake_agent_session: Any) -> None:
        livekit_instrumentation._install_session_hook()
        assert self._is_wrapped(fake_agent_session), "precondition: the hook is installed"

        NetraLiveKitInstrumentor()._uninstrument()

        assert not self._is_wrapped(
            fake_agent_session
        ), "unwrap() cannot walk a dotted attribute path and returns None instead of raising"

    def test_reinstalling_after_uninstrument_wraps_exactly_once(self, fake_agent_session: Any) -> None:
        livekit_instrumentation._install_session_hook()
        NetraLiveKitInstrumentor()._uninstrument()
        livekit_instrumentation._install_session_hook()

        # A stale wrapper left behind by uninstrument would nest here, running the
        # session-id hook twice per start().
        assert self._is_wrapped(fake_agent_session)
        assert not isinstance(fake_agent_session.start.__wrapped__, ObjectProxy)

    def test_install_is_idempotent(self, fake_agent_session: Any) -> None:
        livekit_instrumentation._install_session_hook()
        livekit_instrumentation._install_session_hook()

        assert not isinstance(fake_agent_session.start.__wrapped__, ObjectProxy)


class _TraceShapeHarness:
    """A provider wired for a trace's *shape* rather than its attributes.

    Carries the processors that decide what a trace's root is, in the order
    ``netra/tracer.py`` registers them — the root-instrument filter that owns the
    candidate registry first, then the LLM-call marker before the root bookkeeping
    it reads, the exporting processor after all three, and the LiveKit processors
    last, as ``_instrument()`` appends them.

    ``RootInstrumentFilterProcessor`` is not optional scenery. It is the owner of
    ``ROOT_BLOCK_CANDIDATES``, which is the only thing the exporter consults about
    candidacy, and it runs *before* the LiveKit processors — so it snapshots a
    span's parent before any re-rooting rewrites it. A harness without it cannot
    see that interaction at all.

    The exporting processor defaults to a ``SimpleSpanProcessor``, so every span
    exports in a batch of its own: the cross-batch case, where a child reaches the
    exporter long before the span above it does, is the default here rather than a
    special case to arrange. Pass ``batched=True`` for the shape netra actually
    ships (``disable_batch=False``), where nothing reaches the exporter until the
    flush and an ``on_end`` hook still has time to act.

    Args:
        root_instrument_names: Instruments allowed to emit root spans. Defaults to
            the shipped set, which includes ``livekit``.
        batched: Whether to buffer through a ``BatchSpanProcessor``.
    """

    def __init__(self, root_instrument_names: Optional[Set[str]] = None, batched: bool = False) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        allowed = DEFAULT_ROOT_INSTRUMENT_NAMES if root_instrument_names is None else root_instrument_names
        self.provider.add_span_processor(RootInstrumentFilterProcessor(allowed))
        self.provider.add_span_processor(LlmTraceIdentifierSpanProcessor())
        self.provider.add_span_processor(RootSpanProcessor())
        exporting = FilteringSpanExporter(self.exporter, [])
        self._batch_processor = BatchSpanProcessor(exporting) if batched else None
        self.provider.add_span_processor(
            self._batch_processor if self._batch_processor is not None else SimpleSpanProcessor(exporting)
        )
        self.provider.add_span_processor(SessionRootSpanProcessor())
        self.livekit_tracer = self.provider.get_tracer(LIVEKIT_SCOPE_NAME)

    def tracer(self, scope_name: str) -> Any:
        """Return a tracer for some other instrumentation scope."""
        return self.provider.get_tracer(scope_name)

    def _flush(self) -> None:
        """Drain the batch processor, if this harness has one."""
        if self._batch_processor is not None:
            self._batch_processor.force_flush()

    def exported_names(self) -> List[str]:
        """Return the names of every span that survived export."""
        self._flush()
        return [span.name for span in self.exporter.get_finished_spans()]

    def exported_root_names(self) -> List[str]:
        """Return the names of every exported span that has no parent left."""
        self._flush()
        return [span.name for span in self.exporter.get_finished_spans() if span.parent is None]

    def exported(self, name: str) -> ReadableSpan:
        """Return the single exported span called *name*."""
        self._flush()
        matches = [span for span in self.exporter.get_finished_spans() if span.name == name]
        assert len(matches) == 1, f"expected exactly one exported {name!r} span, got {len(matches)}"
        return matches[0]

    def start_session(self) -> Any:
        """Start an ``agent_session`` the way ``AgentSession.start`` does.

        livekit-agents calls ``tracer.start_span`` with no explicit context and
        attaches the result afterwards, so the parent is resolved from the ambient
        context at ``on_start``.
        """
        return self.livekit_tracer.start_span("agent_session")


def _reset_trace_shape_globals() -> None:
    """Clear the process-global registries a trace's shape is decided from."""
    from netra.processors import root_instrument_filter_processor as rifp

    with rifp._root_candidates_lock:
        rifp.ROOT_BLOCK_CANDIDATES.clear()
        rifp.PINNED_ROOT_BLOCK_CANDIDATES.clear()
    RootSpanProcessor().shutdown()


@pytest.fixture
def shape_harness() -> Any:
    """A ``_TraceShapeHarness`` with every process-global registry isolated."""
    _reset_trace_shape_globals()
    yield _TraceShapeHarness()
    _reset_trace_shape_globals()


@pytest.fixture
def batched_shape_harness() -> Any:
    """A ``_TraceShapeHarness`` that buffers, as ``disable_batch=False`` does."""
    _reset_trace_shape_globals()
    yield _TraceShapeHarness(batched=True)
    _reset_trace_shape_globals()


class TestVoiceTraceRoot:
    """``agent_session`` replaces ``job_entrypoint`` as the root of a job's trace."""

    def test_agent_session_is_exported_as_a_root_span(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = shape_harness.livekit_tracer.start_span("agent_session")
            session.end()

        assert shape_harness.exported("agent_session").parent is None

    def test_agent_session_keeps_the_trace_id_it_was_created_under(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint") as job:
            job_trace_id = job.get_span_context().trace_id
            session = shape_harness.livekit_tracer.start_span("agent_session")
            session.end()

        # The whole point of re-rooting at on_start rather than creating the session
        # span in a cleared context: a new root would have been given a new trace id,
        # and a trace already ingested under this one would split in two.
        assert shape_harness.exported("agent_session").context.trace_id == job_trace_id

    def test_job_entrypoint_is_never_exported(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = shape_harness.livekit_tracer.start_span("agent_session")
            session.end()

        assert shape_harness.exported_names() == ["agent_session"]

    def test_entrypoint_children_exporting_before_the_drop_are_promoted_to_roots(self, shape_harness: Any) -> None:
        """A span the user's entrypoint traced must not be left pointing at the dropped root.

        It ends — and here, exports — while ``job_entrypoint`` is still open, so its
        parent can only be rewritten from the cross-batch candidate registry.
        """
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            with shape_harness.tracer("my.app").start_as_current_span("load_customer_record"):
                pass

        assert shape_harness.exported("load_customer_record").parent is None
        assert "job_entrypoint" not in shape_harness.exported_names()

    def test_agent_session_started_outside_a_job_keeps_its_parent(self, shape_harness: Any) -> None:
        """Console mode, evals, or a session started inside the user's own span."""
        with shape_harness.tracer("my.app").start_as_current_span("handle_request") as caller:
            session = shape_harness.livekit_tracer.start_span("agent_session")
            session.end()

        exported = shape_harness.exported("agent_session")
        assert exported.parent is not None
        assert exported.parent.span_id == caller.get_span_context().span_id

    def test_a_non_livekit_span_named_job_entrypoint_is_left_alone(self, shape_harness: Any) -> None:
        """The scope check is what keeps this processor off another library's spans."""
        with shape_harness.tracer("my.app").start_as_current_span("job_entrypoint"):
            session = shape_harness.livekit_tracer.start_span("agent_session")
            session.end()

        assert "job_entrypoint" in shape_harness.exported_names()
        assert shape_harness.exported("agent_session").parent is not None


class TestVoiceTraceRootBookkeeping:
    """``RootSpanProcessor`` must name ``agent_session``, not the span it replaced."""

    def test_agent_session_is_recorded_as_the_trace_root(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint") as job:
            trace_id = job.get_span_context().trace_id
            session = shape_harness.livekit_tracer.start_span("agent_session")

            assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is session

            session.end()

    def test_the_root_survives_job_entrypoint_ending_mid_call(self, shape_harness: Any) -> None:
        """The entrypoint returns seconds into a call; the session runs on for minutes.

        ``Netra.set_attribute_on_root_span`` and the LLM-call marker resolve the root
        through this mapping, so losing it here means losing them for the rest of the
        call.
        """
        job = shape_harness.livekit_tracer.start_span("job_entrypoint")
        with trace.use_span(job, end_on_exit=False):
            session = shape_harness.livekit_tracer.start_span("agent_session")
        trace_id = job.get_span_context().trace_id

        job.end()

        assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is session
        session.end()
        assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is None

    def test_llm_call_marker_lands_on_agent_session(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = shape_harness.livekit_tracer.start_span("agent_session")
            with trace.use_span(session, end_on_exit=False):
                completion = shape_harness.tracer("netra.instrumentation.openai").start_span("openai.chat")
                completion.set_attribute("gen_ai.request.model", "gpt-4o")
                completion.end()
            session.end()

        assert shape_harness.exported("agent_session").attributes.get("netra.trace.llm.call") is True


class TestJobsThatOpenNoSession:
    """``job_entrypoint`` is only dropped once an ``agent_session`` replaces it.

    A LiveKit job need not be a voice job, and dropping the root of one that opened
    no session leaves its spans as parentless siblings with nothing above them.
    Exercised through the batched harness because that is netra's shipped default
    (``disable_batch=False``) and the only configuration in which an ``on_end``
    decision still lands before export.
    """

    def test_job_entrypoint_survives_when_no_agent_session_is_opened(self, batched_shape_harness: Any) -> None:
        with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            with batched_shape_harness.tracer("my.app").start_as_current_span("load_customer_record"):
                pass
            with batched_shape_harness.tracer("my.app").start_as_current_span("send_summary_email"):
                pass

        assert batched_shape_harness.exported_root_names() == ["job_entrypoint"]
        assert batched_shape_harness.exported("load_customer_record").parent is not None

    def test_job_entrypoint_survives_when_session_start_raises(self, batched_shape_harness: Any) -> None:
        """``start()`` can fail before livekit-agents ever opens the session span."""
        with pytest.raises(RuntimeError):
            with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
                raise RuntimeError("session.start() failed")

        assert batched_shape_harness.exported_root_names() == ["job_entrypoint"]

    def test_job_entrypoint_is_still_dropped_once_a_session_opens(self, batched_shape_harness: Any) -> None:
        """The release must not fire for the case the drop exists for."""
        with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = batched_shape_harness.start_session()
            session.end()

        assert batched_shape_harness.exported_root_names() == ["agent_session"]

    def test_a_session_closing_before_the_entrypoint_returns_still_drops_it(self, batched_shape_harness: Any) -> None:
        """A short call ends its session first, which empties the root mapping.

        The promotion is therefore recorded on ``job_entrypoint`` itself rather than
        inferred from ``RootSpanProcessor``, which by this point no longer names the
        session — otherwise a brief call would look exactly like a job that opened
        no session, and the entrypoint would come back as a second root.
        """
        with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = batched_shape_harness.start_session()
            session.end()
            # The entrypoint goes on doing work after the session has closed.
            with batched_shape_harness.tracer("my.app").start_as_current_span("write_summary"):
                pass

        assert "job_entrypoint" not in batched_shape_harness.exported_names()
        assert batched_shape_harness.exported_root_names() == ["agent_session", "write_summary"]

    def test_entrypoint_children_are_never_left_pointing_at_a_dropped_parent(self, shape_harness: Any) -> None:
        """The reason ``job_entrypoint`` is marked at its start rather than at promotion.

        ``load_customer_record`` ends — and here, exports — while the entrypoint is
        still running and before any session exists. Deferring the mark until a
        session claimed the root would leave this span referencing a parent that
        never arrives, which is worse than the stray root it becomes instead.
        """
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            with shape_harness.tracer("my.app").start_as_current_span("load_customer_record"):
                pass
            session = shape_harness.start_session()
            session.end()

        exported_ids = {span.context.span_id for span in shape_harness.exporter.get_finished_spans()}
        for span in shape_harness.exporter.get_finished_spans():
            assert span.parent is None or span.parent.span_id in exported_ids


class TestMultipleSessionsInOneJob:
    """livekit-agents allows several ``AgentSession``\\ s per job; only one may re-root."""

    def test_the_first_session_keeps_the_root_while_a_sibling_starts(self, shape_harness: Any) -> None:
        """Promoting both would leave the bookkeeping naming whichever started last.

        Every root-span write for the trace — the LLM-call marker,
        ``Netra.set_attribute_on_root_span`` — resolves through that mapping, so the
        first session's data would land on the second session's span.
        """
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint") as job:
            trace_id = job.get_span_context().trace_id
            first = shape_harness.start_session()
            second = shape_harness.start_session()

            assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is first

            first.end()
            second.end()

    def test_a_sibling_session_keeps_the_parent_it_was_created_with(self, shape_harness: Any) -> None:
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint") as job:
            first = shape_harness.start_session()
            second = shape_harness.start_session()
            assert second.parent is not None
            assert second.parent.span_id == job.get_span_context().span_id
            first.end()
            second.end()

    def test_sessions_that_run_one_after_another_are_each_re_rooted(self, shape_harness: Any) -> None:
        """The mapping is released when the recorded root ends, so the next one claims it."""
        with shape_harness.livekit_tracer.start_as_current_span("job_entrypoint") as job:
            trace_id = job.get_span_context().trace_id

            first = shape_harness.start_session()
            assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is first
            first.end()

            second = shape_harness.start_session()
            assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is second
            second.end()

        assert shape_harness.exported_root_names() == ["agent_session", "agent_session"]


class TestVoiceTraceRootUnderRegistryPressure:
    """The drop must outlive a registry flooded by incidental candidates."""

    def test_job_entrypoint_is_dropped_after_the_registry_overflows(self, batched_shape_harness: Any) -> None:
        """Overflow eviction must not hand back the root the trace was re-rooted to lose.

        ``ROOT_BLOCK_CANDIDATES`` evicts by insertion order and does not care whether
        a span is still open, so ``job_entrypoint`` — recorded once, at the very start
        of a job — is among the first to go when a burst of ``httpx`` spans (no HTTP
        client is on the root allow-list) fills the registry. Losing that entry does
        not merely skip a reparent: the entrypoint returns to the export as a second
        root of the trace.
        """
        from netra.processors import root_instrument_filter_processor as rifp

        http_tracer = batched_shape_harness.tracer("opentelemetry.instrumentation.httpx")
        with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = batched_shape_harness.start_session()
            for index in range(rifp._MAX_ROOT_CANDIDATES + 1):
                http_tracer.start_span(f"GET /{index}").end()
            session.end()

        assert "job_entrypoint" not in batched_shape_harness.exported_names()
        assert batched_shape_harness.exported_root_names() == ["agent_session"]

    def test_the_registry_still_honours_its_size_cap(self, batched_shape_harness: Any) -> None:
        """Pinning must not turn into an unbounded exemption."""
        from netra.processors import root_instrument_filter_processor as rifp

        http_tracer = batched_shape_harness.tracer("opentelemetry.instrumentation.httpx")
        with batched_shape_harness.livekit_tracer.start_as_current_span("job_entrypoint"):
            session = batched_shape_harness.start_session()
            for index in range(rifp._MAX_ROOT_CANDIDATES * 2):
                http_tracer.start_span(f"GET /{index}").end()
            session.end()

        assert len(rifp.ROOT_BLOCK_CANDIDATES) <= rifp._MAX_ROOT_CANDIDATES
