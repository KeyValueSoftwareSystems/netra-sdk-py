"""Tests for Netra's LiveKit voice-agent instrumentation.

The suite exercises the package through the real OpenTelemetry SDK rather than
mocks: spans are created from a tracer whose instrumentation scope is
``livekit-agents`` — the only thing the processor gates on — so no
``livekit-agents`` install is required.
"""

import asyncio
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from wrapt import ObjectProxy

from netra.instrumentation import livekit as livekit_instrumentation
from netra.instrumentation.livekit import NetraLiveKitInstrumentor
from netra.instrumentation.livekit.call_span import (
    _MAX_OPEN_CALL_SPANS,
    CALL_SPAN_FIELD,
    REROOTED_ATTRIBUTE,
    call_id_scope,
    call_spans,
    end_all_call_spans,
)
from netra.instrumentation.livekit.provider_binding import _ShieldedTracerProvider
from netra.instrumentation.livekit.trace_processor import SpanMappingProcessor, record_stt_usage
from netra.instrumentation.livekit.utils import (
    AGENT_SESSION_SPAN_NAME,
    CALL_SPAN_NAME,
    JOB_ENTRYPOINT_SPAN_NAME,
    LIVEKIT_SCOPE_NAME,
    MAX_CONVERSATION_MESSAGES_PER_SIDE,
    NETRA_CONVERSATION_TRUNCATED,
    NETRA_ENTITY_TYPE,
    NETRA_SPAN_TYPE,
    USER_TURN_SPAN_NAME,
    ConversationSide,
    content_of_choice_event,
    content_of_event,
    conversation_from_attributes,
    is_zero_usage,
    messages_for_parent,
    messages_from_chat_ctx,
    netra_span_type_for,
    role_of_choice_event,
    stt_pricing_attributes_from,
    tts_pricing_attributes_from,
)
from netra.instrumentation.livekit.wrappers import _listen_for_metrics, wrap_aclose, wrap_start
from netra.processors.root_span_processor import RootSpanProcessor
from netra.processors.session_span_processor import SessionSpanProcessor
from netra.span_wrapper import SpanType

pytestmark = pytest.mark.unit


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

    def test_job_entrypoint_is_no_longer_marked_as_a_workflow(self, harness: _Harness) -> None:
        # The workflow entity moved onto ``livekit-call``, which is now the span
        # wrapping the whole call. Two nested spans both claiming to be the
        # workflow is what this asserts is gone.
        assert "netra.entity.type" not in _record(harness, JOB_ENTRYPOINT_SPAN_NAME, {})

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

    def test_token_counts_are_lifted_for_a_token_priced_model(self, harness: _Harness) -> None:
        # The gpt-4o-mini-tts shape: priced on tokens, so the character count alone
        # would leave the call billing nothing.
        metrics = json.dumps(
            {
                "characters_count": 38,
                "input_tokens": 8,
                "output_tokens": 85,
                "audio_duration": 3.288,
                "metadata": {"model_name": "gpt-4o-mini-tts"},
            }
        )
        attributes = _record(harness, "tts_request", {"lk.tts_metrics": metrics})

        assert attributes["gen_ai.usage.prompt_tokens"] == 8
        assert attributes["gen_ai.usage.completion_tokens"] == 85
        assert attributes["gen_ai.audio.duration"] == pytest.approx(3.288)
        assert attributes["gen_ai.usage.prompt.character_count"] == 38
        assert attributes["netra.usage.source"] == "framework"

    def test_zero_token_counts_are_dropped_rather_than_claimed(self, harness: _Harness) -> None:
        # The cartesia/sonic-3 shape: billed on characters, reporting 0 tokens.
        metrics = json.dumps(
            {
                "characters_count": 87,
                "input_tokens": 0,
                "output_tokens": 0,
                "audio_duration": 4.736875,
                "metadata": {"model_name": "cartesia/sonic-3"},
            }
        )
        attributes = _record(harness, "tts_request", {"lk.tts_metrics": metrics})

        assert "gen_ai.usage.prompt_tokens" not in attributes
        assert "gen_ai.usage.completion_tokens" not in attributes
        assert attributes["gen_ai.usage.prompt.character_count"] == 87
        assert attributes["gen_ai.audio.duration"] == pytest.approx(4.736875)


class TestSttPricing:
    """STT usage arrives out-of-band — see ``record_stt_usage``."""

    @staticmethod
    def _metrics(**overrides: Any) -> Dict[str, Any]:
        """A serialised ``STTMetrics`` for a streaming recognition."""
        payload: Dict[str, Any] = {
            "type": "stt_metrics",
            "request_id": "019ff5ed-0bb4-7ea0",
            "audio_duration": 2.5,
            "input_tokens": 0,
            "output_tokens": 0,
            "streamed": True,
            "metadata": {"model_name": "deepgram/nova-3"},
        }
        payload.update(overrides)
        return payload

    @pytest.fixture
    def call_id(self, harness: _Harness) -> Iterator[int]:
        """Open a call and attach its id, as ``wrap_start`` does around ``start()``.

        Every ``user_turn`` started inside the test therefore registers under this
        call, which is what ``record_stt_usage`` is handed to find it again.
        """
        call = harness.livekit_tracer.start_span(CALL_SPAN_NAME)
        with call_id_scope(call):
            yield call.get_span_context().span_id

    @staticmethod
    def _turn(harness: _Harness) -> Any:
        """Start a ``user_turn`` span in the ambient call."""
        return harness.livekit_tracer.start_span(USER_TURN_SPAN_NAME)

    @staticmethod
    def _turns_by_id(harness: _Harness) -> Dict[int, Dict[str, Any]]:
        """The exported ``user_turn`` spans' attributes, keyed by span id."""
        exported = [span for span in harness.exporter.get_finished_spans() if span.name == USER_TURN_SPAN_NAME]
        return {span.get_span_context().span_id: dict(span.attributes or {}) for span in exported}

    def test_reported_usage_lands_on_the_recording_turn(self, harness: _Harness, call_id: int) -> None:
        span = self._turn(harness)
        record_stt_usage(call_id, self._metrics(input_tokens=12, output_tokens=4))
        span.end()

        attributes = harness.attributes("user_turn")
        assert attributes["gen_ai.audio.duration"] == pytest.approx(2.5)
        assert attributes["gen_ai.usage.prompt_tokens"] == 12
        assert attributes["gen_ai.usage.completion_tokens"] == 4
        assert attributes["gen_ai.request.model"] == "deepgram/nova-3"
        assert attributes["netra.usage.source"] == "framework"

    def test_incremental_samples_accumulate_over_a_turn(self, harness: _Harness, call_id: int) -> None:
        # A streaming STT emits RECOGNITION_USAGE per final transcript, each
        # carrying only the audio since the last one.
        span = self._turn(harness)
        record_stt_usage(call_id, self._metrics(audio_duration=2.5, input_tokens=12))
        record_stt_usage(call_id, self._metrics(audio_duration=1.25, input_tokens=3))
        span.end()

        attributes = harness.attributes("user_turn")
        assert attributes["gen_ai.audio.duration"] == pytest.approx(3.75)
        assert attributes["gen_ai.usage.prompt_tokens"] == 15

    def test_a_json_string_is_accepted_like_a_mapping(self, harness: _Harness, call_id: int) -> None:
        span = self._turn(harness)
        record_stt_usage(call_id, json.dumps(self._metrics()))
        span.end()

        assert harness.attributes("user_turn")["gen_ai.audio.duration"] == pytest.approx(2.5)

    def test_connection_timing_sample_writes_no_usage(self, harness: _Harness, call_id: int) -> None:
        # ``_report_connection_acquired`` reports a zero-duration sample purely to
        # record when the socket was acquired.
        span = self._turn(harness)
        record_stt_usage(call_id, self._metrics(request_id="", audio_duration=0.0, acquire_time=0.4))
        span.end()

        attributes = harness.attributes("user_turn")
        assert "gen_ai.audio.duration" not in attributes
        assert "netra.usage.source" not in attributes

    def test_usage_arriving_after_the_turn_ended_is_dropped(self, harness: _Harness, call_id: int) -> None:
        span = self._turn(harness)
        span.end()

        record_stt_usage(call_id, self._metrics())

        assert "gen_ai.audio.duration" not in harness.attributes("user_turn")

    def test_a_late_turn_end_does_not_evict_its_successor(self, harness: _Harness, call_id: int) -> None:
        first = self._turn(harness)
        second = harness.livekit_tracer.start_span(USER_TURN_SPAN_NAME, context=trace.set_span_in_context(first))
        first.end()

        record_stt_usage(call_id, self._metrics())
        second.end()

        by_id = self._turns_by_id(harness)
        assert by_id[second.get_span_context().span_id]["gen_ai.audio.duration"] == pytest.approx(2.5)
        assert "gen_ai.audio.duration" not in by_id[first.get_span_context().span_id]

    def test_two_calls_in_one_job_keep_their_usage_apart(self, harness: _Harness) -> None:
        # Two sessions in one job share a trace id: ``livekit-call`` inherits the
        # job's rather than minting one, and only the first re-roots. Keyed on the
        # trace, the second turn would take both callers' audio and the first would
        # be billed nothing.
        job = harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        with trace.use_span(job, end_on_exit=False):
            first_call = harness.livekit_tracer.start_span(CALL_SPAN_NAME)
            second_call = harness.livekit_tracer.start_span(CALL_SPAN_NAME)

            with call_id_scope(first_call):
                first_turn = self._turn(harness)
            with call_id_scope(second_call):
                second_turn = self._turn(harness)

        assert first_turn.get_span_context().trace_id == second_turn.get_span_context().trace_id

        record_stt_usage(first_call.get_span_context().span_id, self._metrics(audio_duration=2.5))
        record_stt_usage(second_call.get_span_context().span_id, self._metrics(audio_duration=7.5))
        first_turn.end()
        second_turn.end()

        by_id = self._turns_by_id(harness)
        assert by_id[first_turn.get_span_context().span_id]["gen_ai.audio.duration"] == pytest.approx(2.5)
        assert by_id[second_turn.get_span_context().span_id]["gen_ai.audio.duration"] == pytest.approx(7.5)

    def test_a_turn_outside_a_call_is_never_registered(self, harness: _Harness) -> None:
        # No call id in scope means ``wrap_start`` never ran, so nothing is
        # subscribed to that session's metrics either.
        span = self._turn(harness)
        record_stt_usage(0xDEADBEEF, self._metrics())
        span.end()

        assert "gen_ai.audio.duration" not in harness.attributes("user_turn")

    @pytest.mark.parametrize("payload", ["not json", None, [], {"metadata": "not a mapping"}])
    def test_a_malformed_payload_is_dropped_without_raising(
        self, harness: _Harness, call_id: int, payload: Any
    ) -> None:
        span = self._turn(harness)
        record_stt_usage(call_id, payload)
        span.end()

        attributes = harness.attributes("user_turn")
        assert "gen_ai.audio.duration" not in attributes
        assert "gen_ai.request.model" not in attributes

    def test_usage_for_an_unknown_call_is_dropped_without_raising(self) -> None:
        record_stt_usage(0xDEADBEEF, self._metrics())

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"audio_duration": 2.5}, 2.5),
            ({"audio_duration": 0.0}, None),
            ({"audio_duration": -1.0}, None),
            ({"audio_duration": True}, None),
            ({"audio_duration": 3}, 3.0),
            ({}, None),
        ],
    )
    def test_duration_extraction_tolerates_every_shape(self, payload: Any, expected: Optional[float]) -> None:
        assert stt_pricing_attributes_from(payload).audio_duration == expected

    def test_transcription_reports_no_character_count(self) -> None:
        assert stt_pricing_attributes_from(self._metrics(characters_count=99)).character_count is None


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


class _CallHarness:
    """A provider carrying the whole processor chain a call span depends on.

    Registration order mirrors production (``netra/tracer.py`` then
    ``_instrument()``): Netra's own processors, then the exporting processor, then
    the LiveKit ones. That ordering is what makes ``SpanMappingProcessor.on_end``
    the *later* hook, which is where the call span is closed.
    """

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SessionSpanProcessor())
        self.provider.add_span_processor(RootSpanProcessor())
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.provider.add_span_processor(SpanMappingProcessor())
        self.livekit_tracer = self.provider.get_tracer(LIVEKIT_SCOPE_NAME)

    def finished(self, name: str) -> ReadableSpan:
        """Return the single finished span called *name*."""
        matches = [span for span in self.exporter.get_finished_spans() if span.name == name]
        assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
        return matches[0]

    def finished_count(self, name: str) -> int:
        """Return how many finished spans are called *name*."""
        return len([span for span in self.exporter.get_finished_spans() if span.name == name])

    def attributes(self, name: str) -> Dict[str, Any]:
        """Return the exported attributes of the finished span called *name*."""
        return dict(self.finished(name).attributes or {})


class _FakeAgentSession:
    """Stand-in for LiveKit's ``AgentSession``, reproducing its span lifecycle.

    Only the two things the wrappers actually depend on: ``start()`` opens the
    ``agent_session`` span in whatever context is current — which is how it ends up
    under the call span — and ``_aclose_impl()`` ends it, clearing
    ``_session_span`` exactly as livekit-agents does (``agent_session.py:1148-1150``).
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer
        self._session_span: Optional[Any] = None

    async def start(self, **kwargs: Any) -> str:
        self._session_span = self._tracer.start_span(AGENT_SESSION_SPAN_NAME)
        return "started"

    async def aclose_impl(self, **kwargs: Any) -> None:
        if self._session_span is not None:
            self._session_span.end()
            self._session_span = None


class _SpeakingAgentSession(_FakeAgentSession):
    """A session that opens a ``user_turn`` span and emits metrics, as LiveKit does.

    The turn is opened *inside* ``start()``, under the session span, because that is
    where livekit-agents opens it: from the audio-recognition task, which snapshots
    the context ``wrap_start`` made current. ``on``/``emit`` reproduce the
    ``EventEmitter`` surface the metrics subscription is registered on.
    """

    def __init__(self, tracer: Any) -> None:
        super().__init__(tracer)
        self.user_turn: Optional[Any] = None
        self._listeners: Dict[str, List[Any]] = {}

    def on(self, event: str, callback: Any) -> Any:
        self._listeners.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, argument: Any) -> None:
        for callback in self._listeners.get(event, []):
            callback(argument)

    async def start(self, **kwargs: Any) -> str:
        result = await super().start(**kwargs)
        with trace.use_span(self._session_span, end_on_exit=False):
            self.user_turn = self._tracer.start_span(USER_TURN_SPAN_NAME)
        return result


class _FakeMetricsEvent:
    """LiveKit's ``MetricsCollectedEvent``: a wrapper around one pydantic metrics model."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.metrics = _FakeMetrics(payload)


class _FakeMetrics:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FailedAgentSession(_FakeAgentSession):
    """A session whose ``agent_session`` span ends ``ERROR``, as a failed call's does.

    LiveKit sets the status on its own session span; this reproduces just that, so
    the mirroring onto the call span can be tested without a livekit-agents install.
    """

    CLOSE_ERROR = "the participant hung up mid-turn"

    async def aclose_impl(self, **kwargs: Any) -> None:
        if self._session_span is not None:
            self._session_span.set_status(trace.Status(trace.StatusCode.ERROR, self.CLOSE_ERROR))
        await super().aclose_impl(**kwargs)


@pytest.fixture
def call_harness(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``_CallHarness`` whose provider is the one ``call_span.py`` creates from.

    ``start_call_span`` resolves its tracer off the global provider, which
    ``Netra.init()`` installs in production. Redirecting ``get_tracer`` keeps the
    harness self-contained rather than mutating global OTel state, and asserts the
    real scope name reaches the provider.
    """
    harness = _CallHarness()

    def get_tracer(name: str, *args: Any, **kwargs: Any) -> Any:
        return harness.provider.get_tracer(name, *args, **kwargs)

    monkeypatch.setattr(trace, "get_tracer", get_tracer)
    # Both registries are process-global, so a leaked entry would surface as a
    # phantom call span in a later test.
    call_spans.pop_all()
    RootSpanProcessor().shutdown()
    yield harness
    call_spans.pop_all()
    RootSpanProcessor().shutdown()


@pytest.fixture
def fake_livekit_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the minimum ``livekit.agents`` surface the wrappers read.

    ``_resolve_session_id`` needs ``get_job_context`` and ``is_given`` to be
    importable at all; without them it returns ``None`` and the session-id path is
    never exercised. ``get_job_context`` returns ``None`` here — no job — so the id
    falls back to the room name, which is the console/eval-mode path.
    """
    import sys
    from types import ModuleType

    livekit = ModuleType("livekit")
    agents = ModuleType("livekit.agents")
    utils = ModuleType("livekit.agents.utils")
    agents.get_job_context = lambda required=True: None  # type: ignore[attr-defined]
    utils.is_given = lambda value: value is not None  # type: ignore[attr-defined]
    agents.utils = utils  # type: ignore[attr-defined]
    livekit.agents = agents  # type: ignore[attr-defined]
    for path, module in (("livekit", livekit), ("livekit.agents", agents), ("livekit.agents.utils", utils)):
        monkeypatch.setitem(sys.modules, path, module)


class _FakeRoom:
    def __init__(self, name: str) -> None:
        self.name = name


def _run_call(
    harness: _CallHarness,
    *,
    parent: Optional[Any] = None,
    room: Optional[Any] = None,
) -> _FakeAgentSession:
    """Run one whole call — start then close — and return the session.

    Args:
        harness: The provider to create spans from.
        parent: The span to make current while ``start()`` runs, i.e. the span the
            call span is created underneath. ``None`` for no ambient span.
        room: The ``room`` kwarg ``start()`` is called with.
    """
    session = _FakeAgentSession(harness.livekit_tracer)

    async def call() -> None:
        await wrap_start(session.start, session, (), {"room": room})
        await wrap_aclose(session.aclose_impl, session, (), {})

    if parent is None:
        asyncio.run(call())
    else:
        with trace.use_span(parent, end_on_exit=False):
            asyncio.run(call())
    return session


class TestSttUsageWiring:
    """``wrap_start`` routes the session's STT metrics onto its ``user_turn`` spans."""

    STT_METRICS = {
        "type": "stt_metrics",
        "audio_duration": 2.5,
        "input_tokens": 0,
        "output_tokens": 0,
        "metadata": {"model_name": "deepgram/nova-3"},
    }

    @staticmethod
    def _call(harness: _CallHarness, *events: Dict[str, Any]) -> None:
        """Run one call, emitting *events* while the user turn is open."""
        session = _SpeakingAgentSession(harness.livekit_tracer)

        async def call() -> None:
            await wrap_start(session.start, session, (), {"room": _FakeRoom("console-stt")})
            for payload in events:
                session.emit("metrics_collected", _FakeMetricsEvent(payload))
            assert session.user_turn is not None
            session.user_turn.end()
            await wrap_aclose(session.aclose_impl, session, (), {})

        asyncio.run(call())

    def test_emitted_stt_metrics_price_the_open_user_turn(self, call_harness: _CallHarness) -> None:
        self._call(call_harness, self.STT_METRICS)

        attributes = call_harness.attributes(USER_TURN_SPAN_NAME)
        assert attributes["gen_ai.audio.duration"] == pytest.approx(2.5)
        assert attributes["gen_ai.request.model"] == "deepgram/nova-3"

    def test_other_metrics_on_the_same_event_are_ignored(self, call_harness: _CallHarness) -> None:
        # metrics_collected carries LLM, TTS, VAD and EOU metrics too, and their
        # usage belongs to the spans LiveKit already writes it on.
        self._call(
            call_harness,
            {"type": "tts_metrics", "audio_duration": 9.0, "characters_count": 40},
            {"type": "llm_metrics", "prompt_tokens": 371, "completion_tokens": 21},
        )

        attributes = call_harness.attributes(USER_TURN_SPAN_NAME)
        assert "gen_ai.audio.duration" not in attributes
        assert "gen_ai.usage.prompt_tokens" not in attributes

    def test_a_retried_start_neither_subscribes_twice_nor_double_counts(self, call_harness: _CallHarness) -> None:
        # Both starts run under one job_entrypoint, so their call spans share a
        # trace id — only the first re-roots. A second listener would then record
        # the sample twice and bill the turn 5.0 seconds of audio for 2.5.
        session = _SpeakingAgentSession(call_harness.livekit_tracer)
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)

        async def call() -> None:
            with trace.use_span(job, end_on_exit=False):
                await wrap_start(session.start, session, (), {"room": _FakeRoom("console-stt")})
                assert session.user_turn is not None
                session.user_turn.end()

                await wrap_start(session.start, session, (), {"room": _FakeRoom("console-stt")})
                session.emit("metrics_collected", _FakeMetricsEvent(self.STT_METRICS))
                assert session.user_turn is not None
                session.user_turn.end()
                await wrap_aclose(session.aclose_impl, session, (), {})

        asyncio.run(call())

        assert len(session._listeners["metrics_collected"]) == 1, "the session was subscribed to twice"
        turns = [
            dict(span.attributes or {})
            for span in call_harness.exporter.get_finished_spans()
            if span.name == USER_TURN_SPAN_NAME
        ]
        assert len(turns) == 2
        assert "gen_ai.audio.duration" not in turns[0], "the abandoned turn took usage it never heard"
        assert turns[1]["gen_ai.audio.duration"] == pytest.approx(2.5)

    def test_subscription_bypasses_the_deprecating_session_override(
        self, call_harness: _CallHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # AgentSession.on logs "metrics_collected is deprecated" on every
        # subscription; the SDK must not put that in the user's log.
        import sys
        from types import ModuleType

        subscribed: List[Tuple[Any, str]] = []

        class _EventEmitter:
            def on(self, event: str, callback: Any) -> Any:
                subscribed.append((self, event))
                return callback

        livekit = ModuleType("livekit")
        rtc = ModuleType("livekit.rtc")
        rtc.EventEmitter = _EventEmitter  # type: ignore[attr-defined]
        livekit.rtc = rtc  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "livekit", livekit)
        monkeypatch.setitem(sys.modules, "livekit.rtc", rtc)

        session = _SpeakingAgentSession(call_harness.livekit_tracer)
        _listen_for_metrics(session, lambda event: None)

        assert subscribed == [(session, "metrics_collected")]
        assert session._listeners == {}, "the session's own on() must not have been used"


class TestCallSpanRootsTheTrace:
    """``livekit-call`` replaces ``job_entrypoint`` as the root of a voice trace."""

    def test_call_span_becomes_the_trace_root(self, call_harness: _CallHarness) -> None:
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        _run_call(call_harness, parent=job)
        job.end()

        call = call_harness.finished(CALL_SPAN_NAME)
        entrypoint = call_harness.finished(JOB_ENTRYPOINT_SPAN_NAME)
        session_span = call_harness.finished(AGENT_SESSION_SPAN_NAME)

        assert call.parent is None, "the call span must be the trace root"
        assert entrypoint.parent is not None
        assert entrypoint.parent.span_id == call.context.span_id
        assert session_span.parent is not None
        assert session_span.parent.span_id == call.context.span_id

    def test_re_rooting_keeps_the_job_trace_id(self, call_harness: _CallHarness) -> None:
        # A fresh trace would break audio capture, which reads the ambient trace id
        # after start() has returned — when job_entrypoint is current again.
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        _run_call(call_harness, parent=job)
        job.end()

        trace_ids = {
            call_harness.finished(name).context.trace_id
            for name in (CALL_SPAN_NAME, JOB_ENTRYPOINT_SPAN_NAME, AGENT_SESSION_SPAN_NAME)
        }
        assert len(trace_ids) == 1

    def test_call_span_encloses_both_of_its_children(self, call_harness: _CallHarness) -> None:
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        _run_call(call_harness, parent=job)
        job.end()

        call = call_harness.finished(CALL_SPAN_NAME)
        entrypoint = call_harness.finished(JOB_ENTRYPOINT_SPAN_NAME)
        session_span = call_harness.finished(AGENT_SESSION_SPAN_NAME)

        # Backdated, so the child that started first does not begin before its parent.
        assert call.start_time == entrypoint.start_time
        assert call.start_time <= session_span.start_time
        assert call.end_time is not None and session_span.end_time is not None
        assert call.end_time >= session_span.end_time

    def test_job_entrypoint_records_that_it_was_re_rooted(self, call_harness: _CallHarness) -> None:
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        _run_call(call_harness, parent=job)
        job.end()

        assert call_harness.attributes(JOB_ENTRYPOINT_SPAN_NAME)[REROOTED_ATTRIBUTE] is True

    def test_call_span_is_registered_as_the_traces_root_span(self, call_harness: _CallHarness) -> None:
        # RootSpanProcessor.on_start recorded job_entrypoint first and records with
        # setdefault, so without an explicit replacement the LLM-call marker would
        # keep landing on a span that has already ended.
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        session = _FakeAgentSession(call_harness.livekit_tracer)
        with trace.use_span(job, end_on_exit=False):
            asyncio.run(wrap_start(session.start, session, (), {"room": None}))

        call_span = getattr(session, CALL_SPAN_FIELD)
        trace_id = call_span.get_span_context().trace_id
        assert RootSpanProcessor.get_root_span_by_trace_id(trace_id) is call_span

        asyncio.run(wrap_aclose(session.aclose_impl, session, (), {}))
        job.end()

    def test_call_span_carries_the_workflow_entity_marker(self, call_harness: _CallHarness) -> None:
        _run_call(call_harness)

        attributes = call_harness.attributes(CALL_SPAN_NAME)
        assert attributes[NETRA_ENTITY_TYPE] == "workflow"
        assert attributes[NETRA_SPAN_TYPE] == SpanType.SPAN.value

    def test_call_span_carries_the_session_id(self, call_harness: _CallHarness, fake_livekit_agents: None) -> None:
        # The root span was the one span missing netra.session_id, because the id is
        # only resolvable once start() is called and job_entrypoint predates it.
        _run_call(call_harness, room=_FakeRoom("console-abc123"))

        assert call_harness.attributes(CALL_SPAN_NAME)["netra.session_id"] == "console-abc123"


class TestCallSpanLeavesOtherTracesAlone:
    """The re-rooting only fires on a live, livekit-scoped ``job_entrypoint``."""

    def test_call_span_is_a_natural_root_with_no_ambient_span(self, call_harness: _CallHarness) -> None:
        _run_call(call_harness)

        call = call_harness.finished(CALL_SPAN_NAME)
        session_span = call_harness.finished(AGENT_SESSION_SPAN_NAME)
        assert call.parent is None
        assert session_span.parent is not None
        assert session_span.parent.span_id == call.context.span_id
        assert REROOTED_ATTRIBUTE not in (call.attributes or {})

    def test_a_user_span_stays_the_root(self, call_harness: _CallHarness) -> None:
        # A netra decorator's or a user's own span owns its trace; hijacking it
        # would move their root under ours.
        user_span = call_harness.provider.get_tracer("my.app").start_span("checkout")
        _run_call(call_harness, parent=user_span)
        user_span.end()

        call = call_harness.finished(CALL_SPAN_NAME)
        user = call_harness.finished("checkout")
        assert user.parent is None
        assert call.parent is not None
        assert call.parent.span_id == user.context.span_id

    def test_a_job_entrypoint_from_another_scope_is_not_re_rooted(self, call_harness: _CallHarness) -> None:
        # Name alone must not be enough: any library may name a span job_entrypoint.
        impostor = call_harness.provider.get_tracer("some.other.sdk").start_span(JOB_ENTRYPOINT_SPAN_NAME)
        _run_call(call_harness, parent=impostor)
        impostor.end()

        call = call_harness.finished(CALL_SPAN_NAME)
        assert call.parent is not None
        assert call.parent.span_id == call_harness.finished(JOB_ENTRYPOINT_SPAN_NAME).context.span_id

    def test_a_second_session_in_one_job_does_not_re_root_again(self, call_harness: _CallHarness) -> None:
        job = call_harness.livekit_tracer.start_span(JOB_ENTRYPOINT_SPAN_NAME)
        first = _run_call(call_harness, parent=job)
        second = _run_call(call_harness, parent=job)
        job.end()

        first_call = getattr(first, CALL_SPAN_FIELD)
        second_call = getattr(second, CALL_SPAN_FIELD)
        entrypoint = call_harness.finished(JOB_ENTRYPOINT_SPAN_NAME)

        # job_entrypoint stays under the first call span; the second is an ordinary
        # child rather than a competing root.
        assert entrypoint.parent is not None
        assert entrypoint.parent.span_id == first_call.get_span_context().span_id
        assert second_call.parent is not None
        assert second_call.parent.span_id == entrypoint.context.span_id


class TestCallSpanLifecycle:
    """The call span is closed exactly once, on every path that can close it."""

    def test_agent_session_ending_closes_the_call_span(self, call_harness: _CallHarness) -> None:
        # The primary path: no method wrap involved, so it survives a LiveKit rename
        # of _aclose_impl.
        session = _FakeAgentSession(call_harness.livekit_tracer)
        asyncio.run(wrap_start(session.start, session, (), {"room": None}))
        assert call_harness.finished_count(CALL_SPAN_NAME) == 0, "precondition: still open"

        asyncio.run(session.aclose_impl())

        assert call_harness.finished_count(CALL_SPAN_NAME) == 1

    def test_call_span_is_ended_once_when_both_paths_run(self, call_harness: _CallHarness) -> None:
        _run_call(call_harness)

        # wrap_aclose runs after _aclose_impl, which already ended agent_session and
        # so already triggered the processor path.
        assert call_harness.finished_count(CALL_SPAN_NAME) == 1

    def test_wrap_aclose_closes_the_call_span_without_the_processor(self) -> None:
        # A provider that is not an SDK TracerProvider never gets the LiveKit
        # processors registered, leaving wrap_aclose as the only end path.
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer(LIVEKIT_SCOPE_NAME)

        session = _FakeAgentSession(tracer)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name, *a, **k))
            call_spans.pop_all()
            asyncio.run(wrap_start(session.start, session, (), {"room": None}))
            asyncio.run(wrap_aclose(session.aclose_impl, session, (), {}))

        assert len([span for span in exporter.get_finished_spans() if span.name == CALL_SPAN_NAME]) == 1

    def test_close_that_raises_still_closes_the_call_span(self, call_harness: _CallHarness) -> None:
        session = _FakeAgentSession(call_harness.livekit_tracer)
        asyncio.run(wrap_start(session.start, session, (), {"room": None}))

        async def failing_close(**kwargs: Any) -> None:
            raise RuntimeError("teardown blew up")

        with pytest.raises(RuntimeError, match="teardown blew up"):
            asyncio.run(wrap_aclose(failing_close, session, (), {}))

        assert call_harness.finished_count(CALL_SPAN_NAME) == 1

    def test_a_failing_start_ends_the_call_span_with_an_error(self, call_harness: _CallHarness) -> None:
        session = _FakeAgentSession(call_harness.livekit_tracer)

        async def failing_start(**kwargs: Any) -> None:
            raise RuntimeError("livekit blew up")

        with pytest.raises(RuntimeError, match="livekit blew up"):
            asyncio.run(wrap_start(failing_start, session, (), {"room": None}))

        call = call_harness.finished(CALL_SPAN_NAME)
        assert call.status.status_code is trace.StatusCode.ERROR
        assert [event.name for event in call.events] == ["exception"]

    def test_shutdown_closes_a_call_whose_session_never_closed(self, call_harness: _CallHarness) -> None:
        # The process-exit backstop: an unended span is never exported at all, so
        # without this the whole call loses its root.
        session = _FakeAgentSession(call_harness.livekit_tracer)
        asyncio.run(wrap_start(session.start, session, (), {"room": None}))
        assert call_harness.finished_count(CALL_SPAN_NAME) == 0, "precondition: still open"

        end_all_call_spans()

        assert call_harness.finished_count(CALL_SPAN_NAME) == 1


class TestCallSpanStatus:
    """The call span is the trace root, so it is where trace-level health is read.

    Left to itself an ended span is ``UNSET``, which would make a call that died
    mid-way indistinguishable from a clean one without walking the children.
    """

    def test_a_clean_call_leaves_the_call_span_unset(self, call_harness: _CallHarness) -> None:
        _run_call(call_harness)

        assert call_harness.finished(CALL_SPAN_NAME).status.status_code is trace.StatusCode.UNSET

    def test_a_session_that_ends_in_error_ends_the_call_span_in_error(self, call_harness: _CallHarness) -> None:
        session = _FailedAgentSession(call_harness.livekit_tracer)

        async def call() -> None:
            await wrap_start(session.start, session, (), {"room": None})
            await wrap_aclose(session.aclose_impl, session, (), {})

        asyncio.run(call())

        status = call_harness.finished(CALL_SPAN_NAME).status
        assert status.status_code is trace.StatusCode.ERROR
        assert status.description == _FailedAgentSession.CLOSE_ERROR

    def test_shutdown_marks_the_calls_it_closes_as_failed(self, call_harness: _CallHarness) -> None:
        # Reaching the backstop at all means the process is exiting mid-call, which
        # is not a clean end and must not be exported as one.
        session = _FakeAgentSession(call_harness.livekit_tracer)
        asyncio.run(wrap_start(session.start, session, (), {"room": None}))

        end_all_call_spans()

        status = call_harness.finished(CALL_SPAN_NAME).status
        assert status.status_code is trace.StatusCode.ERROR
        assert "never closed" in (status.description or "")


class TestOpenCallSpansAreBounded:
    """Every registry entry pins a live ``Span``, and only a session close frees one.

    A session abandoned without ever closing therefore leaks one span object per
    call for the process lifetime — invisible on a per-job worker process, but
    unbounded on a thread-executor worker running many jobs in one process.
    """

    @staticmethod
    def _start_abandoned_calls(harness: _CallHarness, count: int) -> None:
        """Start *count* calls and never close any of them."""
        for _ in range(count):
            session = _FakeAgentSession(harness.livekit_tracer)
            asyncio.run(wrap_start(session.start, session, (), {"room": None}))

    def test_the_registry_stops_growing_at_the_cap(self, call_harness: _CallHarness) -> None:
        self._start_abandoned_calls(call_harness, _MAX_OPEN_CALL_SPANS + 5)

        assert len(call_spans.pop_all()) == _MAX_OPEN_CALL_SPANS

    def test_an_evicted_call_is_exported_rather_than_silently_dropped(self, call_harness: _CallHarness) -> None:
        # Eviction has to *end* the span: an unended span never reaches the
        # exporter, so releasing the reference alone would lose the call outright.
        overflow = 3
        self._start_abandoned_calls(call_harness, _MAX_OPEN_CALL_SPANS + overflow)

        evicted = [span for span in call_harness.exporter.get_finished_spans() if span.name == CALL_SPAN_NAME]
        assert len(evicted) == overflow
        assert all(span.status.status_code is trace.StatusCode.ERROR for span in evicted)
        assert all("evicted" in (span.status.description or "") for span in evicted)

    def test_the_oldest_call_is_the_one_evicted(self, call_harness: _CallHarness) -> None:
        first = _FakeAgentSession(call_harness.livekit_tracer)
        asyncio.run(wrap_start(first.start, first, (), {"room": None}))
        first_call_span = getattr(first, CALL_SPAN_FIELD)

        self._start_abandoned_calls(call_harness, _MAX_OPEN_CALL_SPANS)

        evicted = [span for span in call_harness.exporter.get_finished_spans() if span.name == CALL_SPAN_NAME]
        assert len(evicted) == 1
        assert evicted[0].context.span_id == first_call_span.get_span_context().span_id

    def test_a_call_that_closes_normally_still_ends_normally_after_eviction(self, call_harness: _CallHarness) -> None:
        # Eviction ends spans outside the registry lock because the processor chain
        # reaches back into the registry; a survivor closing through that same
        # chain afterwards proves the registry is still usable.
        self._start_abandoned_calls(call_harness, _MAX_OPEN_CALL_SPANS + 1)

        _run_call(call_harness)

        clean = [
            span
            for span in call_harness.exporter.get_finished_spans()
            if span.name == CALL_SPAN_NAME and span.status.status_code is trace.StatusCode.UNSET
        ]
        assert len(clean) == 1


class TestSessionStartHook:
    def test_start_result_is_returned_untouched(self, call_harness: _CallHarness) -> None:
        session = _FakeAgentSession(call_harness.livekit_tracer)

        async def fake_start(**kwargs: Any) -> str:
            return "started"

        result = asyncio.run(wrap_start(fake_start, session, (), {"room": None}))

        assert result == "started"

    def test_start_exceptions_propagate_unchanged(self, call_harness: _CallHarness) -> None:
        session = _FakeAgentSession(call_harness.livekit_tracer)

        async def failing_start(**kwargs: Any) -> None:
            raise RuntimeError("livekit blew up")

        with pytest.raises(RuntimeError, match="livekit blew up"):
            asyncio.run(wrap_start(failing_start, session, (), {}))

    def test_start_result_is_returned_when_the_call_span_cannot_be_stored(self) -> None:
        # An instance with no __dict__ cannot hold the call span. Netra must still
        # not change what the user's start() returns.
        async def fake_start(**kwargs: Any) -> str:
            return "started"

        assert asyncio.run(wrap_start(fake_start, object(), (), {"room": None})) == "started"


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
