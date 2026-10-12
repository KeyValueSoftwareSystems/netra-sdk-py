"""
Unit tests for NetraGoogleGenAiInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

import asyncio
import json
import types
from types import SimpleNamespace
from typing import Collection
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import StatusCode
from wrapt import ObjectProxy

import netra.instrumentation.libraries.google_genai as genai_instrumentation
from netra.instrumentation.libraries.google_genai import NetraGoogleGenAiInstrumentor
from netra.instrumentation.libraries.google_genai.utils import (
    set_decision_request_attributes,
    set_decision_response_attributes,
)
from netra.instrumentation.libraries.google_genai.wrappers import adecisions_wrapper, decisions_wrapper


@pytest.fixture(autouse=True)
def isolated_decisions_hook(monkeypatch):
    """Keep the process-wide decisions post-import hook state from leaking between tests."""
    monkeypatch.setattr(genai_instrumentation, "_decisions_hook_registered", False)
    monkeypatch.setattr(genai_instrumentation, "_decisions_tracer", None)
    hook = Mock()
    monkeypatch.setattr(genai_instrumentation, "register_post_import_hook", hook)
    return hook


class TestNetraGoogleGenAiInstrumentor:
    """Test NetraGoogleGenAiInstrumentor core functionality."""

    def test_initialization(self):
        """Test NetraGoogleGenAiInstrumentor initialization."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        dependencies = instrumentor.instrumentation_dependencies()

        assert isinstance(dependencies, Collection)
        assert "google-genai >= 0.1.0" in dependencies

    @patch("netra.instrumentation.libraries.google_genai.get_tracer")
    @patch("netra.instrumentation.libraries.google_genai.wrap_function_wrapper")
    def test_instrument_with_default_parameters(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        instrumentor = NetraGoogleGenAiInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.libraries.google_genai.get_tracer")
    @patch("netra.instrumentation.libraries.google_genai.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        instrumentor = NetraGoogleGenAiInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.google_genai", mock_get_tracer.call_args[0][1], mock_tracer_provider
        )
        assert mock_wrap_function.call_count == 8

    @patch("netra.instrumentation.libraries.google_genai.unwrap")
    def test_uninstrument(self, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        instrumentor = NetraGoogleGenAiInstrumentor()

        instrumentor._uninstrument()

        assert mock_unwrap.call_count == 8


# ---------------------------------------------------------------------------
# Gemini Decisions API (client.decisions.create)
# ---------------------------------------------------------------------------

DECISION_QUESTIONS = {
    "is_refund": {"type": "noul", "instructions": "Is the user asking for a refund?"},
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
    "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "annoyed", "angry"]},
}
DECISION_ANSWERS = {
    "is_refund": {"type": "noul", "noul": 0.93, "confidence": 0.93},
    "route": {"type": "choice", "choice": "billing", "confidence": 0.8, "probabilities": {"billing": 0.8, "tech": 0.2}},
}
DECISION_USAGE = {
    "total_input_tokens": 120,
    "total_output_tokens": 3,
    "total_thought_tokens": 2,
    "total_tokens": 125,
    "total_cached_tokens": 10,
}


def _keyed(mapping):
    return [{"id": key, **value} for key, value in mapping.items()]


def _recording_span():
    """Return a mock recording span plus the dict its set_attribute calls write into."""
    span = Mock()
    span.is_recording.return_value = True
    captured: dict = {}
    span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
    return span, captured


def _decision_input(input_json):
    """Split a decision ``input`` attribute into its input entries and the questions its last entry holds."""
    *entries, questions_entry = json.loads(input_json)
    assert questions_entry["role"] == "user"
    return entries, json.loads(questions_entry["content"])


class _Model:
    """Stand-in for a google-genai pydantic model: exposes ``model_dump`` and attribute access."""

    def __init__(self, **fields):
        self._fields = fields
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, **kwargs):
        return {key: value.model_dump() if isinstance(value, _Model) else value for key, value in self._fields.items()}


def _response(answers=None, usage=None, prompt_feedback=None):
    return types.SimpleNamespace(
        answers={key: _Model(**value) for key, value in (answers or {}).items()} if answers is not None else None,
        usage=_Model(**usage) if usage else None,
        prompt_feedback=_Model(**prompt_feedback) if prompt_feedback else None,
    )


@pytest.fixture
def content_tracing_on(monkeypatch):
    """Decision content is gated on TRACELOOP_TRACE_CONTENT, which other tests' Config() may have set to false."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")


@pytest.fixture
def default_attribute_max_len(monkeypatch):
    """Decision capture budgets against the active config's limit, which other tests' Config() may have shrunk."""
    monkeypatch.setattr("netra.config._active_config", None)


@pytest.fixture
def attribute_max_len(monkeypatch):
    """Shrink the attribute length limit decision capture budgets against."""

    def set_limit(max_len: int) -> None:
        monkeypatch.setattr("netra.instrumentation.libraries.google_genai.utils.get_attribute_max_len", lambda: max_len)

    return set_limit


class TestDecisionsInstrumentor:
    """Lazy registration of the Decisions API wrappers on NetraGoogleGenAiInstrumentor."""

    @staticmethod
    def _fake_decisions_module():
        class Decisions:
            def create(self, **kwargs):
                return "sync"

        class AsyncDecisions:
            async def create(self, **kwargs):
                return "async"

        return types.SimpleNamespace(Decisions=Decisions, AsyncDecisions=AsyncDecisions)

    @patch("netra.instrumentation.libraries.google_genai.get_tracer")
    @patch("netra.instrumentation.libraries.google_genai.wrap_function_wrapper")
    def test_instrument_defers_decisions_to_a_post_import_hook(
        self, mock_wrap_function, mock_get_tracer, isolated_decisions_hook
    ):
        NetraGoogleGenAiInstrumentor()._instrument()

        isolated_decisions_hook.assert_called_once_with(
            genai_instrumentation._wrap_decisions, "google.genai._gaos.decisions"
        )
        wrapped_targets = [c.args[1] for c in mock_wrap_function.call_args_list]
        assert not any("ecision" in str(target) for target in wrapped_targets)

    @patch("netra.instrumentation.libraries.google_genai.get_tracer")
    @patch("netra.instrumentation.libraries.google_genai.wrap_function_wrapper")
    def test_hook_is_registered_once_across_reinstrumentation(
        self, mock_wrap_function, mock_get_tracer, isolated_decisions_hook
    ):
        instrumentor = NetraGoogleGenAiInstrumentor()
        instrumentor._instrument()
        instrumentor._uninstrument()
        instrumentor._instrument()

        assert isolated_decisions_hook.call_count == 1

    def test_hook_wraps_sync_and_async_create(self, monkeypatch):
        module = self._fake_decisions_module()
        monkeypatch.setattr(genai_instrumentation, "_decisions_tracer", Mock())

        genai_instrumentation._wrap_decisions(module)

        assert isinstance(module.Decisions.__dict__["create"], ObjectProxy)
        assert isinstance(module.AsyncDecisions.__dict__["create"], ObjectProxy)

    def test_hook_does_not_wrap_twice(self, monkeypatch):
        module = self._fake_decisions_module()
        monkeypatch.setattr(genai_instrumentation, "_decisions_tracer", Mock())

        genai_instrumentation._wrap_decisions(module)
        first = module.Decisions.__dict__["create"]
        genai_instrumentation._wrap_decisions(module)

        assert module.Decisions.__dict__["create"] is first
        assert not isinstance(first.__wrapped__, ObjectProxy)

    def test_hook_does_nothing_while_uninstrumented(self):
        module = self._fake_decisions_module()

        genai_instrumentation._wrap_decisions(module)

        assert not isinstance(module.Decisions.__dict__["create"], ObjectProxy)

    def test_uninstrument_unwraps_a_loaded_decisions_module(self, monkeypatch):
        module = self._fake_decisions_module()
        monkeypatch.setattr(genai_instrumentation, "_decisions_tracer", Mock())
        genai_instrumentation._wrap_decisions(module)
        monkeypatch.setitem(genai_instrumentation.sys.modules, "google.genai._gaos.decisions", module)

        genai_instrumentation._disable_decisions()

        assert not isinstance(module.Decisions.__dict__["create"], ObjectProxy)
        assert not isinstance(module.AsyncDecisions.__dict__["create"], ObjectProxy)
        assert genai_instrumentation._decisions_tracer is None

    @patch("netra.instrumentation.libraries.google_genai.get_tracer")
    @patch("netra.instrumentation.libraries.google_genai.wrap_function_wrapper")
    def test_reinstrument_rewraps_an_already_loaded_module(self, mock_wrap_function, mock_get_tracer, monkeypatch):
        module = self._fake_decisions_module()
        monkeypatch.setitem(genai_instrumentation.sys.modules, "google.genai._gaos.decisions", module)
        monkeypatch.setattr(genai_instrumentation, "_decisions_hook_registered", True)

        NetraGoogleGenAiInstrumentor()._instrument()

        wrapped_targets = [(c.args[0], c.args[1]) for c in mock_wrap_function.call_args_list]
        assert (module.Decisions, "create") in wrapped_targets
        assert (module.AsyncDecisions, "create") in wrapped_targets


@pytest.mark.usefixtures("content_tracing_on", "default_attribute_max_len")
class TestDecisionRequestAttributes:
    """set_decision_request_attributes."""

    @staticmethod
    def _capture(kwargs):
        span, captured = _recording_span()
        set_decision_request_attributes(span, kwargs)
        return captured

    def test_model_resource_prefix_is_stripped(self):
        attrs = self._capture({"model": "models/gemini-decision-1", "input": "hi", "questions": DECISION_QUESTIONS})

        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-decision-1"
        assert attrs[SpanAttributes.LLM_RESPONSE_MODEL] == "gemini-decision-1"

    def test_string_input_is_one_user_entry_followed_by_the_questions(self):
        attrs = self._capture({"model": "gemini-decision-1", "input": "Refund me!", "questions": DECISION_QUESTIONS})

        entries, questions = _decision_input(attrs["input"])
        assert entries == [{"role": "user", "content": "Refund me!"}]
        assert questions == _keyed(DECISION_QUESTIONS)

    def test_text_blocks_are_recorded_as_text_and_media_data_is_elided(self):
        blocks = [
            {"type": "text", "text": "Look at this receipt"},
            {"type": "image", "mime_type": "image/png", "data": "aGVsbG8=" * 1000},
            {"type": "document", "uri": "gs://bucket/receipt.pdf", "mime_type": "application/pdf"},
        ]

        attrs = self._capture({"model": "gemini-decision-1", "input": blocks, "questions": DECISION_QUESTIONS})

        entries, _ = _decision_input(attrs["input"])
        assert entries[0] == {"role": "user", "content": "Look at this receipt"}
        assert json.loads(entries[1]["content"]) == {"type": "image", "mime_type": "image/png", "data": "[image]"}
        assert json.loads(entries[2]["content"])["uri"] == "gs://bucket/receipt.pdf"
        assert "aGVsbG8" not in attrs["input"]

    def test_single_content_block_input(self):
        attrs = self._capture(
            {"model": "gemini-decision-1", "input": {"type": "text", "text": "solo"}, "questions": DECISION_QUESTIONS}
        )

        entries, _ = _decision_input(attrs["input"])
        assert entries == [{"role": "user", "content": "solo"}]

    def test_pydantic_questions_are_recorded(self):
        questions = {"q": _Model(type="noul", instructions="Spam?")}

        attrs = self._capture({"model": "gemini-decision-1", "input": "buy now", "questions": questions})

        _, recorded = _decision_input(attrs["input"])
        assert recorded == [{"id": "q", "type": "noul", "instructions": "Spam?"}]

    def test_one_shot_iterable_input_is_not_consumed(self):
        blocks = iter([{"type": "text", "text": "hi"}])

        attrs = self._capture({"model": "gemini-decision-1", "input": blocks, "questions": DECISION_QUESTIONS})

        assert list(blocks) == [{"type": "text", "text": "hi"}]
        entries, _ = _decision_input(attrs["input"])
        assert entries == []

    def test_nothing_is_recorded_when_content_tracing_is_off(self, monkeypatch):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        attrs = self._capture({"model": "gemini-decision-1", "input": "secret", "questions": DECISION_QUESTIONS})

        assert "input" not in attrs
        assert attrs[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-decision-1"

    def test_long_input_never_pushes_out_the_questions(self, attribute_max_len):
        attribute_max_len(1_000)

        attrs = self._capture({"model": "gemini-decision-1", "input": "x" * 50_000, "questions": DECISION_QUESTIONS})

        assert len(attrs["input"]) <= 1_000
        entries, questions = _decision_input(attrs["input"])
        assert questions == _keyed(DECISION_QUESTIONS)
        assert entries[0]["content"].startswith("xxx")

    def test_questions_past_half_the_limit_are_dropped_whole(self, attribute_max_len):
        attribute_max_len(600)
        questions = {f"q{i}": {"type": "noul", "instructions": f"Question {i}?"} for i in range(50)}

        attrs = self._capture({"model": "gemini-decision-1", "input": "hi", "questions": questions})

        assert len(attrs["input"]) <= 600
        _, recorded = _decision_input(attrs["input"])
        assert 0 < len(recorded) < len(questions)
        assert recorded == _keyed(questions)[: len(recorded)]

    def test_unserialisable_questions_never_fail_the_request(self):
        class Boom:
            def model_dump(self, **kwargs):
                raise RuntimeError("boom")

        attrs = self._capture({"model": "gemini-decision-1", "input": "hi", "questions": {"q": Boom()}})

        assert json.loads(attrs["input"]) == [{"role": "user", "content": "hi"}]


@pytest.mark.usefixtures("content_tracing_on", "default_attribute_max_len")
class TestDecisionResponseAttributes:
    """set_decision_response_attributes."""

    @staticmethod
    def _capture(response):
        span, captured = _recording_span()
        set_decision_response_attributes(span, response)
        return captured

    def test_answers_are_recorded_as_output_keyed_by_question_id(self):
        attrs = self._capture(_response(answers=DECISION_ANSWERS, usage=DECISION_USAGE))

        (output,) = json.loads(attrs["output"])
        assert output["role"] == "assistant"
        assert json.loads(output["content"]) == _keyed(DECISION_ANSWERS)
        assert not any(key.startswith(f"{SpanAttributes.LLM_COMPLETIONS}.") for key in attrs)

    def test_usage_is_recorded(self):
        attrs = self._capture(_response(answers=DECISION_ANSWERS, usage=DECISION_USAGE))

        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 120
        assert attrs[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 5
        assert attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 125
        assert attrs[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 10

    def test_block_reason_is_recorded_as_finish_reason(self):
        attrs = self._capture(_response(usage={"total_input_tokens": 5}, prompt_feedback={"block_reason": "safety"}))

        assert attrs[SpanAttributes.LLM_RESPONSE_FINISH_REASON] == "safety"
        assert "output" not in attrs

    def test_usage_is_recorded_but_answers_are_not_when_content_tracing_is_off(self, monkeypatch):
        monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

        attrs = self._capture(_response(answers=DECISION_ANSWERS, usage=DECISION_USAGE))

        assert "output" not in attrs
        assert attrs[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 120

    def test_raw_response_object_records_nothing(self):
        attrs = self._capture(Mock(spec=["status_code", "parse"]))

        assert attrs == {}

    def test_answers_past_the_attribute_limit_are_dropped_whole(self, attribute_max_len):
        attribute_max_len(600)
        answers = {f"q{i}": {"type": "noul", "noul": 0.5, "confidence": 0.5} for i in range(50)}

        attrs = self._capture(_response(answers=answers))

        assert len(attrs["output"]) <= 600
        (output,) = json.loads(attrs["output"])
        recorded = json.loads(output["content"])
        assert 0 < len(recorded) < len(answers)
        assert recorded == _keyed(answers)[: len(recorded)]


@pytest.mark.usefixtures("content_tracing_on", "default_attribute_max_len")
class TestDecisionWrappers:
    """decisions_wrapper / adecisions_wrapper produce one span per call."""

    @pytest.fixture
    def exporter(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return exporter, provider.get_tracer("test")

    KWARGS = {"model": "gemini-decision-1", "input": "hi", "questions": DECISION_QUESTIONS}

    def test_sync_call_records_one_ok_span(self, exporter):
        exp, tracer = exporter
        response = _response(answers=DECISION_ANSWERS, usage=DECISION_USAGE)

        result = decisions_wrapper(tracer)(lambda **kw: response, None, (), dict(self.KWARGS))

        assert result is response
        (span,) = exp.get_finished_spans()
        assert span.name == "genai.decision"
        assert span.status.status_code == StatusCode.OK
        assert span.attributes["gen_ai.system"] == "Gemini"
        assert span.attributes["llm.request.type"] == "decision"
        assert "output" in span.attributes

    def test_async_call_records_one_ok_span(self, exporter):
        exp, tracer = exporter
        response = _response(answers=DECISION_ANSWERS, usage=DECISION_USAGE)

        async def create(**kwargs):
            return response

        result = asyncio.run(adecisions_wrapper(tracer)(create, None, (), dict(self.KWARGS)))

        assert result is response
        (span,) = exp.get_finished_spans()
        assert span.name == "genai.decision"
        assert span.status.status_code == StatusCode.OK

    def test_api_error_is_reraised_and_marks_the_span(self, exporter):
        exp, tracer = exporter

        def create(**kwargs):
            raise ValueError("bad question")

        with pytest.raises(ValueError, match="bad question"):
            decisions_wrapper(tracer)(create, None, (), dict(self.KWARGS))

        (span,) = exp.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR


class TestRequestAttributes:
    """Test structured ``input`` built from GenAI request contents."""

    @staticmethod
    def _input(contents):
        from netra.instrumentation.libraries.google_genai.utils import set_request_attributes

        span = Mock()
        captured: dict = {}
        span.set_attribute.side_effect = lambda key, value: captured.__setitem__(key, value)
        set_request_attributes(span, (), {"contents": contents})
        return json.loads(captured["input"])

    def test_string_contents_become_a_user_message(self):
        assert self._input("hello") == [{"role": "user", "content": "hello"}]

    def test_content_without_a_role_defaults_to_user(self):
        content = SimpleNamespace(role=None, parts=[SimpleNamespace(text="hi")])

        assert self._input([content]) == [{"role": "user", "content": "hi"}]
