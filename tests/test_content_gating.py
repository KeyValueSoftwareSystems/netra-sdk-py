"""Tests that ``trace_content=False`` keeps prompts, function I/O and HTTP bodies off exported spans.

Two layers are covered:

* ``ContentGatingSpanProcessor`` strips content attributes from every span before
  export, whichever code wrote them.
* The decorators, the HTTP instrumentors and the OpenAI / Groq / LiteLLM /
  google-genai instrumentors skip building content at all when it is off.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Tuple
from unittest.mock import Mock, patch

import httpx
import pytest
import requests
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.config import Config
from netra.content_capture import CONTENT_OVERRIDE_CONTEXT_KEY, is_content_capture_enabled
from netra.processors import ContentGatingSpanProcessor, InstrumentationSpanProcessor, SpanIOProcessor
from netra.processors.content_gating_span_processor import is_content_attribute

SECRET = "ZEBRA_SECRET_42"


@pytest.fixture
def content_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")


@pytest.fixture
def content_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")


@pytest.fixture
def gated_pipeline() -> Iterator[Tuple[Any, InMemorySpanExporter]]:
    """A provider wired like ``Tracer`` with ``trace_content=False``: I/O mapping, then gating, then export."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    provider.add_span_processor(SpanIOProcessor())
    provider.add_span_processor(ContentGatingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


@pytest.fixture
def recording_span() -> Iterator[Any]:
    """A live, recording span with no gating processor, for testing source-level checks."""
    provider = TracerProvider()
    span = provider.get_tracer("test").start_span("s")
    yield span
    span.end()
    provider.shutdown()


def _exported_text(exporter: InMemorySpanExporter) -> str:
    parts = []
    for span in exporter.get_finished_spans():
        parts.append(json.dumps(dict(span.attributes or {}), default=str))
        for event in span.events:
            parts.append(json.dumps(dict(event.attributes or {}), default=str))
    return "\n".join(parts)


class TestIsContentAttribute:
    @pytest.mark.parametrize(
        "key",
        [
            "input",
            "output",
            "gen_ai.prompt.0.content",
            "gen_ai.completion.3.content",
            "gen_ai.completion.0.tool_call_id",
            "llm.prompts.0.content",
            "traceloop.entity.input",
            "gen_ai.input.messages",
            "lk.user_transcript",
            "lk.pii.chat_ctx",
            "lk.pii.function_tool.arguments",
            "pydantic_ai.user_prompt",
            "deepgram.agent.conversation.content",
            "netra.user.input",
            "netra.user.output",
            "conversation",
            "netra.prompt",
            "netra.negative_prompt",
            "retrieval.documents.0.content",
            "gen_ai.embeddings.2.text",
            "gen_ai.embeddings.2.vector",
        ],
    )
    def test_content_keys(self, key: str) -> None:
        assert is_content_attribute(key)

    @pytest.mark.parametrize(
        "key",
        [
            "gen_ai.request.model",
            "gen_ai.usage.prompt_tokens",
            "gen_ai.prompt.0.role",
            "gen_ai.completion.0.finish_reason",
            "gen_ai.request.input.mime_type",
            "http.url",
            "http.status_code",
            "netra.entity.type",
            "netra.session_id",
            "netra.prompt.id",
            "netra.prompt.name",
            "netra.prompt.version",
            "netra.prompt.label",
        ],
    )
    def test_metadata_keys(self, key: str) -> None:
        assert not is_content_attribute(key)


class TestContentGatingSpanProcessor:
    def test_strips_content_keeps_metadata(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        tracer, exporter = gated_pipeline
        with tracer.start_as_current_span("llm", attributes={"input": SECRET}) as span:
            span.set_attribute("gen_ai.request.model", "gpt-x")
            span.set_attribute("gen_ai.prompt.0.role", "user")
            span.set_attribute("gen_ai.prompt.0.content", SECRET)
            span.set_attribute("gen_ai.completion.0.content", SECRET)
            span.set_attribute("gen_ai.usage.prompt_tokens", 7)
            span.set_attribute("output", SECRET)

        (exported,) = exporter.get_finished_spans()
        assert SECRET not in _exported_text(exporter)
        assert exported.attributes["gen_ai.request.model"] == "gpt-x"
        assert exported.attributes["gen_ai.usage.prompt_tokens"] == 7
        assert exported.attributes["gen_ai.prompt.0.role"] == "user"
        assert "input" not in exported.attributes
        assert "output" not in exported.attributes

    def test_strips_content_from_events(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        tracer, exporter = gated_pipeline
        with tracer.start_as_current_span("llm") as span:
            span.add_event("gen_ai.content.prompt", {"gen_ai.prompt": SECRET, "kind": "prompt"})

        (exported,) = exporter.get_finished_spans()
        assert SECRET not in _exported_text(exporter)
        assert exported.events[0].name == "gen_ai.content.prompt"
        assert exported.events[0].attributes["kind"] == "prompt"

    def test_override_context_keeps_content(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        tracer, exporter = gated_pipeline
        token = context_api.attach(context_api.set_value(CONTENT_OVERRIDE_CONTEXT_KEY, True))
        try:
            with tracer.start_as_current_span("allowed") as span:
                span.set_attribute("output", SECRET)
        finally:
            context_api.detach(token)
        with tracer.start_as_current_span("gated") as span:
            span.set_attribute("output", SECRET)

        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert spans["allowed"].attributes["output"] == SECRET
        assert "output" not in spans["gated"].attributes

    def test_override_span_ids_released_on_end(self) -> None:
        processor = ContentGatingSpanProcessor()
        provider = TracerProvider()
        provider.add_span_processor(processor)
        token = context_api.attach(context_api.set_value(CONTENT_OVERRIDE_CONTEXT_KEY, True))
        try:
            with provider.get_tracer("test").start_as_current_span("allowed"):
                assert len(processor._override_span_ids) == 1
        finally:
            context_api.detach(token)
        assert not processor._override_span_ids
        provider.shutdown()

    def test_user_set_io_and_conversation_stripped(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        """Values from Netra.set_input/set_output/add_conversation and SpanWrapper.set_prompt never export."""
        tracer, exporter = gated_pipeline
        with tracer.start_as_current_span("agent") as span:
            span.set_attribute("netra.user.input", SECRET)  # Netra.set_input, promoted to input by SpanIOProcessor
            span.set_attribute("netra.user.output", SECRET)
            span.set_attribute("conversation", json.dumps([{"role": "user", "content": SECRET}]))
            span.set_attribute("netra.prompt", SECRET)
            span.set_attribute("netra.prompt.name", "support-bot")

        (exported,) = exporter.get_finished_spans()
        assert SECRET not in _exported_text(exporter)
        assert exported.attributes["netra.prompt.name"] == "support-bot"

    def test_dspy_retrieval_and_embedding_text_stripped(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        tracer, exporter = gated_pipeline
        with tracer.start_as_current_span("dspy") as span:
            span.set_attribute("retrieval.documents.0.content", SECRET)
            span.set_attribute("gen_ai.embeddings.0.text", SECRET)
            span.set_attribute("gen_ai.request.model", "embed-x")

        (exported,) = exporter.get_finished_spans()
        assert SECRET not in _exported_text(exporter)
        assert exported.attributes["gen_ai.request.model"] == "embed-x"

    def test_fails_closed_when_stripping_raises(self, gated_pipeline: Tuple[Any, InMemorySpanExporter]) -> None:
        tracer, exporter = gated_pipeline
        with patch("netra.processors.content_gating_span_processor._without_content", side_effect=RuntimeError("boom")):
            with tracer.start_as_current_span("llm") as span:
                span.set_attribute("output", SECRET)
                span.add_event("e", {"gen_ai.prompt": SECRET})

        (exported,) = exporter.get_finished_spans()
        assert not exported.attributes
        assert not exported.events


class TestTracerRegistration:
    def _installed(self, trace_content: Any) -> list:  # type: ignore[type-arg]
        from netra.tracer import Tracer

        cfg = Mock(spec=Config)
        cfg.app_name, cfg.environment, cfg.otlp_endpoint = "app", "test", None
        cfg.headers, cfg.resource_attributes, cfg.blocked_spans = {}, {}, []
        cfg.disable_batch, cfg.enable_scrubbing = True, False
        if trace_content is not None:
            cfg.trace_content = trace_content
        provider = Mock()
        provider._netra_processors_installed = False
        with (
            patch("netra.tracer.TracerProvider", return_value=provider),
            patch("netra.tracer.trace"),
            patch("netra.tracer.SimpleSpanProcessor"),
        ):
            Tracer(cfg)
        return [type(c.args[0]).__name__ for c in provider.add_span_processor.call_args_list]

    def test_installed_before_exporter_when_off(self) -> None:
        names = self._installed(False)
        assert "ContentGatingSpanProcessor" in names
        assert names.index("ContentGatingSpanProcessor") == len(names) - 2  # last before the exporter

    @pytest.mark.parametrize("trace_content", [True, None])
    def test_not_installed_when_on(self, trace_content: Any) -> None:
        assert "ContentGatingSpanProcessor" not in self._installed(trace_content)

    @pytest.mark.parametrize("trace_content, warned", [(False, True), (True, False)])
    def test_warns_when_reusing_provider_with_content_off(self, trace_content: bool, warned: bool) -> None:
        from netra.tracer import Tracer

        cfg = Mock(spec=Config)
        cfg.app_name, cfg.environment, cfg.otlp_endpoint = "app", "test", None
        cfg.headers, cfg.resource_attributes, cfg.blocked_spans = {}, {}, []
        cfg.disable_batch, cfg.enable_scrubbing, cfg.trace_content = True, False, trace_content
        existing = TracerProvider()
        with (
            patch("netra.tracer.trace.get_tracer_provider", return_value=existing),
            patch("netra.tracer.logger") as logger,
        ):
            Tracer(cfg)
        existing.shutdown()
        messages = " ".join(str(c.args[0]) for c in logger.warning.call_args_list)
        assert ("existing TracerProvider" in messages) is warned


class TestContentCaptureSetting:
    def test_off(self, content_off: None) -> None:
        assert not is_content_capture_enabled()

    def test_override(self, content_off: None) -> None:
        ctx = context_api.set_value(CONTENT_OVERRIDE_CONTEXT_KEY, True)
        assert is_content_capture_enabled(ctx)

    def test_default_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        assert is_content_capture_enabled()


class TestDecorators:
    def test_inputs_and_outputs_skipped(self, content_off: None, recording_span: Any) -> None:
        from netra.decorators import _add_output_attributes, _add_span_attributes

        def handle(patient_note: str) -> str:
            return patient_note

        _add_span_attributes(recording_span, handle, (SECRET,), {}, "workflow")
        _add_output_attributes(recording_span, SECRET)

        assert recording_span.attributes[f"{Config.LIBRARY_NAME}.entity.type"] == "workflow"
        assert "input" not in recording_span.attributes
        assert "output" not in recording_span.attributes

    def test_recorded_when_on(self, content_on: None, recording_span: Any) -> None:
        from netra.decorators import _add_output_attributes, _add_span_attributes

        def handle(patient_note: str) -> str:
            return patient_note

        _add_span_attributes(recording_span, handle, (SECRET,), {}, "workflow")
        _add_output_attributes(recording_span, SECRET)

        assert SECRET in recording_span.attributes["input"]
        assert SECRET in recording_span.attributes["output"]


class TestHttpBodies:
    def test_httpx(self, content_off: None, recording_span: Any) -> None:
        from netra.instrumentation.libraries.httpx import utils

        request = httpx.Request("POST", "https://api.example.com/v1", json={"note": SECRET})
        utils.set_span_input(recording_span, request)
        utils.set_span_output(recording_span, httpx.Response(200, json={"reply": SECRET}, request=request))
        assert "input" not in recording_span.attributes
        assert "output" not in recording_span.attributes

    def test_requests(self, content_off: None, recording_span: Any) -> None:
        from netra.instrumentation.libraries.requests import utils

        prepared = requests.Request("POST", "https://api.example.com/v1", json={"note": SECRET}).prepare()
        utils.set_span_input(recording_span, prepared)
        response = requests.Response()
        response.status_code, response._content = 200, json.dumps({"reply": SECRET}).encode()
        utils.set_span_output(recording_span, response)
        assert "input" not in recording_span.attributes
        assert "output" not in recording_span.attributes

    def test_fastapi(self, content_off: None, recording_span: Any) -> None:
        from netra.instrumentation.libraries.fastapi import utils

        scope = {"type": "http", "method": "POST", "path": "/chat", "headers": [], "query_string": b""}
        utils.set_span_input(recording_span, scope, json.dumps({"note": SECRET}).encode())
        utils.set_span_output(recording_span, 200, [], json.dumps({"reply": SECRET}).encode())
        assert "input" not in recording_span.attributes
        assert "output" not in recording_span.attributes

    def test_fastapi_middleware_does_not_buffer_bodies(self, content_off: None) -> None:
        import asyncio

        from netra.instrumentation.libraries.fastapi import middleware as mw

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": SECRET.encode()})

        async def receive() -> Dict[str, Any]:
            return {"type": "http.request", "body": SECRET.encode(), "more_body": False}

        async def send(message: Dict[str, Any]) -> None:
            return None

        scope = {"type": "http", "method": "POST", "path": "/chat", "headers": [], "query_string": b""}
        provider = TracerProvider()
        middleware = mw.NetraFastAPIMiddleware(app, provider.get_tracer("test"))
        with (
            patch.object(mw, "set_span_input") as set_input,
            patch.object(mw, "set_span_output") as set_output,
        ):
            asyncio.run(middleware(scope, receive, send))
        provider.shutdown()

        assert set_input.call_args.args[2] == b""
        assert set_output.call_args.args[3] == b""


def _chat_response() -> Dict[str, Any]:
    return {
        "model": "m-1",
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        "choices": [{"message": {"role": "assistant", "content": SECRET}, "finish_reason": "stop"}],
    }


class TestLlmInstrumentors:
    @pytest.mark.parametrize(
        "module",
        [
            "netra.instrumentation.libraries.openai.utils",
            "netra.instrumentation.libraries.groq.utils",
            "netra.instrumentation.libraries.litellm.utils",
        ],
    )
    def test_chat_prompt_and_completion_skipped(self, module: str, content_off: None, recording_span: Any) -> None:
        import importlib

        utils = importlib.import_module(module)
        kwargs = {"model": "m-1", "messages": [{"role": "user", "content": SECRET}]}
        utils.set_request_attributes(recording_span, kwargs, "chat")
        utils.set_response_attributes(recording_span, _chat_response())

        attrs = dict(recording_span.attributes)
        assert SECRET not in json.dumps(attrs, default=str)
        assert attrs["gen_ai.request.model"] == "m-1"
        assert attrs["gen_ai.response.model"] == "m-1"

    @pytest.mark.parametrize(
        "module",
        [
            "netra.instrumentation.libraries.openai.utils",
            "netra.instrumentation.libraries.groq.utils",
            "netra.instrumentation.libraries.litellm.utils",
        ],
    )
    def test_chat_prompt_recorded_when_on(self, module: str, content_on: None, recording_span: Any) -> None:
        import importlib

        utils = importlib.import_module(module)
        utils.set_request_attributes(
            recording_span, {"model": "m-1", "messages": [{"role": "user", "content": SECRET}]}, "chat"
        )
        utils.set_response_attributes(recording_span, _chat_response())
        assert recording_span.attributes["gen_ai.prompt.0.content"] == SECRET
        assert recording_span.attributes["gen_ai.completion.0.content"] == SECRET

    def test_openai_responses_api_skipped(self, content_off: None, recording_span: Any) -> None:
        from netra.instrumentation.libraries.openai import utils

        utils.set_request_attributes(
            recording_span, {"model": "m-1", "instructions": SECRET, "input": SECRET}, "response"
        )
        utils.set_response_attributes(recording_span, {"model": "m-1", "output_text": SECRET})
        assert SECRET not in json.dumps(dict(recording_span.attributes), default=str)

    def test_google_genai_skipped(self, content_off: None, recording_span: Any) -> None:
        from netra.instrumentation.libraries.google_genai import utils

        utils.set_request_attributes(recording_span, (), {"model": "gemini-x", "contents": SECRET})
        part = SimpleNamespace(text=SECRET)
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                total_token_count=7,
                candidates_token_count=4,
                thoughts_token_count=None,
                cached_content_token_count=None,
                prompt_token_count=3,
            ),
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        )
        utils.set_response_attributes(recording_span, response)
        utils.set_response_attributes(recording_span, SECRET)

        attrs = dict(recording_span.attributes)
        assert SECRET not in json.dumps(attrs, default=str)
        assert attrs["gen_ai.request.model"] == "gemini-x"
        assert attrs["gen_ai.usage.prompt_tokens"] == 3
