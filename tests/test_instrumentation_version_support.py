"""Instrumentors keep working across SDK layout changes, and say so when they cannot.

Covers H-19: an instrumentor that silently stops applying on a new major
version of its library leaves customers with no spans and no explanation.
The SDKs themselves are not installed in the test environment, so their
layouts are reproduced as fake packages under ``tmp_path``.
"""

import importlib
import importlib.util
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Collection, Dict, Iterator

import pytest
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.instrumentation.instruments import InstrumentSet
from netra.instrumentation.wiring import activation
from netra.instrumentation.wiring.activation import Activation, run_activation
from netra.instrumentation.wiring.registry import InstrumentorSpec
from netra.instrumentation.wiring.triggers import INSTRUMENT_TRIGGERS


@pytest.fixture(autouse=True)
def fresh_inactive_warnings() -> Iterator[None]:
    """Each test sees the once-per-process warning state from scratch."""
    activation._instrumentations_warned_inactive.clear()
    yield
    activation._instrumentations_warned_inactive.clear()


def _inactive_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "instrumentation is inactive" in str(w.message)
    ]


# ---------------------------------------------------------------------------
# One-time "instrumentation is inactive" warning
# ---------------------------------------------------------------------------


class _UnsupportedVersionInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Declines to apply: no installed pytest satisfies the range."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("pytest < 1",)

    def _instrument(self, **kwargs: Any) -> None:
        raise AssertionError("instrument() must not apply on a dependency conflict")

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class TestInactiveInstrumentationWarning:
    def test_an_unsupported_library_version_is_reported_with_both_versions(self) -> None:
        spec = InstrumentorSpec(("pytest",), __name__, "_UnsupportedVersionInstrumentor")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            activation._apply_instrumentor(spec, "FAKELIB")

        messages = _inactive_warnings(caught)
        assert len(messages) == 1
        assert "FAKELIB" in messages[0]
        assert "pytest < 1" in messages[0]
        assert f"pytest {pytest.__version__}" in messages[0]

    def test_the_warning_fires_once_however_often_activation_is_attempted(self) -> None:
        spec = InstrumentorSpec(("pytest",), __name__, "_UnsupportedVersionInstrumentor")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                activation._apply_instrumentor(spec, "FAKELIB")

        assert len(_inactive_warnings(caught)) == 1

    def test_an_activation_that_raises_is_reported_once_and_contained(self) -> None:
        def explode() -> None:
            raise ImportError("cannot import name 'AssistantMessage'")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_activation(Activation("FAKELIB", explode))
            run_activation(Activation("FAKELIB", explode))

        messages = _inactive_warnings(caught)
        assert len(messages) == 1
        assert "ImportError" in messages[0] and "AssistantMessage" in messages[0]

    def test_a_supported_cohere_major_is_not_rejected(self) -> None:
        pytest.importorskip("opentelemetry.instrumentation.cohere")
        from netra.instrumentation.libraries.cohere import _instruments

        # cohere 6.x and 7.x keep the ClientV2 / AsyncClientV2 surface wrapped here.
        assert _instruments == ("cohere >=4.2.7, <8",)


# ---------------------------------------------------------------------------
# Trigger modules
# ---------------------------------------------------------------------------


def _is_namespace_package(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    return spec is not None and spec.origin is None and spec.submodule_search_locations is not None


def _is_findable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _is_regular_module(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


class TestTriggerModules:
    @pytest.mark.parametrize(
        "instrument",
        sorted(INSTRUMENT_TRIGGERS, key=lambda instrument: instrument.name),
        ids=lambda instrument: instrument.name,
    )
    def test_an_installed_library_has_a_trigger_that_is_a_real_module(self, instrument: InstrumentSet) -> None:
        # A post-import hook on a namespace package never fires, so a library
        # whose only trigger is one would never be instrumented.
        triggers = INSTRUMENT_TRIGGERS[instrument]
        if not any(_is_findable(trigger) for trigger in triggers):
            pytest.skip(f"{instrument.name}: library not installed")

        assert any(
            _is_regular_module(trigger) for trigger in triggers
        ), f"{instrument.name}: every trigger in {triggers} is a namespace package or missing"


# ---------------------------------------------------------------------------
# Fake SDK layouts
# ---------------------------------------------------------------------------


def _write_package(root: Path, files: Dict[str, str]) -> None:
    for relative_path, source in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))


@pytest.fixture()
def fake_sdk_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A directory on ``sys.path`` for fake SDKs.

    Any real copy already imported is hidden for the test and restored after,
    and the fakes are unloaded.
    """
    faked = {"mistralai", "cartesia"}
    for name in [name for name in sys.modules if name.split(".")[0] in faked]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    for name in [name for name in sys.modules if name.split(".")[0] in faked]:
        del sys.modules[name]


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(span_exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


_MISTRAL_V3_LAYOUT = {
    # No mistralai/__init__.py: SDK 3.x makes it a namespace package.
    "mistralai/client/__init__.py": "",
    "mistralai/client/chat.py": """
        from types import SimpleNamespace

        class Chat:
            def complete(self, **kwargs):
                return SimpleNamespace(
                    id="resp-1",
                    model=kwargs["model"],
                    choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(role="assistant", content="hello"))],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
                )

            async def complete_async(self, **kwargs):
                return self.complete(**kwargs)

            def stream(self, **kwargs):
                def events():
                    for text, finish_reason in (("hel", None), ("lo", "stop")):
                        delta = SimpleNamespace(role="assistant", content=text)
                        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
                        yield SimpleNamespace(data=SimpleNamespace(id="s-1", model=kwargs["model"], choices=[choice], usage=None))
                return events()

            async def stream_async(self, **kwargs):
                async def events():
                    for event in self.stream(**kwargs):
                        yield event
                return events()
    """,
    "mistralai/client/embeddings.py": """
        class Embeddings:
            def create(self, **kwargs):
                return {"id": "emb-1"}
    """,
}


class TestMistralLayouts:
    def test_the_instrumentor_imports_without_the_sdk_types(self, fake_sdk_path: Path) -> None:
        # The 1.x type imports at module scope raised ImportError on 3.x, so the
        # instrumentor itself could not be imported.
        _write_package(fake_sdk_path, _MISTRAL_V3_LAYOUT)
        module = importlib.reload(importlib.import_module("netra.instrumentation.libraries.mistralai"))
        assert module.MistralAiInstrumentor is not None

    def test_the_3x_namespace_layout_has_a_trigger_that_fires(self, fake_sdk_path: Path) -> None:
        _write_package(fake_sdk_path, _MISTRAL_V3_LAYOUT)

        assert _is_namespace_package("mistralai")
        assert any(_is_regular_module(trigger) for trigger in INSTRUMENT_TRIGGERS[InstrumentSet.MISTRALAI])

    def test_the_3x_client_is_wrapped_and_traced(self, fake_sdk_path: Path, tracer_provider, span_exporter) -> None:
        _write_package(fake_sdk_path, _MISTRAL_V3_LAYOUT)
        from netra.instrumentation.libraries.mistralai import (
            WRAPPED_METHODS_V3,
            MistralAiInstrumentor,
            _select_wrapped_methods,
        )

        assert _select_wrapped_methods() == WRAPPED_METHODS_V3

        instrumentor = MistralAiInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        try:
            from mistralai.client.chat import Chat

            response = Chat().complete(model="mistral-small", messages=[{"role": "user", "content": "hi"}])
            streamed = [event.data.choices[0].delta.content for event in Chat().stream(model="mistral-small")]
        finally:
            instrumentor.uninstrument()

        assert response.choices[0].message.content == "hello"
        assert streamed == ["hel", "lo"]
        spans = {span.name: dict(span.attributes or {}) for span in span_exporter.get_finished_spans()}
        assert spans["mistralai.chat.complete"]["gen_ai.response.id"] == "resp-1"
        assert spans["mistralai.chat.complete"]["gen_ai.completion.0.content"] == "hello"
        assert spans["mistralai.chat.stream"]["gen_ai.completion.0.content"] == "hello"
        assert spans["mistralai.chat.stream"]["gen_ai.completion.0.finish_reason"] == "stop"
        # uninstrument really unwrapped
        assert not hasattr(Chat.complete, "__wrapped__")

    def test_an_unrecognised_layout_warns_once_instead_of_failing_silently(self, fake_sdk_path: Path) -> None:
        _write_package(fake_sdk_path, {"mistralai/__init__.py": ""})
        from netra.instrumentation.libraries.mistralai import MistralAiInstrumentor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instrumentor = MistralAiInstrumentor()
            instrumentor.instrument(skip_dep_check=True)
            instrumentor.uninstrument()
            instrumentor.instrument(skip_dep_check=True)
            instrumentor.uninstrument()

        messages = _inactive_warnings(caught)
        assert len(messages) == 1
        assert "MISTRALAI" in messages[0]

    def test_a_chunk_the_accumulator_cannot_parse_does_not_reach_the_caller(self) -> None:
        from netra.instrumentation.libraries.mistralai import _accumulate_chunk, _new_accumulated_response

        class Hostile:
            @property
            def data(self) -> Any:
                raise RuntimeError("unexpected chunk shape")

        accumulated = _new_accumulated_response()
        _accumulate_chunk(accumulated, Hostile())  # must not raise
        assert accumulated.choices == []


_CARTESIA_V4_LAYOUT = {
    "cartesia/__init__.py": "",
    "cartesia/resources/__init__.py": "",
    "cartesia/resources/tts.py": """
        class TTSResource:
            def generate(self, **kwargs):
                return b"audio"

            def generate_sse(self, **kwargs):
                return iter(())

            sse = generate_sse

            def bytes(self, **kwargs):
                return iter([self.generate(**kwargs)])


        class AsyncTTSResource:
            async def generate(self, **kwargs):
                return b"audio"

            async def generate_sse(self, **kwargs):
                return None

            sse = generate_sse
    """,
}


class TestCartesiaLayouts:
    def test_the_4x_resources_are_wrapped_and_traced(self, fake_sdk_path: Path, tracer_provider, span_exporter) -> None:
        import asyncio

        _write_package(fake_sdk_path, _CARTESIA_V4_LAYOUT)
        from netra.instrumentation.libraries.cartesia import NetraCartesiaInstrumentor

        instrumentor = NetraCartesiaInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        try:
            from cartesia.resources.tts import AsyncTTSResource, TTSResource

            kwargs = {"model_id": "sonic-3", "transcript": "Hi there"}
            TTSResource().generate(**kwargs)
            # bytes delegates to generate: one span, not two
            list(TTSResource().bytes(**kwargs))
            asyncio.run(AsyncTTSResource().generate(**kwargs))
        finally:
            instrumentor.uninstrument()

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == ["cartesia.tts"] * 3
        assert all(span.attributes["gen_ai.request.model"] == "sonic-3" for span in spans)
        assert not hasattr(TTSResource.generate, "__wrapped__")

    def test_no_known_layout_warns_once(self, fake_sdk_path: Path) -> None:
        _write_package(fake_sdk_path, {"cartesia/__init__.py": ""})
        from netra.instrumentation.libraries.cartesia import NetraCartesiaInstrumentor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instrumentor = NetraCartesiaInstrumentor()
            instrumentor.instrument(skip_dep_check=True)
            instrumentor.uninstrument()
            instrumentor.instrument(skip_dep_check=True)
            instrumentor.uninstrument()

        messages = _inactive_warnings(caught)
        assert len(messages) == 1
        assert "CARTESIA" in messages[0]

    def test_a_request_attribute_failure_does_not_break_the_call(self) -> None:
        from netra.instrumentation.libraries.cartesia.utils import set_request_attributes

        class BrokenSpan:
            def is_recording(self) -> bool:
                raise RuntimeError("span exploded")

        set_request_attributes(BrokenSpan(), {"transcript": "hi"})  # must not raise

    def test_a_binary_response_text_method_is_not_recorded_as_a_transcript(self) -> None:
        from netra.instrumentation.libraries.cartesia.utils import set_response_attributes

        class BinaryResponse:
            def text(self) -> str:
                return "decoded"

        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("t").start_as_current_span("s") as span:
            set_response_attributes(span, BinaryResponse())

        attributes = dict(exporter.get_finished_spans()[0].attributes or {})
        assert "gen_ai.response.text" not in attributes
        assert "gen_ai.prompt.1.content" not in attributes

    def test_a_non_streaming_websocket_send_returns_its_single_output(self) -> None:
        from netra.instrumentation.libraries.cartesia.wrappers import TtsWebSocketProxy

        class FakeConnection:
            def send(self, *, stream: bool = True, **kwargs: Any) -> Any:
                return iter([b"a", b"b"]) if stream else b"ab"

            def context(self) -> str:
                return "ctx"

            def close(self) -> None:
                pass

        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("t")

        single = TtsWebSocketProxy(FakeConnection(), tracer.start_span("ws"), 0.0)
        assert single.send(transcript="hi", stream=False) == b"ab"
        assert single.context() == "ctx"

        streamed = TtsWebSocketProxy(FakeConnection(), tracer.start_span("ws"), 0.0)
        assert list(streamed.send(transcript="hi")) == [b"a", b"b"]

        assert len(exporter.get_finished_spans()) == 2
