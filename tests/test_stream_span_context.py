"""Stream wrappers must not leave their span "current" between chunks.

The ElevenLabs, Agno, Deepgram, Cohere, Mistral and Google ADK stream wrappers used to attach the stream
span to the context and return while still attached, detaching only when the
stream was fully consumed. After ``for chunk in stream: break``, every later
span in that thread or task was parented under the dead stream span, and the
stream span itself was never exported. The span must now be current only while
a chunk is being pulled, and must be ended on ``close()`` or garbage collection.
"""

import asyncio
import gc
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Iterator, List

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from netra.instrumentation.libraries.agno.wrappers import (
    _async_stream_start,
    _sync_stream_start,
    model_aresponse_stream_capture_wrapper,
    model_response_stream_capture_wrapper,
)
from netra.instrumentation.libraries.cohere import _wrap as cohere_wrap
from netra.instrumentation.libraries.deepgram.wrappers import wrap_async_generator
from netra.instrumentation.libraries.elevenlabs.wrappers import async_generator_wrapper, stream_dialogue_wrapper
from netra.instrumentation.libraries.google_adk.wrappers import (
    base_agent_run_async_wrapper,
    run_and_handle_error_wrapper,
)
from netra.instrumentation.libraries.mistralai import _awrap as mistral_awrap
from netra.instrumentation.libraries.mistralai import _wrap as mistral_wrap

STREAM_SPAN = "stream"
INNER_SPAN = "provider.inner"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


class _FakeModel:
    id = STREAM_SPAN


def _sync_source(tracer: Tracer, n: int = 3) -> Callable[..., Iterator[str]]:
    """A provider stream that opens a child span while producing each chunk."""

    def produce(*_args: Any, **_kwargs: Any) -> Iterator[str]:
        for i in range(n):
            with tracer.start_as_current_span(INNER_SPAN):
                pass
            yield f"chunk-{i}"

    return produce


def _async_source(tracer: Tracer, n: int = 3) -> Callable[..., AsyncIterator[str]]:
    async def produce(*_args: Any, **_kwargs: Any) -> AsyncIterator[str]:
        for i in range(n):
            with tracer.start_as_current_span(INNER_SPAN):
                pass
            yield f"chunk-{i}"

    return produce


def _sync_streams(tracer: Tracer) -> List[Callable[[], Any]]:
    source = _sync_source(tracer)
    return [
        lambda: stream_dialogue_wrapper(tracer, STREAM_SPAN, "speech")(source, None, (), {}),
        lambda: _sync_stream_start(tracer, source, None, (), {}, STREAM_SPAN, "agent"),
        lambda: model_response_stream_capture_wrapper(tracer)(source, _FakeModel(), (), {}),
        lambda: cohere_wrap(tracer, {"span_name": STREAM_SPAN, "method": "chat_stream"})(source, None, (), {}),
        lambda: mistral_wrap(tracer, {"span_name": STREAM_SPAN, "method": "chat", "streaming": True})(
            source, None, (), {}
        ),
    ]


def _async_streams(tracer: Tracer) -> List[Callable[[], Any]]:
    source = _async_source(tracer)

    async def deepgram_generate(*_args: Any, **_kwargs: Any) -> AsyncIterator[str]:
        async for chunk in source():
            yield chunk

    deepgram_wrapper = wrap_async_generator(tracer)
    return [
        lambda: async_generator_wrapper(tracer, STREAM_SPAN, "speech")(source, None, (), {}),
        lambda: _async_stream_start(tracer, source, None, (), {}, STREAM_SPAN, "agent"),
        lambda: model_aresponse_stream_capture_wrapper(tracer)(source, _FakeModel(), (), {}),
        lambda: deepgram_wrapper(deepgram_generate, None, (), {}),
        lambda: base_agent_run_async_wrapper(tracer)(source, SimpleNamespace(name=STREAM_SPAN), (), {}),
        lambda: run_and_handle_error_wrapper(tracer)(source, None, (), {}),
    ]


SYNC_IDS = ["elevenlabs", "agno-agent", "agno-llm", "cohere", "mistral"]
ASYNC_IDS = ["elevenlabs", "agno-agent", "agno-llm", "deepgram", "adk-agent", "adk-llm"]


def _stream_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = [s for s in exporter.get_finished_spans() if s.name not in (INNER_SPAN, "unrelated")]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


def _assert_inner_spans_are_children(exporter: InMemorySpanExporter, stream: ReadableSpan) -> None:
    inner = [s for s in exporter.get_finished_spans() if s.name == INNER_SPAN]
    assert inner
    for span in inner:
        assert span.parent is not None and span.parent.span_id == stream.context.span_id


@pytest.mark.unit
@pytest.mark.parametrize("index", range(len(SYNC_IDS)), ids=SYNC_IDS)
def test_sync_break_early_restores_context_and_close_exports(
    tracer: Tracer, exporter: InMemorySpanExporter, index: int
) -> None:
    stream = _sync_streams(tracer)[index]()

    assert trace.get_current_span() is trace.INVALID_SPAN
    for _ in stream:
        break
    assert trace.get_current_span() is trace.INVALID_SPAN

    with tracer.start_as_current_span("unrelated"):
        pass
    unrelated = next(s for s in exporter.get_finished_spans() if s.name == "unrelated")
    assert unrelated.parent is None

    stream.close()
    stream_span = _stream_span(exporter)
    _assert_inner_spans_are_children(exporter, stream_span)


@pytest.mark.unit
@pytest.mark.parametrize("index", range(len(SYNC_IDS)), ids=SYNC_IDS)
def test_sync_abandoned_stream_is_exported_on_gc(tracer: Tracer, exporter: InMemorySpanExporter, index: int) -> None:
    stream = _sync_streams(tracer)[index]()
    next(iter(stream))
    del stream
    gc.collect()

    _stream_span(exporter)


@pytest.mark.unit
@pytest.mark.parametrize("index", range(len(SYNC_IDS)), ids=SYNC_IDS)
def test_sync_full_consumption_exports_once(tracer: Tracer, exporter: InMemorySpanExporter, index: int) -> None:
    assert list(_sync_streams(tracer)[index]()) == ["chunk-0", "chunk-1", "chunk-2"]
    stream_span = _stream_span(exporter)
    _assert_inner_spans_are_children(exporter, stream_span)
    assert trace.get_current_span() is trace.INVALID_SPAN


@pytest.mark.unit
@pytest.mark.parametrize("index", range(len(ASYNC_IDS)), ids=ASYNC_IDS)
def test_async_break_early_restores_context(tracer: Tracer, exporter: InMemorySpanExporter, index: int) -> None:
    async def run() -> None:
        stream = _async_streams(tracer)[index]()
        async for _ in stream:
            break
        assert trace.get_current_span() is trace.INVALID_SPAN

        with tracer.start_as_current_span("unrelated"):
            pass
        await stream.aclose()

    asyncio.run(run())

    unrelated = next(s for s in exporter.get_finished_spans() if s.name == "unrelated")
    assert unrelated.parent is None
    stream_span = _stream_span(exporter)
    _assert_inner_spans_are_children(exporter, stream_span)


@pytest.mark.unit
@pytest.mark.parametrize("index", range(len(ASYNC_IDS)), ids=ASYNC_IDS)
def test_async_stream_consumed_in_another_task_does_not_fail_to_detach(
    tracer: Tracer, exporter: InMemorySpanExporter, index: int, caplog: pytest.LogCaptureFixture
) -> None:
    async def consume(stream: Any) -> List[str]:
        return [chunk async for chunk in stream]

    async def run() -> List[str]:
        stream = _async_streams(tracer)[index]()
        return await asyncio.create_task(consume(stream))

    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        assert asyncio.run(run()) == ["chunk-0", "chunk-1", "chunk-2"]

    assert "Failed to detach context" not in caplog.text
    _stream_span(exporter)


class _FakeEventStream:
    """Stands in for Mistral's ``EventStream``: an iterator that is also a context manager."""

    def __init__(self, source: Iterator[str]) -> None:
        self._source = source
        self.closed = False

    def __iter__(self) -> "_FakeEventStream":
        return self

    def __next__(self) -> str:
        return next(self._source)

    def __enter__(self) -> "_FakeEventStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True


@pytest.mark.unit
def test_mistral_stream_keeps_the_sdk_context_manager(tracer: Tracer, exporter: InMemorySpanExporter) -> None:
    inner = _FakeEventStream(_sync_source(tracer)())
    stream = mistral_wrap(tracer, {"span_name": STREAM_SPAN, "method": "chat", "streaming": True})(
        lambda *a, **k: inner, None, (), {}
    )

    with stream as event_stream:
        assert [chunk for chunk in event_stream] == ["chunk-0", "chunk-1", "chunk-2"]

    assert inner.closed
    _assert_inner_spans_are_children(exporter, _stream_span(exporter))


@pytest.mark.unit
def test_mistral_async_break_early_restores_context(tracer: Tracer, exporter: InMemorySpanExporter) -> None:
    source = _async_source(tracer)

    async def stream_async(*_args: Any, **_kwargs: Any) -> AsyncIterator[str]:
        return source()

    async def run() -> None:
        stream = await mistral_awrap(tracer, {"span_name": STREAM_SPAN, "method": "chat", "streaming": True})(
            stream_async, None, (), {}
        )
        assert trace.get_current_span() is trace.INVALID_SPAN
        async for _ in stream:
            break
        assert trace.get_current_span() is trace.INVALID_SPAN
        with tracer.start_as_current_span("unrelated"):
            pass
        await stream.aclose()

    asyncio.run(run())

    unrelated = next(s for s in exporter.get_finished_spans() if s.name == "unrelated")
    assert unrelated.parent is None
    _assert_inner_spans_are_children(exporter, _stream_span(exporter))


@pytest.mark.unit
def test_cohere_stream_that_is_never_read_leaves_no_span_attached(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    cohere_wrap(tracer, {"span_name": STREAM_SPAN, "method": "chat_stream"})(_sync_source(tracer), None, (), {})

    assert trace.get_current_span() is trace.INVALID_SPAN
