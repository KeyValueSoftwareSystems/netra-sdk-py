"""
Span lifecycle tests for the Netra decorators when the decorated callable does not return normally.

Covers audit finding H-20: ``asyncio.CancelledError``, ``KeyboardInterrupt``,
``SystemExit`` and ``GeneratorExit`` are ``BaseException``s, and used to skip
``span.end()``, so the span (and, for a root span, the whole trace) was never
exported. Also covers two related bugs found while verifying it: the entity
stack frame leaked on every failure, and failed spans were exported with
status UNSET instead of ERROR.
"""

import asyncio
import threading
from typing import Any, Callable, Iterator, List

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import netra.decorators as decorators_module
from netra.decorators import task, workflow
from netra.processors.root_span_processor import RootSpanProcessor
from netra.session_manager import SessionManager

CANCELLED_ATTR = "netra.entity.cancelled"
ERROR_ATTR = "netra.entity.error"


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Route decorator spans through a private provider with the root-span processor."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(RootSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(decorators_module.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    yield exporter
    provider.shutdown()


def _run_in_thread(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a fresh thread so context left behind cannot leak into other tests."""
    box: List[Any] = []
    thread = threading.Thread(target=lambda: box.append(fn()))
    thread.start()
    thread.join()
    return box[0]


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def _assert_no_leaked_state() -> None:
    assert SessionManager.get_stack_info() == {"workflows": [], "tasks": [], "agents": [], "spans": []}


class TestSyncFunctionWrapper:
    @pytest.mark.parametrize("exc", [ValueError("boom"), KeyboardInterrupt(), SystemExit(2)])
    def test_failure_ends_span_with_error_status(self, exporter: InMemorySpanExporter, exc: BaseException) -> None:
        @task(name="failing")
        def failing() -> None:
            raise exc

        def run() -> None:
            with pytest.raises(type(exc)):
                failing()
            _assert_no_leaked_state()

        _run_in_thread(run)

        span = _only_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert any(event.name == "exception" for event in span.events)
        assert CANCELLED_ATTR not in span.attributes

    @pytest.mark.parametrize("code", [0, None])
    def test_successful_system_exit_is_not_an_error(self, exporter: InMemorySpanExporter, code: Any) -> None:
        @task(name="exits")
        def exits() -> None:
            raise SystemExit(code)

        def run() -> None:
            with pytest.raises(SystemExit):
                exits()
            _assert_no_leaked_state()

        _run_in_thread(run)

        span = _only_span(exporter)
        assert span.status.status_code == StatusCode.UNSET
        assert ERROR_ATTR not in span.attributes

    def test_exception_propagates_unchanged(self, exporter: InMemorySpanExporter) -> None:
        original = RuntimeError("original")

        @task(name="raises")
        def raises() -> None:
            raise original

        def run() -> BaseException:
            try:
                raises()
            except RuntimeError as e:
                return e
            raise AssertionError("expected RuntimeError")

        assert _run_in_thread(run) is original

    def test_repeated_failures_do_not_leak_entity_frames(self, exporter: InMemorySpanExporter) -> None:
        @task(name="bad")
        def bad() -> None:
            raise ValueError("bad")

        @task(name="good")
        def good() -> str:
            return "ok"

        def run() -> None:
            for _ in range(4):
                with pytest.raises(ValueError):
                    bad()
            _assert_no_leaked_state()
            good()

        _run_in_thread(run)

        good_span = [s for s in exporter.get_finished_spans() if s.name == "good"][0]
        assert good_span.attributes.get("netra.task.name") in (None, "good")

    def test_root_span_registry_does_not_grow(self, exporter: InMemorySpanExporter) -> None:
        @workflow(name="root")
        def root() -> None:
            raise KeyboardInterrupt

        before = len(RootSpanProcessor._root_spans)

        def run() -> None:
            for _ in range(3):
                with pytest.raises(KeyboardInterrupt):
                    root()

        _run_in_thread(run)

        assert len(exporter.get_finished_spans()) == 3
        assert len(RootSpanProcessor._root_spans) == before


class TestAsyncFunctionWrapper:
    def test_cancelled_workflow_exports_its_span(self, exporter: InMemorySpanExporter) -> None:
        started = asyncio.Event

        @workflow(name="cancelled")
        async def long_running(ready: asyncio.Event) -> None:
            ready.set()
            await asyncio.sleep(60)

        async def main() -> None:
            ready = started()
            job = asyncio.create_task(long_running(ready))
            await ready.wait()
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
            _assert_no_leaked_state()

        before = len(RootSpanProcessor._root_spans)
        _run_in_thread(lambda: asyncio.run(main()))

        span = _only_span(exporter)
        assert span.attributes.get(CANCELLED_ATTR) is True
        assert span.status.status_code == StatusCode.UNSET
        assert len(RootSpanProcessor._root_spans) == before

    @pytest.mark.parametrize("exc", [ValueError("boom"), KeyboardInterrupt(), SystemExit(1)])
    def test_failure_ends_span_with_error_status(self, exporter: InMemorySpanExporter, exc: BaseException) -> None:
        @task(name="failing")
        async def failing() -> None:
            raise exc

        async def main() -> None:
            with pytest.raises(type(exc)):
                await failing()
            _assert_no_leaked_state()

        _run_in_thread(lambda: asyncio.run(main()))

        span = _only_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert any(event.name == "exception" for event in span.events)


class TestGeneratorWrappers:
    def test_sync_generator_failure_sets_error_status(self, exporter: InMemorySpanExporter) -> None:
        @task(name="gen")
        def gen() -> Iterator[int]:
            yield 1
            raise ValueError("mid-stream")

        def run() -> None:
            with pytest.raises(ValueError):
                list(gen())
            _assert_no_leaked_state()

        _run_in_thread(run)

        span = _only_span(exporter)
        assert span.status.status_code == StatusCode.ERROR

    def test_sync_generator_early_close_marks_cancelled(self, exporter: InMemorySpanExporter) -> None:
        @task(name="gen")
        def gen() -> Iterator[int]:
            yield 1
            yield 2

        def run() -> None:
            stream = gen()
            next(stream)
            stream.close()
            _assert_no_leaked_state()

        _run_in_thread(run)

        span = _only_span(exporter)
        assert span.attributes.get(CANCELLED_ATTR) is True
        assert span.status.status_code == StatusCode.UNSET

    def test_async_generator_failure_sets_error_status(self, exporter: InMemorySpanExporter) -> None:
        @task(name="agen")
        async def agen() -> Any:
            yield 1
            raise ValueError("mid-stream")

        async def main() -> None:
            with pytest.raises(ValueError):
                async for _ in agen():
                    pass

        _run_in_thread(lambda: asyncio.run(main()))

        span = _only_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
