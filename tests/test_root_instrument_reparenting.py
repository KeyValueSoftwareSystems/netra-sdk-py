"""Tests for root-instrument filtering with drop-and-reparent semantics.

When a root span comes from an instrumentation that is not in
``root_instruments``, only that span is dropped and its children are reparented
onto its parent — recursively — instead of discarding the whole subtree.

The tests drive the real :class:`RootInstrumentFilterProcessor` (to populate the
shared candidate registry) and the real :class:`FilteringSpanExporter` (to make
the drop + reparent decision), using lightweight span doubles so parent/scope
wiring is explicit.
"""

import threading

import pytest

from netra.exporters.filtering_span_exporter import FilteringSpanExporter
from netra.processors import root_instrument_filter_processor as rifp
from netra.processors.root_instrument_filter_processor import RootInstrumentFilterProcessor

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isolate the process-global candidate registry between tests."""
    with rifp._root_candidates_lock:
        rifp.ROOT_BLOCK_CANDIDATES.clear()
    yield
    with rifp._root_candidates_lock:
        rifp.ROOT_BLOCK_CANDIDATES.clear()


class FakeSpanContext:
    def __init__(self, span_id: int, trace_id: int = 0xABC, is_remote: bool = False) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.is_remote = is_remote


class FakeScope:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSpan:
    """Minimal stand-in exposing the surface the processor/exporter read."""

    def __init__(
        self,
        span_id: int,
        scope_name: str | None,
        parent_ctx: FakeSpanContext | None = None,
        name: str = "span",
        trace_id: int = 0xABC,
    ) -> None:
        self.context = FakeSpanContext(span_id, trace_id)
        self.instrumentation_scope = FakeScope(scope_name) if scope_name else None
        self.parent = parent_ctx
        self.name = name
        self.attributes: dict = {}

    def get_span_context(self) -> FakeSpanContext:
        return self.context

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class RecordingExporter:
    def __init__(self) -> None:
        self.exported: list = []

    def export(self, spans):
        self.exported.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def scope(name: str) -> str:
    return f"netra.instrumentation.{name}"


def make_pipeline(allowed: set[str]):
    processor = RootInstrumentFilterProcessor(allowed)
    recorder = RecordingExporter()
    exporter = FilteringSpanExporter(recorder, patterns=[])
    return processor, exporter, recorder


def exported_ids(recorder: RecordingExporter) -> set[int]:
    return {s.context.span_id for s in recorder.exported}


# ---------------------------------------------------------------------------
# Processor: candidate recording
# ---------------------------------------------------------------------------


def test_processor_records_only_non_allowed_instrumentation():
    processor = RootInstrumentFilterProcessor({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"))
    openai = FakeSpan(2, scope("openai"))
    manual = FakeSpan(3, "my.app.tracer")

    for span in (fastapi, openai, manual):
        processor.on_start(span)

    candidates = rifp.get_root_block_candidates()
    assert set(candidates) == {1}  # only the non-allowed FastAPI span


# ---------------------------------------------------------------------------
# Exporter: drop + reparent
# ---------------------------------------------------------------------------


def test_drops_root_and_reparents_child():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(openai)
    exporter.export([fastapi, openai])

    assert exported_ids(recorder) == {2}
    assert openai.parent is None  # promoted to root


def test_recursive_peel_promotes_deepest_allowed_span():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    asgi = FakeSpan(2, scope("asgi"), parent_ctx=fastapi.context)
    openai = FakeSpan(3, scope("openai"), parent_ctx=asgi.context)

    for span in (fastapi, asgi, openai):
        processor.on_start(span)
    exporter.export([fastapi, asgi, openai])

    assert exported_ids(recorder) == {3}
    assert openai.parent is None


def test_peel_stops_at_allowed_ancestor():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)
    httpx = FakeSpan(3, scope("httpx"), parent_ctx=openai.context)

    for span in (fastapi, openai, httpx):
        processor.on_start(span)
    exporter.export([fastapi, openai, httpx])

    # FastAPI dropped; OpenAI promoted to root; HTTPX kept under OpenAI untouched
    assert exported_ids(recorder) == {2, 3}
    assert openai.parent is None
    assert httpx.parent is openai.context


def test_non_root_span_under_surviving_parent_is_not_dropped():
    # A non-allowed instrumentation span that is *not* root-connected stays.
    processor, exporter, recorder = make_pipeline({"openai"})

    manual_root = FakeSpan(1, "my.app.tracer", parent_ctx=None)
    fastapi = FakeSpan(2, scope("fastapi"), parent_ctx=manual_root.context)

    processor.on_start(manual_root)
    processor.on_start(fastapi)
    exporter.export([manual_root, fastapi])

    assert exported_ids(recorder) == {1, 2}
    assert fastapi.parent is manual_root.context


def test_whole_branch_with_no_allowed_span_is_dropped():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    httpx = FakeSpan(2, scope("httpx"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(httpx)
    exporter.export([fastapi, httpx])

    assert exported_ids(recorder) == set()


def test_cross_batch_reparenting_via_registry():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(openai)

    # Batch 1: only the (dropped) FastAPI root exports.
    exporter.export([fastapi])
    assert exported_ids(recorder) == set()

    # Batch 2: the child exports later and is still reparented to root.
    exporter.export([openai])
    assert exported_ids(recorder) == {2}
    assert openai.parent is None


def test_remote_parent_root_is_not_dropped():
    processor, exporter, recorder = make_pipeline({"openai"})

    remote_parent = FakeSpanContext(999, is_remote=True)
    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=remote_parent)

    processor.on_start(fastapi)
    exporter.export([fastapi])

    # Continues an upstream distributed trace → not peeled.
    assert exported_ids(recorder) == {1}


def test_allowed_root_instrument_exports_normally():
    processor, exporter, recorder = make_pipeline({"fastapi", "openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(openai)
    exporter.export([fastapi, openai])

    assert exported_ids(recorder) == {1, 2}
    assert openai.parent is fastapi.context


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@pytest.mark.thread_safety
def test_concurrent_on_start_is_safe_and_bounded():
    processor = RootInstrumentFilterProcessor({"openai"})
    errors: list = []

    def worker(base: int) -> None:
        try:
            for i in range(200):
                processor.on_start(FakeSpan(base + i, scope("fastapi")))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(rifp.ROOT_BLOCK_CANDIDATES) <= rifp._MAX_ROOT_CANDIDATES
