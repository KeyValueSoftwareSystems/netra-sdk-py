"""Tests for root-instrument filtering with drop-and-reparent semantics.

When a root span comes from an instrumentation that is not in
``root_instruments``, only that span is dropped and its children are reparented
onto its parent — recursively — instead of discarding the whole subtree.

The tests drive the real :class:`RootInstrumentFilterProcessor` (to populate the
shared candidate registry) and the real :class:`FilteringSpanExporter` (to make
the drop + reparent decision), using lightweight span doubles so parent/scope
wiring is explicit.
"""

import logging
import threading
import time

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from netra.exporters.filtering_span_exporter import FilteringSpanExporter
from netra.processors import root_instrument_filter_processor as rifp
from netra.processors.root_instrument_filter_processor import ROOT_BLOCK_CANDIDATE_FIELD, RootInstrumentFilterProcessor

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
# Robustness: candidacy must survive registry eviction / clear
# ---------------------------------------------------------------------------


def test_evicted_registry_still_drops_in_batch_root():
    # The blocked root leaks unless candidacy is carried durably on the span:
    # simulate the registry being evicted (overflow/TTL) between on_start and
    # export.  The FastAPI root must still be dropped via its span marker.
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(openai)

    # Registry wiped before export (models 4096-overflow / TTL eviction).
    with rifp._root_candidates_lock:
        rifp.ROOT_BLOCK_CANDIDATES.clear()

    exporter.export([fastapi, openai])

    assert exported_ids(recorder) == {2}  # FastAPI still dropped
    assert openai.parent is None


def test_candidate_marker_is_off_the_attribute_map():
    processor, exporter, recorder = make_pipeline({"openai"})

    # Non-root-connected candidate (under a surviving manual root) is kept.
    manual_root = FakeSpan(1, "my.app.tracer", parent_ctx=None)
    fastapi = FakeSpan(2, scope("fastapi"), parent_ctx=manual_root.context)

    processor.on_start(manual_root)
    processor.on_start(fastapi)

    # Candidacy is carried as a plain instance attribute, never an OTel span
    # attribute, so it cannot be evicted by the attribute limit and is never
    # exported.
    assert getattr(fastapi, ROOT_BLOCK_CANDIDATE_FIELD, False) is True  # marked at on_start
    assert ROOT_BLOCK_CANDIDATE_FIELD not in fastapi.attributes
    assert getattr(manual_root, ROOT_BLOCK_CANDIDATE_FIELD, False) is False  # manual span untouched

    exporter.export([manual_root, fastapi])

    assert exported_ids(recorder) == {1, 2}
    # Nothing to strip: the marker was never in the exported attribute map.
    assert ROOT_BLOCK_CANDIDATE_FIELD not in fastapi.attributes


def test_shutdown_does_not_clear_registry_before_flush():
    processor, exporter, recorder = make_pipeline({"openai"})

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)

    processor.on_start(fastapi)
    processor.on_start(openai)

    # Processor shuts down first (registered before the exporter); the registry
    # must survive so the exporter's final flush still drops the blocked root.
    processor.shutdown()
    assert len(rifp.ROOT_BLOCK_CANDIDATES) >= 1

    exporter.export([fastapi, openai])
    assert exported_ids(recorder) == {2}
    assert openai.parent is None


def test_unrelated_registry_entries_do_not_affect_batch():
    # A large backlog of candidates from other traces must not be dropped or
    # traversed as part of this batch (correctness + bounded work).
    processor, exporter, recorder = make_pipeline({"openai"})

    for other in range(1000, 1500):
        processor.on_start(FakeSpan(other, scope("fastapi"), parent_ctx=None, trace_id=other))

    fastapi = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    openai = FakeSpan(2, scope("openai"), parent_ctx=fastapi.context)
    processor.on_start(fastapi)
    processor.on_start(openai)

    exporter.export([fastapi, openai])

    # Only spans in this batch are affected; unrelated candidates are neither
    # exported nor reparented here.
    assert exported_ids(recorder) == {2}
    assert openai.parent is None


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


# ---------------------------------------------------------------------------
# Interaction between name/local blocking and the root-instrument peel
# ---------------------------------------------------------------------------


def test_name_blocked_ancestor_drops_disallowed_candidate_child():
    # A root dropped by a global name pattern must not let a disallowed
    # instrumentation child be promoted to root in its place.
    processor = RootInstrumentFilterProcessor({"openai"})
    recorder = RecordingExporter()
    exporter = FilteringSpanExporter(recorder, patterns=["BlockedRoot"])

    blocked_root = FakeSpan(1, None, parent_ctx=None, name="BlockedRoot")  # name-blocked, not a candidate
    fastapi = FakeSpan(2, scope("fastapi"), parent_ctx=blocked_root.context, name="fastapi.request")

    processor.on_start(blocked_root)
    processor.on_start(fastapi)
    exporter.export([blocked_root, fastapi])

    # Root dropped by name; FastAPI dropped because it would otherwise become a
    # root from an instrument excluded from root_instruments.
    assert exported_ids(recorder) == set()


def test_name_blocked_ancestor_keeps_allowed_grandchild_as_root():
    # With an allowed span beneath the name-blocked root, it survives and is
    # promoted to root; the disallowed intermediate is dropped.
    processor = RootInstrumentFilterProcessor({"openai"})
    recorder = RecordingExporter()
    exporter = FilteringSpanExporter(recorder, patterns=["BlockedRoot"])

    blocked_root = FakeSpan(1, None, parent_ctx=None, name="BlockedRoot")
    fastapi = FakeSpan(2, scope("fastapi"), parent_ctx=blocked_root.context, name="fastapi.request")
    openai = FakeSpan(3, scope("openai"), parent_ctx=fastapi.context, name="openai.chat")

    for span in (blocked_root, fastapi, openai):
        processor.on_start(span)
    exporter.export([blocked_root, fastapi, openai])

    assert exported_ids(recorder) == {3}
    assert openai.parent is None


# ---------------------------------------------------------------------------
# TTL: an active candidate must not be evicted before it ends
# ---------------------------------------------------------------------------


def test_active_candidate_not_evicted_by_unrelated_span_end():
    processor = RootInstrumentFilterProcessor({"openai"})

    # Long-lived blocked root starts and stays active.
    root = FakeSpan(1, scope("fastapi"), parent_ctx=None)
    processor.on_start(root)

    # An unrelated (allowed) span cycles through start/end many times, each of
    # which triggers an eviction pass. The still-active root must survive.
    for other in range(2, 12):
        unrelated = FakeSpan(other, scope("openai"), parent_ctx=None, trace_id=other)
        processor.on_start(unrelated)
        processor.on_end(unrelated)

    assert 1 in rifp.ROOT_BLOCK_CANDIDATES  # active root retained regardless of elapsed time

    # Once the root ends its TTL clock starts; age it past the TTL and confirm
    # it is then reclaimable.
    processor.on_end(root)
    with rifp._root_candidates_lock:
        parent_ctx, _ended = rifp.ROOT_BLOCK_CANDIDATES[1]
        rifp.ROOT_BLOCK_CANDIDATES[1] = (parent_ctx, time.monotonic() - rifp._ROOT_CANDIDATE_TTL_SECONDS - 1)
    trigger = FakeSpan(99, scope("openai"), parent_ctx=None, trace_id=99)
    processor.on_start(trigger)
    processor.on_end(trigger)

    assert 1 not in rifp.ROOT_BLOCK_CANDIDATES  # ended + stale -> evicted


# ---------------------------------------------------------------------------
# Marker durability: real spans under a tight attribute limit
# ---------------------------------------------------------------------------


def test_candidate_marker_survives_span_attribute_limit():
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider

    from netra.exporters.utils import is_root_block_candidate

    provider = TracerProvider(span_limits=SpanLimits(max_attributes=1))
    provider.add_span_processor(RootInstrumentFilterProcessor({"openai"}))
    tracer = provider.get_tracer("netra.instrumentation.fastapi")

    span = tracer.start_span("fastapi.request")
    # Instrumentation records its own attribute; with max_attributes=1 this would
    # evict an attribute-based marker.
    span.set_attribute("http.method", "GET")
    span.end()

    assert is_root_block_candidate(span) is True
    assert ROOT_BLOCK_CANDIDATE_FIELD not in dict(span.attributes or {})


# ---------------------------------------------------------------------------
# End to end on real SDK spans
#
# The FakeSpan doubles above hand the exporter the *same object* the processor
# marked at on_start.  A real ``Span`` does not: ``Span.end()`` calls
# ``on_end(self._readable_span())``, a fresh ``ReadableSpan`` built from a fixed
# field list that carries no instance attributes.  These tests drive a real
# TracerProvider so that copy is in the loop.
# ---------------------------------------------------------------------------


# Both processors are exercised: SimpleSpanProcessor exports each span in its own
# single-span batch synchronously inside on_end, while BatchSpanProcessor — the
# production default unless ``disable_batch`` is set — queues spans and exports
# several at a time from a worker thread.  Batch composition is exactly what the
# original leak depended on, so neither shape may be left untested.
sdk_span_processors = pytest.mark.parametrize(
    "span_processor_cls",
    [SimpleSpanProcessor, BatchSpanProcessor],
    ids=["simple", "batch"],
)


@pytest.fixture
def make_sdk_pipeline():
    """Build real providers whose exporter is the real FilteringSpanExporter.

    Providers are shut down at teardown so a BatchSpanProcessor worker thread
    does not outlive the test that created it.
    """
    providers = []

    def _make(allowed: set[str], span_processor_cls: type[SpanProcessor]):
        recorder = RecordingExporter()
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(RootInstrumentFilterProcessor(allowed))
        provider.add_span_processor(span_processor_cls(FilteringSpanExporter(recorder, patterns=[])))
        providers.append(provider)
        return provider, recorder

    yield _make

    for provider in providers:
        provider.shutdown()


def exported_names(recorder: RecordingExporter) -> list[str]:
    return [s.name for s in recorder.exported]


@sdk_span_processors
def test_childless_blocked_root_on_a_real_span_is_dropped(make_sdk_pipeline, span_processor_cls):
    # The leak this guards against: a blocked root with no children is nobody's
    # parent, so the exporter's cross-batch registry lookup never fires for it.
    # It is dropped only if the export copy itself carries the candidacy marker.
    provider, recorder = make_sdk_pipeline({"openai"}, span_processor_cls)

    provider.get_tracer(scope("redis")).start_span("SET").end()
    provider.force_flush()

    assert exported_names(recorder) == []


@sdk_span_processors
def test_childless_allowed_root_on_a_real_span_is_exported(make_sdk_pipeline, span_processor_cls):
    provider, recorder = make_sdk_pipeline({"openai"}, span_processor_cls)

    provider.get_tracer(scope("openai")).start_span("openai.chat").end()
    provider.force_flush()

    assert exported_names(recorder) == ["openai.chat"]


@sdk_span_processors
def test_blocked_instrumentation_span_under_a_manual_root_is_kept_on_real_spans(make_sdk_pipeline, span_processor_cls):
    # Only *root-connected* candidates are dropped: the same redis span that is
    # dropped at the root survives under a manual span, which is never a candidate.
    # Under BatchSpanProcessor both spans share one batch, so this also covers the
    # in-batch overlay resolving a parent that is present alongside its child.
    provider, recorder = make_sdk_pipeline({"openai"}, span_processor_cls)

    manual_root = provider.get_tracer("my.app.tracer").start_span("cache-lookup-workflow")
    with trace.use_span(manual_root, end_on_exit=False):
        provider.get_tracer(scope("redis")).start_span("GET").end()
    manual_root.end()
    provider.force_flush()

    assert exported_names(recorder) == ["GET", "cache-lookup-workflow"]
    assert recorder.exported[0].parent.span_id == manual_root.get_span_context().span_id


@sdk_span_processors
def test_blocked_child_exporting_before_its_blocked_root_is_dropped_via_the_registry(
    make_sdk_pipeline, span_processor_cls
):
    # The in-batch marker cannot resolve this one: under SimpleSpanProcessor the
    # child exports alone, in a batch its blocked root has not reached yet, so the
    # child is root-connected only via the cross-batch candidate registry.
    provider, recorder = make_sdk_pipeline({"openai"}, span_processor_cls)

    blocked_root = provider.get_tracer(scope("requests")).start_span("GET /health")
    with trace.use_span(blocked_root, end_on_exit=False):
        provider.get_tracer(scope("redis")).start_span("SET").end()
    blocked_root.end()
    provider.force_flush()

    assert exported_names(recorder) == []


@pytest.mark.thread_safety
def test_blocked_roots_ended_concurrently_are_all_dropped(make_sdk_pipeline):
    # on_end now does correctness-critical work (stamping the export copy) next to
    # registry mutation and TTL eviction, all under concurrent span ends.  Batching
    # is used so the ends and the exporter's reads genuinely overlap.
    provider, recorder = make_sdk_pipeline({"openai"}, BatchSpanProcessor)
    thread_count = 8
    spans_per_thread = 25
    barrier = threading.Barrier(thread_count)
    failures: list[Exception] = []

    def end_blocked_roots() -> None:
        try:
            barrier.wait()  # maximise overlap: every thread starts ending spans together
            for _ in range(spans_per_thread):
                provider.get_tracer(scope("redis")).start_span("SET").end()
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=end_blocked_roots) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    provider.force_flush()

    assert failures == []
    assert exported_names(recorder) == []


def test_marker_stamping_failure_is_reported_not_swallowed(caplog):
    # The whole fix rests on this setattr succeeding.  If it ever stops (a future
    # SDK giving ReadableSpan __slots__), filtering goes inert — so the failure has
    # to be visible rather than silently dropping every span back into the export.
    class SpanRejectingAttributes:
        __slots__ = ()
        name = "SET"

    rifp._marker_failure_reported = False
    try:
        with caplog.at_level(logging.WARNING, logger=rifp.__name__):
            RootInstrumentFilterProcessor._mark_candidate(SpanRejectingAttributes())
    finally:
        rifp._marker_failure_reported = False

    records = [r for r in caplog.records if r.name == rifp.__name__]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert "root-block candidacy marker" in records[0].message


def test_repeated_marker_stamping_failures_are_reported_once_at_warning(caplog):
    # A structural failure must not become one WARNING per ending span.
    class SpanRejectingAttributes:
        __slots__ = ()
        name = "SET"

    rifp._marker_failure_reported = False
    try:
        with caplog.at_level(logging.DEBUG, logger=rifp.__name__):
            for _ in range(3):
                RootInstrumentFilterProcessor._mark_candidate(SpanRejectingAttributes())
    finally:
        rifp._marker_failure_reported = False

    records = [r for r in caplog.records if r.name == rifp.__name__]
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG, logging.DEBUG]


def test_the_export_copy_of_a_blocked_span_carries_the_candidate_marker():
    # Pins the mechanism behind the end-to-end tests in this section: the marker
    # set on the live span at on_start is absent from the ReadableSpan the SDK
    # hands downstream, so the processor has to re-stamp it at on_end.
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider

    from netra.exporters.utils import is_root_block_candidate

    seen: list[ReadableSpan] = []

    class CaptureProcessor(SpanProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            seen.append(span)

    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(RootInstrumentFilterProcessor({"openai"}))
    provider.add_span_processor(CaptureProcessor())

    live_span = provider.get_tracer(scope("redis")).start_span("SET")
    live_span.end()

    (export_copy,) = seen
    assert export_copy is not live_span  # the SDK snapshots the span on end
    assert is_root_block_candidate(export_copy) is True
