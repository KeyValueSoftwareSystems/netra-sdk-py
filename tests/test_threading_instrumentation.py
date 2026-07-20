"""Tests for thread context propagation (NET-1276).

Verifies that:
  1. ``THREADING`` is part of the default (non-root) instrument set, so the
     OTel threading instrumentor is enabled by ``Netra.init()`` out of the box.
  2. With the threading instrumentor active, spans created inside a
     ``ThreadPoolExecutor`` attach to the active parent workflow trace instead
     of becoming independent root traces.

Tracers are bound directly to a local ``TracerProvider`` (rather than the
global one) so these tests neither depend on nor mutate global tracing state.
The threading instrumentor propagates the OTel *context*, which is
provider-independent, so a locally-bound tracer exercises it faithfully.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List

import pytest
from opentelemetry import trace
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.config import Config
from netra.instrumentation.instruments import (
    DEFAULT_INSTRUMENTS,
    DEFAULT_INSTRUMENTS_FOR_ROOT,
    InstrumentSet,
)
from netra.processors.session_span_processor import SessionSpanProcessor
from netra.session_manager import SessionManager

_WORKFLOW_ATTR = f"{Config.LIBRARY_NAME}.workflow.name"
_TASK_ATTR = f"{Config.LIBRARY_NAME}.task.name"


class TestThreadingInDefaults:
    def test_threading_enabled_by_default(self) -> None:
        """Threading context propagation ships on by default."""
        assert InstrumentSet.THREADING in DEFAULT_INSTRUMENTS

    def test_threading_not_a_root_instrument(self) -> None:
        """The threading instrumentor only propagates context; it emits no
        spans of its own, so it must not be in the root allow-list."""
        assert InstrumentSet.THREADING not in DEFAULT_INSTRUMENTS_FOR_ROOT


@pytest.fixture  # type: ignore[misc]
def exporter_and_tracer() -> Iterator[tuple[InMemorySpanExporter, trace.Tracer]]:
    """A local in-memory exporter and a tracer bound directly to its provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider.get_tracer("test")


@pytest.fixture  # type: ignore[misc]
def threading_instrumented() -> Iterator[None]:
    """Ensure the threading instrumentor is active for the test and restore
    the prior state afterwards."""
    instrumentor = ThreadingInstrumentor()
    was_instrumented = instrumentor.is_instrumented_by_opentelemetry
    if not was_instrumented:
        instrumentor.instrument()
    try:
        yield
    finally:
        if not was_instrumented:
            instrumentor.uninstrument()


class TestThreadPoolPropagation:
    def test_threadpool_children_attach_to_workflow_trace(
        self,
        exporter_and_tracer: tuple[InMemorySpanExporter, trace.Tracer],
        threading_instrumented: None,
    ) -> None:
        """Spans created in ThreadPoolExecutor workers share the workflow's
        trace_id and are parented to the workflow span."""
        exporter, tracer = exporter_and_tracer

        def run_tool_in_thread(tool_id: int) -> str:
            with tracer.start_as_current_span(f"parallel_tool_{tool_id}"):
                return f"tool_{tool_id}:OK"

        with tracer.start_as_current_span("workflow") as wf:
            workflow_trace_id = wf.get_span_context().trace_id
            workflow_span_id = wf.get_span_context().span_id
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(run_tool_in_thread, i) for i in range(4)]
                for future in as_completed(futures):
                    future.result()

        spans: List[ReadableSpan] = list(exporter.get_finished_spans())
        children = [s for s in spans if s.name.startswith("parallel_tool_")]

        assert len(children) == 4
        for child in children:
            assert child.context.trace_id == workflow_trace_id
            assert child.parent is not None
            assert child.parent.span_id == workflow_span_id

    def test_without_instrumentor_children_are_roots(
        self, exporter_and_tracer: tuple[InMemorySpanExporter, trace.Tracer]
    ) -> None:
        """Control: without the threading instrumentor, worker spans detach
        into their own root traces (reproduces the NET-1276 bug)."""
        exporter, tracer = exporter_and_tracer

        instrumentor = ThreadingInstrumentor()
        was_instrumented = instrumentor.is_instrumented_by_opentelemetry
        if was_instrumented:
            instrumentor.uninstrument()
        try:

            def child(i: int) -> None:
                with tracer.start_as_current_span(f"parallel_tool_{i}"):
                    pass

            with tracer.start_as_current_span("workflow") as wf:
                workflow_trace_id = wf.get_span_context().trace_id
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for f in as_completed([executor.submit(child, i) for i in range(4)]):
                        f.result()

            children = [s for s in exporter.get_finished_spans() if s.name.startswith("parallel_tool_")]
            assert len(children) == 4
            assert all(c.context.trace_id != workflow_trace_id for c in children)
        finally:
            if was_instrumented:
                instrumentor.instrument()


class TestEntityContextPropagation:
    """NET-1276 Problem 2: SessionManager entity stacks live in the OTel
    context, so they propagate into worker threads (via the threading
    instrumentor) and stay isolated between concurrent siblings."""

    def test_worker_inherits_workflow_entity(self, threading_instrumented: None) -> None:
        """A worker thread sees the workflow entity pushed by the parent."""
        token = SessionManager.push_entity("workflow", "wf1")
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                attrs: Dict[str, str] = executor.submit(SessionManager.get_current_entity_attributes).result()
        finally:
            SessionManager.pop_entity("workflow", token)

        assert attrs.get(_WORKFLOW_ATTR) == "wf1"
        # And the parent's stack is restored after pop.
        assert SessionManager.get_current_entity_attributes().get(_WORKFLOW_ATTR) is None

    def test_concurrent_siblings_do_not_cross_contaminate(self, threading_instrumented: None) -> None:
        """Each worker pushes its own task under the shared workflow; workers
        must see their own task name, never a sibling's."""
        token = SessionManager.push_entity("workflow", "wf1")

        def worker(task_id: int) -> Dict[str, str]:
            task_token = SessionManager.push_entity("task", f"task_{task_id}")
            try:
                # Interleave to give races a chance to surface.
                for _ in range(50):
                    attrs = SessionManager.get_current_entity_attributes()
                    if attrs.get(_TASK_ATTR) != f"task_{task_id}":
                        return {"error": attrs.get(_TASK_ATTR, "<missing>")}
                return SessionManager.get_current_entity_attributes()
            finally:
                SessionManager.pop_entity("task", task_token)

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = [f.result() for f in [executor.submit(worker, i) for i in range(4)]]
        finally:
            SessionManager.pop_entity("workflow", token)

        for i, attrs in enumerate(results):
            assert "error" not in attrs, f"worker {i} observed sibling task {attrs.get('error')}"
            assert attrs.get(_WORKFLOW_ATTR) == "wf1"
            assert attrs.get(_TASK_ATTR) == f"task_{i}"

    def test_worker_span_is_stamped_with_workflow_name(self, threading_instrumented: None) -> None:
        """A span STARTED inside a worker is stamped with the parent workflow's
        name by SessionSpanProcessor (which reads the propagated OTel context)."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SessionSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        token = SessionManager.push_entity("workflow", "wf1")
        try:

            def worker(i: int) -> None:
                with tracer.start_as_current_span(f"tool_{i}"):
                    pass

            with ThreadPoolExecutor(max_workers=3) as executor:
                for f in as_completed([executor.submit(worker, i) for i in range(3)]):
                    f.result()
        finally:
            SessionManager.pop_entity("workflow", token)

        tool_spans = [s for s in exporter.get_finished_spans() if s.name.startswith("tool_")]
        assert len(tool_spans) == 3
        for span in tool_spans:
            assert span.attributes is not None
            assert span.attributes.get(_WORKFLOW_ATTR) == "wf1"
