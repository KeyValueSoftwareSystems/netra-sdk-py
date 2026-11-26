"""Tests for ``Netra.session_scope`` and ``Netra.clear_session_context``."""

import asyncio
import logging
import queue
import threading
from typing import Dict, Iterator, List, Optional, Tuple

import pytest
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra import Netra
from netra.logging_utils import configure_package_logging
from netra.netra_logger import NOTICE_LOGGER_NAME
from netra.processors.session_span_processor import SessionSpanProcessor
from netra.session_manager import ATTR_SESSION_ID, ATTR_TENANT_ID, ATTR_USER_ID, SessionManager

Ids = Tuple[Optional[object], Optional[object], Optional[object]]
NO_IDS: Ids = (None, None, None)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def clean_context() -> Iterator[None]:
    """Run each test in an empty context, discarding its sticky attaches afterwards."""
    token = otel_context.attach(Context())
    try:
        yield
    finally:
        otel_context.detach(token)


@pytest.fixture  # type: ignore[misc]
def span_exporter() -> InMemorySpanExporter:
    """An in-memory exporter collecting the spans a test produces."""
    return InMemorySpanExporter()


@pytest.fixture  # type: ignore[misc]
def tracer(span_exporter: InMemorySpanExporter) -> Iterator[trace.Tracer]:
    """A tracer whose provider runs ``SessionSpanProcessor`` then exports."""
    provider = TracerProvider()
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider.get_tracer(__name__)
    provider.shutdown()


def _ids_by_span(span_exporter: InMemorySpanExporter) -> Dict[str, Ids]:
    """Map each finished span's name to its (session, user, tenant) attributes."""
    result: Dict[str, Ids] = {}
    for span in span_exporter.get_finished_spans():
        attrs = span.attributes or {}
        result[span.name] = (attrs.get(ATTR_SESSION_ID), attrs.get(ATTR_USER_ID), attrs.get(ATTR_TENANT_ID))
    return result


def _run_job(tracer: trace.Tracer, name: str, user: Optional[str] = None, tenant: Optional[str] = None) -> None:
    """A worker job: optionally sets ids the sticky way, then records one span."""
    if user:
        Netra.set_user_id(user)
    if tenant:
        Netra.set_tenant_id(tenant)
    with tracer.start_as_current_span(name):
        pass


JOBS: List[Tuple[str, Optional[str], Optional[str]]] = [
    ("job-a", "alice", "acme"),
    ("job-b", None, None),
]


class TestStickyBehaviourUnchanged:
    """``set_*_id`` keep outliving the call that set them."""

    def test_id_set_in_helper_still_applies_after_it_returns(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        def identify() -> None:
            Netra.set_user_id("alice")

        identify()
        with tracer.start_as_current_span("later"):
            pass

        assert _ids_by_span(span_exporter)["later"] == (None, "alice", None)

    def test_unscoped_worker_loop_still_leaks(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        """Baseline: without a scope or clear, the next job inherits the ids."""
        for job in JOBS:
            _run_job(tracer, *job)

        assert _ids_by_span(span_exporter)["job-b"] == (None, "alice", "acme")


class TestSessionScope:
    """``Netra.session_scope`` labels spans inside it and restores the context on exit."""

    def test_labels_spans_inside_and_not_after(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        with Netra.session_scope(session_id="s-1", user_id="alice", tenant_id="acme"):
            with tracer.start_as_current_span("inside"):
                pass
        with tracer.start_as_current_span("after"):
            pass

        ids = _ids_by_span(span_exporter)
        assert ids["inside"] == ("s-1", "alice", "acme")
        assert ids["after"] == NO_IDS

    def test_worker_loop_with_scope_does_not_leak(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        for name, user, tenant in JOBS:
            with Netra.session_scope(user_id=user, tenant_id=tenant):
                _run_job(tracer, name)

        ids = _ids_by_span(span_exporter)
        assert ids["job-a"] == (None, "alice", "acme")
        assert ids["job-b"] == NO_IDS

    def test_empty_scope_discards_sticky_ids_set_inside(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """With no ids, the scope still isolates jobs that call ``set_*_id`` themselves."""
        for job in JOBS:
            with Netra.session_scope():
                _run_job(tracer, *job)

        ids = _ids_by_span(span_exporter)
        assert ids["job-a"] == (None, "alice", "acme")
        assert ids["job-b"] == NO_IDS

    def test_restores_context_when_block_raises(self) -> None:
        with pytest.raises(RuntimeError):
            with Netra.session_scope(user_id="alice"):
                Netra.set_tenant_id("acme")
                raise RuntimeError("job failed")

        assert baggage.get_baggage("user_id") is None
        assert baggage.get_baggage("tenant_id") is None

    def test_unspecified_fields_inherit_from_enclosing_context(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        Netra.set_tenant_id("acme")  # e.g. set once at worker startup
        with Netra.session_scope(user_id="alice"):
            with tracer.start_as_current_span("inside"):
                pass
        with tracer.start_as_current_span("after"):
            pass

        ids = _ids_by_span(span_exporter)
        assert ids["inside"] == (None, "alice", "acme")
        assert ids["after"] == (None, None, "acme")

    def test_scope_overrides_enclosing_value_then_restores_it(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        Netra.set_user_id("startup-user")
        with Netra.session_scope(user_id="alice"):
            with tracer.start_as_current_span("inside"):
                pass
        with tracer.start_as_current_span("after"):
            pass

        ids = _ids_by_span(span_exporter)
        assert ids["inside"][1] == "alice"
        assert ids["after"][1] == "startup-user"

    def test_does_not_label_span_active_on_entry(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """A batch span around many scoped jobs must not end up with the last job's user."""
        with tracer.start_as_current_span("batch"):
            for name, user, _ in JOBS:
                with Netra.session_scope(user_id=user):
                    with tracer.start_as_current_span(name):
                        pass

        ids = _ids_by_span(span_exporter)
        assert ids["batch"] == NO_IDS
        assert ids["job-a"][1] == "alice"
        assert ids["job-b"] == NO_IDS

    def test_non_string_id_is_reported_and_ignored(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
    ) -> None:
        configure_package_logging(debug_mode=False)

        with caplog.at_level(logging.DEBUG):
            with Netra.session_scope(user_id=123, tenant_id="acme"):  # type: ignore[arg-type]
                with tracer.start_as_current_span("inside"):
                    pass

        assert _ids_by_span(span_exporter)["inside"] == (None, None, "acme")
        notices = [r for r in caplog.records if r.name == NOTICE_LOGGER_NAME]
        assert [(r.levelno, "session_scope: user_id must be a string" in r.getMessage()) for r in notices] == [
            (logging.ERROR, True)
        ]

    def test_worker_thread_consumer_loop(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        jobs: "queue.Queue[Optional[Tuple[str, Optional[str], Optional[str]]]]" = queue.Queue()
        for job in JOBS:
            jobs.put(job)
        jobs.put(None)

        def consumer() -> None:
            while (item := jobs.get()) is not None:
                with Netra.session_scope():
                    _run_job(tracer, *item)

        worker = threading.Thread(target=consumer)
        worker.start()
        worker.join()

        ids = _ids_by_span(span_exporter)
        assert ids["job-a"] == (None, "alice", "acme")
        assert ids["job-b"] == NO_IDS

    def test_asyncio_consumer_loop(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        async def main() -> None:
            jobs: "asyncio.Queue[Optional[Tuple[str, Optional[str], Optional[str]]]]" = asyncio.Queue()
            for job in JOBS:
                await jobs.put(job)
            await jobs.put(None)

            async def consumer() -> None:
                while (item := await jobs.get()) is not None:
                    name, user, tenant = item
                    with Netra.session_scope(user_id=user, tenant_id=tenant):
                        await asyncio.sleep(0)  # the scope survives awaits inside the job
                        _run_job(tracer, name)

            await asyncio.create_task(consumer())

        asyncio.run(main())

        ids = _ids_by_span(span_exporter)
        assert ids["job-a"] == (None, "alice", "acme")
        assert ids["job-b"] == NO_IDS


class TestClearSessionContext:
    """``Netra.clear_session_context`` removes the ids from spans started afterwards."""

    def test_worker_loop_with_clear_does_not_leak(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        for job in JOBS:
            Netra.clear_session_context()
            _run_job(tracer, *job)

        ids = _ids_by_span(span_exporter)
        assert ids["job-a"] == (None, "alice", "acme")
        assert ids["job-b"] == NO_IDS

    def test_clears_all_three_ids(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        Netra.set_session_id("s-1")
        Netra.set_user_id("alice")
        Netra.set_tenant_id("acme")
        Netra.clear_session_context()
        with tracer.start_as_current_span("after"):
            pass

        assert _ids_by_span(span_exporter)["after"] == NO_IDS

    def test_leaves_other_baggage_alone(self) -> None:
        otel_context.attach(baggage.set_baggage("unrelated", "keep-me"))
        Netra.set_user_id("alice")
        Netra.clear_session_context()

        assert baggage.get_baggage("unrelated") == "keep-me"
        assert baggage.get_baggage("user_id") is None

    def test_span_already_started_keeps_its_ids(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        Netra.set_user_id("alice")
        with tracer.start_as_current_span("running"):
            Netra.clear_session_context()
            with tracer.start_as_current_span("child-after-clear"):
                pass

        ids = _ids_by_span(span_exporter)
        assert ids["running"][1] == "alice"
        assert ids["child-after-clear"] == NO_IDS

    def test_ids_can_be_set_again_after_clear(self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter) -> None:
        Netra.set_user_id("alice")
        Netra.clear_session_context()
        Netra.set_user_id("bob")
        with tracer.start_as_current_span("after"):
            pass

        assert _ids_by_span(span_exporter)["after"][1] == "bob"

    def test_no_attach_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attached: List[object] = []
        monkeypatch.setattr(otel_context, "attach", lambda ctx: attached.append(ctx))

        SessionManager.clear_session_context()

        assert attached == []
