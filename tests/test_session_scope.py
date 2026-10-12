"""
Tests for scoping and clearing session identity (audit finding H-12).

``Netra.set_session_id`` / ``set_user_id`` / ``set_tenant_id`` attach a context
that is never detached, so on a long-lived thread one request's ids were
inherited by the next. These tests cover ``Netra.session_scope`` (restored on
exit), ``Netra.clear_session`` and clearing a single key with ``""`` / ``None``,
while keeping the sticky setters backward compatible.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra import Netra
from netra.processors.session_span_processor import SessionSpanProcessor
from netra.session_manager import ATTR_SESSION_ID, ATTR_TENANT_ID, ATTR_USER_ID, SessionManager

_IDS = {"session_id": ATTR_SESSION_ID, "user_id": ATTR_USER_ID, "tenant_id": ATTR_TENANT_ID}


@pytest.fixture
def tracer() -> Iterator[Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer(__name__)
    provider.shutdown()


def _ids_on_new_span(tracer: Any) -> Dict[str, Optional[str]]:
    """Start and end a span and return the session ids it was stamped with."""
    span = tracer.start_span("probe")
    span.end()
    return {key: span.attributes.get(attr) for key, attr in _IDS.items()}


def _ids_in_baggage() -> Dict[str, Optional[str]]:
    ctx = otel_context.get_current()
    return {key: baggage.get_baggage(key, ctx) for key in _IDS}  # type: ignore[misc]


def _run_in_thread(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a fresh thread: the sticky setters leave context behind by design."""
    box: List[Any] = []
    errors: List[BaseException] = []

    def target() -> None:
        try:
            box.append(fn())
        except BaseException as e:  # surface assertion errors to the test thread
            errors.append(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return box[0] if box else None


EMPTY = {"session_id": None, "user_id": None, "tenant_id": None}


class TestSessionScope:
    def test_long_lived_thread_does_not_bleed_between_requests(self, tracer: Any) -> None:
        """A worker loop that scopes each request never sees the previous request's ids."""

        def worker() -> List[Dict[str, Optional[str]]]:
            seen = []
            requests = [
                {"session_id": "s1", "user_id": "userA", "tenant_id": "tenantA"},
                {},
                {"user_id": "userB"},
            ]
            for ids in requests:
                with Netra.session_scope(**ids):
                    seen.append(_ids_on_new_span(tracer))
            return seen

        seen = _run_in_thread(worker)
        assert seen[0] == {"session_id": "s1", "user_id": "userA", "tenant_id": "tenantA"}
        assert seen[1] == EMPTY
        # No mixed identity: tenantA from request 1 is not combined with userB.
        assert seen[2] == {"session_id": None, "user_id": "userB", "tenant_id": None}

    def test_scope_restores_previous_values_on_exit(self, tracer: Any) -> None:
        def run() -> Dict[str, Any]:
            with Netra.session_scope(session_id="outer", user_id="u-outer"):
                with Netra.session_scope(user_id="u-inner"):
                    inner = _ids_on_new_span(tracer)
                after_inner = _ids_on_new_span(tracer)
            return {"inner": inner, "after_inner": after_inner, "after_outer": _ids_on_new_span(tracer)}

        result = _run_in_thread(run)
        # Unspecified keys are inherited from the enclosing scope.
        assert result["inner"] == {"session_id": "outer", "user_id": "u-inner", "tenant_id": None}
        assert result["after_inner"] == {"session_id": "outer", "user_id": "u-outer", "tenant_id": None}
        assert result["after_outer"] == EMPTY

    def test_scope_restores_when_body_raises(self, tracer: Any) -> None:
        def run() -> Dict[str, Optional[str]]:
            with pytest.raises(RuntimeError):
                with Netra.session_scope(session_id="s", user_id="u", tenant_id="t"):
                    raise RuntimeError("handler failed")
            return _ids_on_new_span(tracer)

        assert _run_in_thread(run) == EMPTY

    def test_sibling_asyncio_tasks_are_isolated(self, tracer: Any) -> None:
        async def handle(user: str) -> Dict[str, Optional[str]]:
            with Netra.session_scope(user_id=user):
                await asyncio.sleep(0)
                return _ids_on_new_span(tracer)

        async def main() -> List[Dict[str, Optional[str]]]:
            return list(await asyncio.gather(handle("a"), handle("b")))

        results = _run_in_thread(lambda: asyncio.run(main()))
        assert [r["user_id"] for r in results] == ["a", "b"]

    def test_session_manager_scope_still_works(self, tracer: Any) -> None:
        """The SessionManager-level scope (used by the LiveKit wrapper) is unchanged."""

        def run() -> Dict[str, Any]:
            with SessionManager.session_scope(session_id="room-1"):
                inside = _ids_on_new_span(tracer)
            return {"inside": inside, "after": _ids_on_new_span(tracer)}

        result = _run_in_thread(run)
        assert result["inside"]["session_id"] == "room-1"
        assert result["after"] == EMPTY


class TestClearing:
    @pytest.mark.parametrize("key", ["session_id", "user_id", "tenant_id"])
    @pytest.mark.parametrize("clear_value", ["", None])
    def test_empty_or_none_clears_only_that_key(self, tracer: Any, key: str, clear_value: Optional[str]) -> None:
        def run() -> Dict[str, Optional[str]]:
            Netra.set_session_id("s")
            Netra.set_user_id("u")
            Netra.set_tenant_id("t")
            getattr(Netra, f"set_{key}")(clear_value)
            return _ids_on_new_span(tracer)

        result = _run_in_thread(run)
        expected = {"session_id": "s", "user_id": "u", "tenant_id": "t"}
        expected[key] = None
        assert result == expected

    def test_clear_session_clears_all_keys(self, tracer: Any) -> None:
        def run() -> Dict[str, Any]:
            Netra.set_session_id("s")
            Netra.set_user_id("u")
            Netra.set_tenant_id("t")
            Netra.clear_session()
            return {"span": _ids_on_new_span(tracer), "baggage": _ids_in_baggage()}

        result = _run_in_thread(run)
        assert result["span"] == EMPTY
        assert result["baggage"] == EMPTY

    def test_clear_when_nothing_is_set_is_a_no_op(self, tracer: Any) -> None:
        def run() -> Dict[str, Optional[str]]:
            Netra.clear_session()
            Netra.set_user_id("")
            return _ids_on_new_span(tracer)

        assert _run_in_thread(run) == EMPTY

    def test_non_string_value_is_rejected_without_clearing(self, tracer: Any) -> None:
        def run() -> Dict[str, Optional[str]]:
            Netra.set_user_id("u")
            Netra.set_user_id(123)  # type: ignore[arg-type]
            return _ids_on_new_span(tracer)

        assert _run_in_thread(run)["user_id"] == "u"


class TestStickySettersBackwardCompat:
    def test_set_at_startup_applies_to_later_spans_on_the_same_thread(self, tracer: Any) -> None:
        def run() -> List[Dict[str, Optional[str]]]:
            Netra.set_session_id("startup-session")
            Netra.set_tenant_id("startup-tenant")
            return [_ids_on_new_span(tracer), _ids_on_new_span(tracer)]

        for ids in _run_in_thread(run):
            assert ids == {"session_id": "startup-session", "user_id": None, "tenant_id": "startup-tenant"}

    def test_scope_inside_sticky_values_inherits_and_restores(self, tracer: Any) -> None:
        def run() -> Dict[str, Any]:
            Netra.set_tenant_id("tenant")
            with Netra.session_scope(user_id="u"):
                inside = _ids_on_new_span(tracer)
            return {"inside": inside, "after": _ids_on_new_span(tracer)}

        result = _run_in_thread(run)
        assert result["inside"] == {"session_id": None, "user_id": "u", "tenant_id": "tenant"}
        assert result["after"] == {"session_id": None, "user_id": None, "tenant_id": "tenant"}
