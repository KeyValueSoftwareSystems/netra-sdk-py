"""Tests for host-gated session baggage across every injector.

Covers the gated global propagator, Netra's own injection helpers, the request
hooks given to the upstream urllib, urllib3 and aiohttp_client instrumentors,
end to end against a local HTTP server.
"""

import asyncio
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple
from unittest.mock import patch

import pytest
from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import get_global_textmap, inject, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from netra.instrumentation.http import propagation
from netra.instrumentation.http.hooks import (
    aiohttp_instrument_kwargs,
    urllib3_instrument_kwargs,
    urllib3_request_hook,
    urllib_instrument_kwargs,
)
from netra.instrumentation.http.propagation import (
    HostGatedPropagator,
    ensure_gated_propagator,
    inject_context,
    inject_local,
    install_gated_propagator,
    register_hook_destination,
)
from netra.instrumentation.instruments import InstrumentSet
from netra.instrumentation.wiring.activation import _instrument_kwargs
from netra.instrumentation.wiring.registry import CUSTOM_INSTRUMENTORS

_PROVIDER = TracerProvider()
_TRACER = _PROVIDER.get_tracer(__name__)


@pytest.fixture(autouse=True)
def gated_propagator() -> Iterator[None]:
    """Install the gated propagator around the default W3C propagators for each test."""
    previous = get_global_textmap()
    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()]))
    install_gated_propagator()
    yield
    set_global_textmap(previous)
    propagation._gating_installed = False
    propagation._replacement_warned = False
    with propagation._hook_destinations_lock:
        propagation._hook_destinations.clear()


@contextmanager
def allowlist(*hosts: str) -> Iterator[None]:
    """Patch the active config's baggage allowlist."""
    cfg = SimpleNamespace(propagate_baggage_hosts=tuple(hosts))
    with patch.object(propagation, "get_active_config", return_value=cfg):
        yield


@contextmanager
def session_span() -> Iterator[trace.Span]:
    """Attach session baggage and open a span, like a traced request handler."""
    ctx = baggage.set_baggage("session_id", "sess-1")
    ctx = baggage.set_baggage("user_id", "user-1", ctx)
    ctx = baggage.set_baggage("tenant_id", "tenant-1", ctx)
    token = context_api.attach(ctx)
    try:
        with _TRACER.start_as_current_span("request") as span:
            yield span
    finally:
        context_api.detach(token)


class _Recorder(BaseHTTPRequestHandler):
    received: List[Dict[str, str]] = []

    def do_GET(self) -> None:  # noqa: N802
        type(self).received.append({k.lower(): v for k, v in self.headers.items()})
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    """A local HTTP server recording the headers of every request it receives."""
    _Recorder.received = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Recorder.received
    httpd.shutdown()
    httpd.server_close()


class TestGatedPropagator:
    def test_install_is_idempotent(self) -> None:
        install_gated_propagator()
        current = get_global_textmap()
        assert isinstance(current, HostGatedPropagator)
        assert not isinstance(current.inner, HostGatedPropagator)

    def test_bare_inject_strips_baggage_but_keeps_trace_context(self) -> None:
        with allowlist("internal.corp"), session_span():
            carrier: Dict[str, str] = {}
            inject(carrier)
        assert "traceparent" in carrier
        assert "baggage" not in carrier

    def test_user_tracecontext_only_propagator_is_preserved(self) -> None:
        set_global_textmap(TraceContextTextMapPropagator())
        install_gated_propagator()
        with allowlist("internal.corp"), session_span():
            carrier: Dict[str, str] = {}
            inject_context(carrier, "https://api.internal.corp/x")
        assert "traceparent" in carrier
        assert "baggage" not in carrier

    def test_fields_and_extract_delegate(self) -> None:
        gated = get_global_textmap()
        assert {"traceparent", "baggage"} <= gated.fields
        ctx = gated.extract({"baggage": "session_id=abc"})
        assert baggage.get_baggage("session_id", ctx) == "abc"


class TestInjectContext:
    @pytest.mark.parametrize(
        "url",
        ["https://api.openai.com/v1/chat", "https://notinternal.corp/x", None, ""],
    )
    def test_strips_baggage_for_untrusted_destinations(self, url: Any) -> None:
        with allowlist("internal.corp"), session_span():
            carrier: Dict[str, str] = {}
            inject_context(carrier, url)
        assert "traceparent" in carrier
        assert "baggage" not in carrier

    def test_keeps_baggage_for_allowlisted_host(self) -> None:
        with allowlist("internal.corp"), session_span():
            carrier: Dict[str, str] = {}
            inject_context(carrier, "https://api.internal.corp/x")
        assert "session_id=sess-1" in carrier["baggage"]
        assert "tenant_id=tenant-1" in carrier["baggage"]

    def test_missing_url_ignores_hook_registration(self) -> None:
        with allowlist("internal.corp"), session_span() as span:
            register_hook_destination(span, "https://api.internal.corp/x")
            carrier: Dict[str, str] = {}
            inject_context(carrier, None)
        assert "baggage" not in carrier

    def test_inject_local_keeps_baggage_without_allowlist(self) -> None:
        with allowlist(), session_span():
            carrier: Dict[str, str] = {}
            inject_local(carrier)
        assert "session_id=sess-1" in carrier["baggage"]


class TestHookRegistry:
    def test_registration_is_consumed_by_one_inject(self) -> None:
        with allowlist("internal.corp"), session_span() as span:
            register_hook_destination(span, "https://api.internal.corp/x")
            first: Dict[str, str] = {}
            inject(first)
            second: Dict[str, str] = {}
            inject(second)
        assert "baggage" in first
        assert "baggage" not in second

    def test_registration_does_not_apply_to_another_span(self) -> None:
        with allowlist("internal.corp"), session_span() as span:
            register_hook_destination(span, "https://api.internal.corp/x")
            with _TRACER.start_as_current_span("other"):
                carrier: Dict[str, str] = {}
                inject(carrier)
        assert "baggage" not in carrier

    def test_invalid_span_is_not_registered(self) -> None:
        register_hook_destination(trace.INVALID_SPAN, "https://api.internal.corp/x")
        assert not propagation._hook_destinations

    def test_registry_is_bounded(self) -> None:
        for _ in range(propagation._HOOK_DESTINATIONS_MAX + 50):
            with _TRACER.start_as_current_span("s") as span:
                register_hook_destination(span, "https://api.internal.corp/x")
        assert len(propagation._hook_destinations) == propagation._HOOK_DESTINATIONS_MAX


class TestUrllib3Hook:
    def test_registers_request_url_as_is(self) -> None:
        # The upstream instrumentor resolves the path against the pool before
        # the hook runs, so request_info.url is already absolute.
        pool = SimpleNamespace(scheme="http", host="proxy.local", port=3128)
        with _TRACER.start_as_current_span("s") as span:
            urllib3_request_hook(span, pool, SimpleNamespace(url="https://api.internal.corp:8443/v1/x"))
        assert list(propagation._hook_destinations.values()) == ["https://api.internal.corp:8443/v1/x"]


def _spec(instrument: InstrumentSet) -> Any:
    (spec,) = CUSTOM_INSTRUMENTORS[instrument]
    return spec


class TestRegistryWiring:
    def test_aiohttp_uses_upstream_instrumentor(self) -> None:
        assert _spec(InstrumentSet.AIOHTTP).module == "opentelemetry.instrumentation.aiohttp_client"

    def test_upstream_http_instrumentors_get_request_hooks(self) -> None:
        assert _instrument_kwargs(_spec(InstrumentSet.URLLIB)) == urllib_instrument_kwargs()
        assert _instrument_kwargs(_spec(InstrumentSet.URLLIB3)) == urllib3_instrument_kwargs()
        assert _instrument_kwargs(_spec(InstrumentSet.AIOHTTP)) == aiohttp_instrument_kwargs()


class TestReplacedPropagator:
    """A propagator set after Netra.init() is re-wrapped instead of bypassing gating."""

    @staticmethod
    def _replace() -> CompositePropagator:
        replacement = CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
        set_global_textmap(replacement)
        return replacement

    def test_inject_context_rewraps_and_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        replacement = self._replace()
        with caplog.at_level("WARNING", logger=propagation.__name__), session_span():
            inject_context({}, "https://api.openai.com/v1")
            self._replace()
            inject_context({}, "https://api.openai.com/v1")
        current = get_global_textmap()
        assert isinstance(current, HostGatedPropagator)
        assert not isinstance(current.inner, HostGatedPropagator)
        assert current.inner is not replacement  # wrapped the second replacement
        assert sum("was replaced after Netra.init()" in r.message for r in caplog.records) == 1

    def test_request_hook_rewraps_before_upstream_inject(self) -> None:
        self._replace()
        with allowlist("internal.corp"), session_span() as span:
            urllib3_request_hook(span, None, SimpleNamespace(url="https://api.openai.com/v1"))
            carrier: Dict[str, str] = {}
            inject(carrier)  # what the upstream instrumentor does next
        assert "traceparent" in carrier
        assert "baggage" not in carrier

    def test_noop_before_install(self) -> None:
        propagation._gating_installed = False
        replacement = self._replace()
        ensure_gated_propagator()
        assert get_global_textmap() is replacement


@pytest.mark.parametrize(("hosts", "expect_baggage"), [(("127.0.0.1",), True), (("internal.corp",), False)])
class TestEndToEnd:
    def test_urllib3(
        self, server: Tuple[str, List[Dict[str, str]]], hosts: Tuple[str, ...], expect_baggage: bool
    ) -> None:
        import urllib3
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

        base, received = server
        instrumentor = URLLib3Instrumentor()
        instrumentor.instrument(tracer_provider=_PROVIDER, **_instrument_kwargs(_spec(InstrumentSet.URLLIB3)))
        try:
            with allowlist(*hosts), session_span():
                urllib3.PoolManager().request("GET", f"{base}/x")
        finally:
            instrumentor.uninstrument()
        assert "traceparent" in received[0]
        assert ("baggage" in received[0]) is expect_baggage

    def test_urllib(
        self, server: Tuple[str, List[Dict[str, str]]], hosts: Tuple[str, ...], expect_baggage: bool
    ) -> None:
        from opentelemetry.instrumentation.urllib import URLLibInstrumentor

        base, received = server
        instrumentor = URLLibInstrumentor()
        instrumentor.instrument(tracer_provider=_PROVIDER, **_instrument_kwargs(_spec(InstrumentSet.URLLIB)))
        try:
            with allowlist(*hosts), session_span():
                urllib.request.urlopen(f"{base}/x").read()
        finally:
            instrumentor.uninstrument()
        assert "traceparent" in received[0]
        assert ("baggage" in received[0]) is expect_baggage

    def test_aiohttp(
        self, server: Tuple[str, List[Dict[str, str]]], hosts: Tuple[str, ...], expect_baggage: bool
    ) -> None:
        import aiohttp
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

        base, received = server

        async def fetch() -> None:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/x") as response:
                    await response.read()
            # Relative path: the destination must resolve through base_url.
            async with aiohttp.ClientSession(base_url=base) as session:
                async with session.get("/y") as response:
                    await response.read()

        instrumentor = AioHttpClientInstrumentor()
        instrumentor.instrument(tracer_provider=_PROVIDER, **_instrument_kwargs(_spec(InstrumentSet.AIOHTTP)))
        try:
            with allowlist(*hosts), session_span():
                asyncio.run(fetch())
        finally:
            instrumentor.uninstrument()
        assert len(received) == 2
        for headers in received:
            assert "traceparent" in headers
            assert ("baggage" in headers) is expect_baggage
