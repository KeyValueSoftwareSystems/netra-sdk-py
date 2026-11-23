"""Tests for host-gated session baggage across every injector.

Covers the gated global propagator, Netra's own injection helpers, the request
hooks given to the upstream urllib, urllib3 and aiohttp_client instrumentors,
end to end against a local HTTP server.
"""

import asyncio
import json
import logging
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple
from unittest.mock import patch
from urllib.parse import quote, unquote

import pytest
from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import get_global_textmap, inject, set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from netra.exporters.filtering_span_exporter import FilteringSpanExporter
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
    extract_local,
    inject_context,
    inject_local,
    install_gated_propagator,
    register_hook_destination,
)
from netra.instrumentation.instruments import InstrumentSet
from netra.instrumentation.libraries.subprocess.utils import extract_subprocess_context
from netra.instrumentation.wiring.activation import _instrument_kwargs
from netra.instrumentation.wiring.registry import CUSTOM_INSTRUMENTORS
from netra.processors.local_filtering_span_processor import (
    _LOCAL_BLOCKED_SPANS_BAGGAGE_KEY,
    LocalFilteringSpanProcessor,
    block_spans_local,
)
from netra.processors.session_span_processor import SessionSpanProcessor
from netra.session_manager import (
    ATTR_SESSION_ID,
    ATTR_TENANT_ID,
    ATTR_USER_ID,
    CUSTOM_BAGGAGE_PREFIX,
    CUSTOM_KEYS_BAGGAGE_KEY,
    SESSION_BAGGAGE_KEYS,
)

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
    propagation._inbound_session_notice_logged = False
    with propagation._hook_destinations_lock:
        propagation._hook_destinations.clear()


@contextmanager
def allowlist(*hosts: str, trust_inbound: bool = False) -> Iterator[None]:
    """Patch the active config's baggage allowlist and inbound-trust setting."""
    cfg = SimpleNamespace(propagate_baggage_hosts=tuple(hosts), trust_inbound_baggage=trust_inbound)
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
        if self.path.startswith("/redirect?to="):
            self.send_response(302)
            self.send_header("Location", unquote(self.path[len("/redirect?to=") :]))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
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
        ctx = gated.extract({"baggage": "app.region=eu"})
        assert baggage.get_baggage("app.region", ctx) == "eu"


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

    def test_control_keys_never_sent_to_another_host(self) -> None:
        with allowlist("internal.corp"), session_span(), block_spans_local(["llm*"]):
            carrier: Dict[str, str] = {}
            inject_context(carrier, "https://api.internal.corp/x")
        assert "session_id=sess-1" in carrier["baggage"]
        assert "netra.local_blocked_spans" not in carrier["baggage"]

    def test_inject_local_keeps_control_keys(self) -> None:
        with allowlist(), session_span(), block_spans_local(["llm*"]):
            carrier: Dict[str, str] = {}
            inject_local(carrier)
        assert "netra.local_blocked_spans" in carrier["baggage"]


def _forged_headers() -> Dict[str, str]:
    """Inbound headers a caller could send to pose as another user and hide its spans."""
    entries = {
        "session_id": "forged-session",
        "user_id": "victim",
        "tenant_id": "other-customer",
        "custom_keys": "plan",
        "custom.plan": "enterprise",
        "netra.local_blocked_spans": json.dumps(["GET*", "llm*"]),
        "app.region": "eu",
    }
    return {
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "baggage": ",".join(f"{k}={quote(v, safe='')}" for k, v in entries.items()),
    }


_RESERVED = ("session_id", "user_id", "tenant_id", "custom_keys", "custom.plan", "netra.local_blocked_spans")


class TestInboundBaggage:
    """``extract`` is a trust boundary: callers must not set Netra-reserved baggage."""

    def test_reserved_keys_dropped_by_default(self) -> None:
        with allowlist():
            ctx = get_global_textmap().extract(_forged_headers())
        for key in _RESERVED:
            assert baggage.get_baggage(key, ctx) is None, key

    def test_trace_context_and_other_baggage_kept(self) -> None:
        with allowlist():
            ctx = get_global_textmap().extract(_forged_headers())
        assert baggage.get_baggage("app.region", ctx) == "eu"
        span_context = trace.get_current_span(ctx).get_span_context()
        assert span_context.is_valid
        assert format(span_context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"

    def test_session_keys_accepted_when_inbound_trusted(self) -> None:
        with allowlist(trust_inbound=True):
            ctx = get_global_textmap().extract(_forged_headers())
        assert baggage.get_baggage("user_id", ctx) == "victim"
        assert baggage.get_baggage("tenant_id", ctx) == "other-customer"
        assert baggage.get_baggage("custom.plan", ctx) == "enterprise"
        assert baggage.get_baggage("netra.local_blocked_spans", ctx) is None

    def test_header_cannot_override_values_already_in_context(self) -> None:
        base = baggage.set_baggage("user_id", "real-user")
        base = baggage.set_baggage("netra.local_blocked_spans", json.dumps(["internal*"]), base)
        with allowlist():
            ctx = get_global_textmap().extract(_forged_headers(), context=base)
        assert baggage.get_baggage("user_id", ctx) == "real-user"
        assert baggage.get_baggage("netra.local_blocked_spans", ctx) == json.dumps(["internal*"])
        assert baggage.get_baggage("tenant_id", ctx) is None

    def test_notice_logged_once(self, caplog: pytest.LogCaptureFixture) -> None:
        with allowlist(), caplog.at_level(logging.WARNING, logger="netra.notice"):
            get_global_textmap().extract(_forged_headers())
            get_global_textmap().extract(_forged_headers())
        notices = [r for r in caplog.records if "NETRA_TRUST_INBOUND_BAGGAGE" in r.getMessage()]
        assert len(notices) == 1

    def test_no_notice_without_session_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        with allowlist(), caplog.at_level(logging.WARNING, logger="netra.notice"):
            get_global_textmap().extract({"baggage": "app.region=eu"})
        assert not [r for r in caplog.records if "NETRA_TRUST_INBOUND_BAGGAGE" in r.getMessage()]

    def test_every_key_the_processors_read_is_filtered(self) -> None:
        # Built from the constants the processors read, so a key added there is covered here.
        keys = [
            *SESSION_BAGGAGE_KEYS,
            CUSTOM_KEYS_BAGGAGE_KEY,
            f"{CUSTOM_BAGGAGE_PREFIX}plan",
            _LOCAL_BLOCKED_SPANS_BAGGAGE_KEY,
        ]
        with allowlist():
            ctx = get_global_textmap().extract({"baggage": ",".join(f"{k}=forged" for k in keys)})
        assert {k: baggage.get_baggage(k, ctx) for k in keys} == {k: None for k in keys}

    def test_extract_local_trusts_this_hosts_carrier(self) -> None:
        with allowlist():
            ctx = extract_local(_forged_headers())
        assert baggage.get_baggage("user_id", ctx) == "victim"
        assert baggage.get_baggage("netra.local_blocked_spans", ctx) is not None

    def test_subprocess_child_restores_parent_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with allowlist(), session_span():
            env: Dict[str, str] = {}
            inject_local(env)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        with allowlist():
            token = extract_subprocess_context()
        assert token is not None
        try:
            assert baggage.get_baggage("user_id") == "user-1"
        finally:
            context_api.detach(token)


class TestFastAPIInboundBaggage:
    """End to end: a forged ``baggage`` header neither re-labels nor hides a FastAPI request."""

    @pytest.fixture
    def app_spans(self) -> Iterator[Tuple[Any, InMemorySpanExporter]]:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from netra.instrumentation.libraries.fastapi import NetraFastAPIInstrumentor

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(LocalFilteringSpanProcessor())
        provider.add_span_processor(SessionSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(FilteringSpanExporter(exporter, [])))

        app = FastAPI()

        @app.get("/chat")
        def chat() -> Dict[str, bool]:
            with provider.get_tracer("app").start_as_current_span("llm.call"):
                pass
            return {"ok": True}

        NetraFastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        yield TestClient(app), exporter

    def test_forged_identity_and_block_ignored(self, app_spans: Tuple[Any, InMemorySpanExporter]) -> None:
        client, exporter = app_spans
        with allowlist():
            client.get("/chat", headers=_forged_headers())
        spans = exporter.get_finished_spans()
        assert sorted(s.name for s in spans) == ["GET", "llm.call"]
        for span in spans:
            attrs = dict(span.attributes or {})
            assert not [k for k in attrs if k in (ATTR_SESSION_ID, ATTR_USER_ID, ATTR_TENANT_ID)], attrs
            assert "netra.custom.plan" not in attrs
            assert "netra.local_blocked_spans" not in attrs

    def test_trusted_callers_set_session(self, app_spans: Tuple[Any, InMemorySpanExporter]) -> None:
        client, exporter = app_spans
        with allowlist(trust_inbound=True):
            client.get("/chat", headers=_forged_headers())
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            assert (span.attributes or {}).get(ATTR_USER_ID) == "victim"


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
        aiohttp_kwargs = _instrument_kwargs(_spec(InstrumentSet.AIOHTTP))
        assert aiohttp_kwargs["request_hook"] is aiohttp_instrument_kwargs()["request_hook"]
        (trace_config,) = aiohttp_kwargs["trace_configs"]
        assert list(trace_config.on_request_redirect)


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


def _get_httpx(url: str) -> None:
    import httpx

    httpx.get(url, follow_redirects=True)


def _get_requests(url: str) -> None:
    import requests

    requests.get(url)


def _get_urllib3(url: str) -> None:
    import urllib3

    urllib3.PoolManager().request("GET", url)


def _get_urllib(url: str) -> None:
    urllib.request.urlopen(url).read()


def _get_aiohttp(url: str) -> None:
    import aiohttp

    async def fetch() -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                await response.read()

    asyncio.run(fetch())


@contextmanager
def _instrumented(instrument: InstrumentSet) -> Iterator[None]:
    """Apply *instrument* exactly as Netra's activation does, then remove it."""
    from importlib import import_module

    spec = _spec(instrument)
    instrumentor = getattr(import_module(spec.module), spec.class_name)()
    instrumentor.instrument(tracer_provider=_PROVIDER, **_instrument_kwargs(spec))
    try:
        yield
    finally:
        instrumentor.uninstrument()


@pytest.mark.parametrize(
    ("instrument", "get"),
    [
        (InstrumentSet.HTTPX, _get_httpx),
        (InstrumentSet.REQUESTS, _get_requests),
        (InstrumentSet.URLLIB3, _get_urllib3),
        (InstrumentSet.URLLIB, _get_urllib),
        (InstrumentSet.AIOHTTP, _get_aiohttp),
    ],
)
@pytest.mark.parametrize(
    ("hosts", "expect_baggage_after_redirect"),
    [(("localhost",), False), (("localhost", "127.0.0.1"), True)],
)
class TestRedirects:
    """An allowlisted first hop must not carry baggage across a redirect to a host that is not."""

    def test_cross_host_redirect(
        self,
        server: Tuple[str, List[Dict[str, str]]],
        instrument: InstrumentSet,
        get: Any,
        hosts: Tuple[str, ...],
        expect_baggage_after_redirect: bool,
    ) -> None:
        base, received = server
        port = base.rsplit(":", 1)[1]
        start = f"http://localhost:{port}/redirect?to={quote(f'http://127.0.0.1:{port}/final', safe='')}"
        with _instrumented(instrument), allowlist(*hosts), session_span():
            get(start)
        assert len(received) == 2
        first, final = received
        assert "session_id=sess-1" in first["baggage"]
        assert ("baggage" in final) is expect_baggage_after_redirect
