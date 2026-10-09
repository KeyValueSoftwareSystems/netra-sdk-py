"""Host-scoped context propagation for every outbound injector.

Netra stores the session identity (``session_id``/``user_id``/``tenant_id`` and
any custom session keys) as W3C baggage.  The default OpenTelemetry global
propagator serialises baggage into a ``baggage:`` header on *every* outbound
request, so a bare :func:`opentelemetry.propagate.inject` leaks that identity to
third-party APIs (LLM providers included).

Gating therefore happens in the global propagator itself:
:func:`install_gated_propagator` wraps whatever global propagator is configured
in a :class:`HostGatedPropagator`.  On every ``inject`` -- from Netra's own
wrappers, upstream OpenTelemetry instrumentors, or user code -- baggage is
removed from the context before the inner propagator sees it, unless the
destination is known and allowed.  Trace context (``traceparent``/``tracestate``)
is always propagated.

The destination reaches the propagator in one of two ways:

* Callers that own the injection pass it in the context:
  :func:`inject_context` (Netra's httpx, requests and aiohttp wrappers) and
  :func:`inject_local` (subprocess environments, always allowed).
* Upstream instrumentors that call ``inject(headers)`` themselves get a
  ``request_hook`` (``netra.instrumentation.http.hooks``) that registers the
  destination against the client span just before they inject; the propagator
  consumes that registration for the current span.

The allowlist is fail-closed: with no hosts configured -- or no known
destination, as for gRPC, message brokers or botocore -- baggage propagates to
*nobody*.  Internal services are opted back in via the
``NETRA_PROPAGATE_BAGGAGE_HOSTS`` environment variable.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, MutableMapping, Optional, Set, Tuple, cast
from urllib.parse import urlparse

from opentelemetry import baggage, trace
from opentelemetry.context import Context, create_key, get_current, get_value, set_value
from opentelemetry.propagate import get_global_textmap, inject, set_global_textmap
from opentelemetry.propagators.textmap import (
    CarrierT,
    Getter,
    Setter,
    TextMapPropagator,
    default_getter,
    default_setter,
)

from netra.config import get_active_config

logger = logging.getLogger(__name__)

# OpenTelemetry's W3CBaggagePropagator always writes this (lowercase) header.
# We match case-insensitively when stripping, to be safe against carriers that
# normalise header casing differently.
_BAGGAGE_HEADER = "baggage"

# Context key carrying the destination of the injection in progress: a URL, or
# LOCAL_DESTINATION for carriers that never leave the host.
_DESTINATION_KEY = create_key("netra.propagation.destination")

# Destination of an in-process child (subprocess environment): always allowed.
LOCAL_DESTINATION = "netra:local"

# Destinations registered by request hooks, keyed by the client span they were
# registered under.  Bounded so an injection that never happens (a request that
# fails between hook and inject) cannot grow it without limit.
_HOOK_DESTINATIONS_MAX = 1024
_hook_destinations: "OrderedDict[Tuple[int, int], str]" = OrderedDict()
_hook_destinations_lock = threading.Lock()

# Set by install_gated_propagator(); ensure_gated_propagator() only re-wraps
# after that, and warns about a replaced propagator once per process.
_gating_installed = False
_replacement_warned = False

# Clients whose redirect guard has already logged a failure at warning level.
_redirect_guard_warned: Set[str] = set()


def _host_allowed(url: Optional[str]) -> bool:
    """Return True when *url*'s host is on the configured internal allowlist.

    Fail-closed: returns False when *url* is missing/unparseable, when no
    allowlist is configured, or when nothing matches.  A host matches an
    allowlist entry when it equals the entry or is a subdomain of it (so
    ``mycorp.net`` matches ``api.mycorp.net`` but not ``notmycorp.net``).
    """
    cfg = get_active_config()
    allow = getattr(cfg, "propagate_baggage_hosts", ()) if cfg is not None else ()
    if not url or not allow:
        return False
    try:
        # rstrip(".") normalizes a fully-qualified trailing-dot host ("svc.corp.net.")
        # so it matches the allowlist and cannot dodge the "." + entry suffix boundary.
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    return any(host == entry or host.endswith("." + entry) for entry in allow)


def _span_key(span: trace.Span) -> Optional[Tuple[int, int]]:
    """Return the registry key for *span*, or None when its context is invalid."""
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return None
    return (span_context.trace_id, span_context.span_id)


def register_hook_destination(span: trace.Span, url: Optional[str]) -> None:
    """Record *url* as the destination of the request *span* is about to inject for.

    Called from upstream instrumentors' request hooks, which run inside the
    client span immediately before ``inject(headers)``.  Spans with an invalid
    context are skipped, so their injection falls back to stripping baggage.
    """
    key = _span_key(span)
    if key is None or not url:
        return
    with _hook_destinations_lock:
        _hook_destinations[key] = url
        _hook_destinations.move_to_end(key)
        while len(_hook_destinations) > _HOOK_DESTINATIONS_MAX:
            _hook_destinations.popitem(last=False)


def _pop_hook_destination(ctx: Context) -> Optional[str]:
    """Consume the destination registered for *ctx*'s current span, if any.

    One-shot: a registration grants baggage to exactly one injection under the
    span it was made for, never to a request under a different span.
    """
    key = _span_key(trace.get_current_span(ctx))
    if key is None:
        return None
    with _hook_destinations_lock:
        return _hook_destinations.pop(key, None)


class HostGatedPropagator(TextMapPropagator):  # type: ignore[misc]
    """Wrap a propagator so baggage is only injected for allowed destinations.

    Baggage is cleared from the context before delegating, rather than the
    header being stripped afterwards, so gating works for any carrier type
    (header dicts, gRPC metadata, environment mappings) and any propagator mix.
    Extraction is delegated unchanged.
    """

    def __init__(self, inner: TextMapPropagator) -> None:
        self._inner = inner

    @property
    def inner(self) -> TextMapPropagator:
        """The wrapped propagator."""
        return self._inner

    def extract(
        self,
        carrier: CarrierT,
        context: Optional[Context] = None,
        getter: Getter[CarrierT] = default_getter,
    ) -> Context:
        return self._inner.extract(carrier, context=context, getter=getter)

    def inject(
        self,
        carrier: CarrierT,
        context: Optional[Context] = None,
        setter: Setter[CarrierT] = default_setter,
    ) -> None:
        # Context subclasses dict, so an empty one is falsy: compare to None.
        ctx = get_current() if context is None else context
        destination = cast(Optional[str], get_value(_DESTINATION_KEY, ctx))
        if destination is None:
            destination = _pop_hook_destination(ctx)
        if destination != LOCAL_DESTINATION and not _host_allowed(destination):
            ctx = baggage.clear(ctx)
        self._inner.inject(carrier, context=ctx, setter=setter)

    @property
    def fields(self) -> Set[str]:
        return set(self._inner.fields)


def install_gated_propagator() -> None:
    """Wrap the current global propagator in a :class:`HostGatedPropagator`.

    Idempotent.  Whatever propagator is configured (``OTEL_PROPAGATORS`` or a
    user's ``set_global_textmap``) is preserved as the inner propagator.
    """
    global _gating_installed
    _gating_installed = True
    current = get_global_textmap()
    if isinstance(current, HostGatedPropagator):
        return
    set_global_textmap(HostGatedPropagator(current))
    logger.debug("Installed host-gated baggage propagator around %s", type(current).__name__)


def ensure_gated_propagator() -> None:
    """Re-wrap the global propagator if something replaced it after ``Netra.init()``.

    Called just before injections Netra can see (its own HTTP wrappers and the
    upstream instrumentors' request hooks), so a later ``set_global_textmap``
    -- another SDK's or the user's -- cannot silently disable baggage gating.
    The replacement is kept as the inner propagator.  A no-op until
    :func:`install_gated_propagator` has run, so Netra never wraps a
    propagator before it is initialised.
    """
    global _replacement_warned
    if not _gating_installed:
        return
    current = get_global_textmap()
    if isinstance(current, HostGatedPropagator):
        return
    set_global_textmap(HostGatedPropagator(current))
    if not _replacement_warned:
        _replacement_warned = True
        logger.warning(
            "The global OpenTelemetry propagator was replaced after Netra.init() (now %s); "
            "re-wrapping it so session baggage stays limited to NETRA_PROPAGATE_BAGGAGE_HOSTS. "
            "Configure propagators before Netra.init() to avoid this.",
            type(current).__name__,
        )


def inject_context(carrier: MutableMapping[str, Any], url: Optional[str] = None) -> None:
    """Inject propagation headers into *carrier*, host-gating session baggage.

    Trace context is always injected.  The W3C ``baggage`` header -- which
    carries Netra's session identity -- is removed unless *url*'s host is on the
    internal allowlist (see :func:`_host_allowed`).

    Args:
        carrier: The header mapping to inject into (mutated in place).
        url: The outbound request URL, used to decide whether baggage may be
            propagated.  When ``None`` the host is treated as untrusted and
            baggage is stripped.
    """
    ensure_gated_propagator()
    # "" (not None) for an unknown URL, so a missing URL never falls through to
    # a hook registration for the current span.
    inject(carrier, context=set_value(_DESTINATION_KEY, url or ""))
    # Defence in depth: also strip a baggage header already present in the
    # carrier, or written by a propagator that ignores the context.
    strip_baggage_unless_allowed(carrier, url)


def strip_baggage_unless_allowed(carrier: MutableMapping[Any, Any], url: Optional[str]) -> None:
    """Remove any ``baggage`` header from *carrier* unless *url*'s host is allowlisted.

    The gated propagator only decides whether baggage is *added*.  HTTP clients
    that follow redirects copy the previous hop's headers onto the next request
    without injecting again, so a ``baggage`` header sent to an allowlisted host
    would otherwise follow a cross-host redirect.  Redirect guards call this
    with each hop's headers and destination.

    Matching is case-insensitive and removes every occurrence, so it works on
    plain dicts, ``httpx.Headers`` and aiohttp's ``CIMultiDict``.
    """
    if _host_allowed(url):
        return
    _strip_baggage(carrier)


def _strip_baggage(carrier: MutableMapping[Any, Any]) -> None:
    """Remove every ``baggage`` header from *carrier*, matching case-insensitively."""
    for key in [k for k in carrier if isinstance(k, str) and k.lower() == _BAGGAGE_HEADER]:
        carrier.pop(key, None)


def guard_redirect_baggage(carrier: MutableMapping[Any, Any], url: Optional[str], client: str) -> None:
    """Redirect-guard entry point: strip baggage for a non-allowlisted hop, failing closed.

    If the allowlist check fails, baggage is removed unconditionally rather
    than sent to an unchecked host.  The failure is logged as a warning once
    per *client*, then at debug, so a persistent fault cannot flood the logs.
    """
    try:
        strip_baggage_unless_allowed(carrier, url)
        return
    except Exception:
        level = logging.DEBUG if client in _redirect_guard_warned else logging.WARNING
        _redirect_guard_warned.add(client)
        logger.log(level, "Failed to gate baggage on %s redirect; stripping it", client, exc_info=True)
    try:
        _strip_baggage(carrier)
    except Exception:
        logger.debug("Failed to strip baggage from %s redirect", client, exc_info=True)


def inject_local(carrier: MutableMapping[Any, Any]) -> None:
    """Inject the full context, baggage included, into a carrier that stays on this host."""
    inject(carrier, context=set_value(_DESTINATION_KEY, LOCAL_DESTINATION))
