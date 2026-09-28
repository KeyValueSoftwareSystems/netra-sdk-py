"""Host-scoped context propagation for Netra's HTTP instrumentation.

Netra stores the session identity (``session_id``/``user_id``/``tenant_id`` and
any custom session keys) as W3C baggage.  The default OpenTelemetry global
propagator serialises baggage into a ``baggage:`` header on *every* outbound
request, so a bare :func:`opentelemetry.propagate.inject` leaks that identity to
third-party APIs (LLM providers included).

:func:`inject_context` is the single injection entry point used by all of
Netra's HTTP client wrappers.  It injects the full context as before (trace
context, and whatever else the global propagator carries) and then removes the
``baggage`` header entirely unless the destination host is on an explicit
internal allowlist.  Trace context (``traceparent``/``tracestate``) is always
propagated; the whole W3C ``baggage`` header is host-gated -- Netra's session
identity lives there, so gating the header is what closes the leak, and any
other baggage a caller may have set is gated with it rather than being allowed
to reach third parties.

The allowlist is fail-closed: with no hosts configured, baggage propagates to
*nobody*, which closes the leak with zero configuration.  Internal services are
opted back in via the ``NETRA_PROPAGATE_BAGGAGE_HOSTS`` environment variable.
"""

import logging
from typing import MutableMapping, Optional
from urllib.parse import urlparse

from opentelemetry.propagate import inject

from netra.config import get_active_config

logger = logging.getLogger(__name__)

# OpenTelemetry's W3CBaggagePropagator always writes this (lowercase) header.
# We match case-insensitively when stripping, to be safe against carriers that
# normalise header casing differently.
_BAGGAGE_HEADER = "baggage"


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


def inject_context(carrier: MutableMapping[str, str], url: Optional[str] = None) -> None:
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
    inject(carrier)
    if _host_allowed(url):
        return
    # Strip session-identity baggage for untrusted/unknown hosts.
    for key in [k for k in carrier if isinstance(k, str) and k.lower() == _BAGGAGE_HEADER]:
        carrier.pop(key, None)
