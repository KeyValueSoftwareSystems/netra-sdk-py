"""Request hooks that tell the gated propagator where upstream instrumentors send.

The upstream OpenTelemetry ``urllib``, ``urllib3`` and ``aiohttp_client``
instrumentors call the global ``inject(headers)`` themselves, so the destination never reaches
:class:`~netra.instrumentation.http.propagation.HostGatedPropagator` through
the context.  Each invokes ``request_hook`` with the client span immediately
before injecting under it; these hooks register the request URL against that span so
allowlisted hosts still receive baggage.  Running just before that inject,
they also restore the gated propagator if it was replaced after ``Netra.init()``.

Kept import-light: ``netra.instrumentation.wiring.registry`` names the kwargs
factories below by string and they are only imported on activation.
"""

import logging
from typing import Any

from opentelemetry.trace import Span

from netra.instrumentation.http.propagation import ensure_gated_propagator, register_hook_destination

logger = logging.getLogger(__name__)


def urllib3_request_hook(span: Span, pool: Any, request_info: Any) -> None:
    """Register the destination of a urllib3 request for baggage gating.

    ``request_info.url`` is already absolute: the instrumentor resolves a path
    against the connection pool before invoking the hook.
    """
    try:
        ensure_gated_propagator()
        register_hook_destination(span, getattr(request_info, "url", None))
    except Exception:
        logger.debug("Failed to register urllib3 destination for baggage gating", exc_info=True)


def urllib_request_hook(span: Span, request: Any) -> None:
    """Register the destination of a urllib request for baggage gating."""
    try:
        ensure_gated_propagator()
        register_hook_destination(span, getattr(request, "full_url", None))
    except Exception:
        logger.debug("Failed to register urllib destination for baggage gating", exc_info=True)


def aiohttp_request_hook(span: Span, params: Any) -> None:
    """Register the destination of an aiohttp request for baggage gating.

    ``params.url`` is already resolved against the session's ``base_url``.
    """
    try:
        ensure_gated_propagator()
        url = getattr(params, "url", None)
        register_hook_destination(span, str(url) if url is not None else None)
    except Exception:
        logger.debug("Failed to register aiohttp destination for baggage gating", exc_info=True)


def urllib3_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``URLLib3Instrumentor().instrument()``."""
    return {"request_hook": urllib3_request_hook}


def urllib_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``URLLibInstrumentor().instrument()``."""
    return {"request_hook": urllib_request_hook}


def aiohttp_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``AioHttpClientInstrumentor().instrument()``."""
    return {"request_hook": aiohttp_request_hook}
