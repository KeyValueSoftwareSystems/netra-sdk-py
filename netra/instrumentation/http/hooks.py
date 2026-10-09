"""Request hooks that tell the gated propagator where upstream instrumentors send.

The upstream OpenTelemetry ``urllib``, ``urllib3`` and ``aiohttp_client``
instrumentors call the global ``inject(headers)`` themselves, so the destination never reaches
:class:`~netra.instrumentation.http.propagation.HostGatedPropagator` through
the context.  Each invokes ``request_hook`` with the client span immediately
before injecting under it; these hooks register the request URL against that span so
allowlisted hosts still receive baggage.  Running just before that inject,
they also restore the gated propagator if it was replaced after ``Netra.init()``.

The gate only decides whether baggage is *added*.  When ``urllib`` or
``aiohttp`` follows a redirect, the next hop reuses the previous hop's headers
without injecting again, so the kwargs factories also install redirect guards
that strip ``baggage`` when the redirect target is not allowlisted.

Kept import-light: ``netra.instrumentation.wiring.registry`` names the kwargs
factories below by string and they are only imported on activation.
"""

import logging
import threading
from typing import Any, Callable, Optional

from opentelemetry.trace import Span
from wrapt import wrap_function_wrapper

from netra.instrumentation.http.propagation import (
    ensure_gated_propagator,
    guard_redirect_baggage,
    register_hook_destination,
)

logger = logging.getLogger(__name__)

_urllib_redirect_guard_lock = threading.Lock()
_urllib_redirect_guarded = False


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


def _urllib_redirect_request(wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any) -> Any:
    """Strip baggage from a urllib redirect request whose target is not allowlisted.

    ``HTTPRedirectHandler.redirect_request`` copies the original request's
    headers, including an injected ``baggage``, onto the request for the next
    hop, and that hop is not instrumented again.
    """
    new_request = wrapped(*args, **kwargs)
    if new_request is not None:
        guard_redirect_baggage(new_request.headers, getattr(new_request, "full_url", None), "urllib")
    return new_request


def _guard_urllib_redirects() -> None:
    """Install the urllib redirect guard once per process."""
    global _urllib_redirect_guarded
    with _urllib_redirect_guard_lock:
        if _urllib_redirect_guarded:
            return
        wrap_function_wrapper("urllib.request", "HTTPRedirectHandler.redirect_request", _urllib_redirect_request)
        _urllib_redirect_guarded = True


async def _aiohttp_on_request_redirect(_session: Any, _trace_config_ctx: Any, params: Any) -> None:
    """Strip baggage before aiohttp follows a redirect to a host that is not allowlisted.

    aiohttp fires ``on_request_start`` (where the instrumentor injects) once,
    before its redirect loop, and reuses the same header mapping for every hop;
    ``params.headers`` is that mapping.  A missing or unparseable ``Location``
    is treated as an unknown destination, so baggage is stripped.
    """
    next_url: Optional[str] = None
    try:
        location = params.response.headers.get("Location") or params.response.headers.get("URI")
        if location:
            from yarl import URL

            next_url = str(params.url.join(URL(location)))
    except Exception:
        next_url = None  # unknown destination: strip
    headers = getattr(params, "headers", None)
    if headers is not None:
        guard_redirect_baggage(headers, next_url, "aiohttp")


def _aiohttp_redirect_trace_config() -> Any:
    """An ``aiohttp.TraceConfig`` carrying the redirect guard."""
    import aiohttp

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_redirect.append(_aiohttp_on_request_redirect)
    return trace_config


def urllib3_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``URLLib3Instrumentor().instrument()``."""
    return {"request_hook": urllib3_request_hook}


def urllib_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``URLLibInstrumentor().instrument()``; also guards urllib redirects."""
    _guard_urllib_redirects()
    return {"request_hook": urllib_request_hook}


def aiohttp_instrument_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``AioHttpClientInstrumentor().instrument()``, redirect guard included."""
    return {"request_hook": aiohttp_request_hook, "trace_configs": [_aiohttp_redirect_trace_config()]}
