import logging
import os
from typing import Any, Dict, Mapping, Optional

from opentelemetry import context as context_api
from opentelemetry import propagate, trace

logger = logging.getLogger(__name__)


def _log_traceparent(traceparent: Optional[str]) -> None:
    """Log whether a traceparent header was injected."""
    if traceparent:
        logger.debug("Injecting traceparent into subprocess env: %s", traceparent)
    else:
        logger.debug("No active span; subprocess env forwarded without traceparent.")


def inject_subprocess_context(
    env: Optional[Mapping[Any, Any]] = None,
) -> Dict[Any, Any]:
    """Return a copy of *env* (or ``os.environ``) with the current OTel trace context injected.

    Thread-safe: reads the calling thread's ContextVar, writes into a fresh dict.

    Handles both ``str``-keyed and ``bytes``-keyed env mappings.  When the
    original mapping uses ``bytes`` keys the injected W3C trace headers are
    encoded to ``bytes`` as well so that the subprocess env stays type-consistent.

    Args:
        env: The environment mapping passed to ``subprocess.Popen``.  When
            ``None``, a copy of ``os.environ`` is used so the caller's env is
            not mutated.

    Returns:
        A new dict containing all entries from *env* (or ``os.environ``) plus
        the W3C ``traceparent`` (and ``tracestate``, if present) for the
        currently active span.  Returns the dict unchanged if there is no
        active span.
    """
    if env is None:
        carrier: Dict[str, str] = dict(os.environ)
        propagate.inject(carrier)
        _log_traceparent(carrier.get("traceparent"))
        return carrier

    uses_bytes = any(isinstance(k, bytes) for k in env)

    if uses_bytes:
        str_carrier: Dict[str, str] = {}
        propagate.inject(str_carrier)
        result: Dict[Any, Any] = dict(env)
        for key, value in str_carrier.items():
            result[key.encode()] = value.encode()
        _log_traceparent(str_carrier.get("traceparent"))
        return result

    carrier_copy: Dict[Any, Any] = dict(env)
    propagate.inject(carrier_copy)
    _log_traceparent(carrier_copy.get("traceparent"))
    return carrier_copy


def extract_subprocess_context() -> Any:
    """Extract the W3C trace context from ``os.environ`` and attach it as the current context.

    Intended to be called once during SDK initialisation in a child process.
    Reads the ``traceparent`` (and ``tracestate``) values written by the
    parent's :func:`inject_subprocess_context` and attaches the recovered
    context so that all subsequent spans become children of the parent's
    active span.

    Returns:
        An opaque OTel context token (as returned by
        :func:`opentelemetry.context.attach`), or ``None`` if no
        ``traceparent`` was found in the environment or the extracted context
        was invalid.
    """
    try:
        raw = os.environ.get("traceparent")
        if not raw:
            return None

        carrier = {k.lower(): v for k, v in os.environ.items()}
        ctx = propagate.extract(carrier)
        span_ctx = trace.get_current_span(ctx).get_span_context()

        if not span_ctx.is_valid:
            logger.warning(
                "Found traceparent in environment (%s) but extracted context is invalid; ignoring.",
                raw,
            )
            return None

        token = context_api.attach(ctx)
        logger.debug(
            "Restored parent process trace context — trace_id=%s span_id=%s",
            format(span_ctx.trace_id, "032x"),
            format(span_ctx.span_id, "016x"),
        )
        return token
    except Exception as e:
        logger.error("Failed to extract parent process trace context from environment: %s", e)
        return None
