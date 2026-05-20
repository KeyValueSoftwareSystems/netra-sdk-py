import logging
import os
from typing import Dict, Optional

from opentelemetry import context as context_api
from opentelemetry import propagate, trace

logger = logging.getLogger(__name__)


def inject_subprocess_context(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of *env* (or os.environ) with the current OTel trace context injected.

    Thread-safe: reads the calling thread's ContextVar, writes into a fresh dict.

    Args:
        env: The environment dict passed to subprocess.Popen. When None, a copy
            of os.environ is used so the caller's env is not mutated.

    Returns:
        A new dict containing all entries from *env* (or os.environ) plus the
        W3C ``traceparent`` (and ``tracestate``, if present) for the currently
        active span. Returns the dict unchanged if there is no active span.
    """
    carrier = dict(env) if env is not None else dict(os.environ)
    propagate.inject(carrier)
    traceparent = carrier.get("traceparent")
    if traceparent:
        logger.debug("Injecting traceparent into subprocess env: %s", traceparent)
    else:
        logger.debug("No active span; subprocess env forwarded without traceparent.")
    return carrier


def extract_subprocess_context() -> Optional[context_api.Token]:
    """Extract the W3C trace context from os.environ and attach it as the current context.

    Intended to be called once during SDK initialisation in a child process. Reads
    the ``traceparent`` (and ``tracestate``) values written by the parent's
    ``inject_subprocess_context`` and attaches the recovered context so that all
    subsequent spans become children of the parent's active span.

    Returns:
        An OTel context token representing the attached parent process trace context,
        or ``None`` if no ``traceparent`` was found in the environment or the
        extracted context was invalid.
    """
    try:
        raw = os.environ.get("traceparent")
        if not raw:
            return None

        carrier = {k.lower(): v for k, v in os.environ.items()}
        ctx = propagate.extract(carrier)
        span_ctx = trace.get_current_span(ctx).get_span_context()

        if not span_ctx.is_valid:
            logger.warning("Found traceparent in environment (%s) but extracted context is invalid; ignoring.", raw)
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
