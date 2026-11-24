"""Single source of truth for whether span content may be recorded.

``Config.trace_content`` (``NETRA_TRACE_CONTENT``) is published as the
``TRACELOOP_TRACE_CONTENT`` environment variable so the upstream Traceloop
instrumentors honour it. Netra's own code asks :func:`is_content_capture_enabled`
instead of re-reading that variable, and :mod:`netra.processors.content_gating_span_processor`
strips anything that still slips through before a span is exported.

"Content" means what the application's users and models said or sent: prompts,
completions, function arguments and return values, and HTTP bodies. Metadata
such as model names, token counts, timings, status codes and URLs is not content.
"""

import os
from typing import Optional

from opentelemetry import context as context_api

TRACE_CONTENT_ENV = "TRACELOOP_TRACE_CONTENT"

# Context key the Traceloop instrumentors use to force content capture back on for
# one block of code while it is globally off. Netra honours the same key so a
# single override behaves the same across every instrumentor.
CONTENT_OVERRIDE_CONTEXT_KEY = "override_enable_content_tracing"


def is_content_capture_globally_enabled() -> bool:
    """Return the process-wide setting, ignoring any per-context override."""
    return (os.getenv(TRACE_CONTENT_ENV) or "true").lower() == "true"


def is_content_override_active(context: Optional[context_api.Context] = None) -> bool:
    """Return True if content capture was force-enabled for *context* (default: current)."""
    return bool(context_api.get_value(CONTENT_OVERRIDE_CONTEXT_KEY, context))


def is_content_capture_enabled(context: Optional[context_api.Context] = None) -> bool:
    """Return True if content may be recorded in *context* (default: the current context).

    Content capture is on unless ``trace_content`` is off, and can be force-enabled
    for a block of code through :data:`CONTENT_OVERRIDE_CONTEXT_KEY`.
    """
    return is_content_capture_globally_enabled() or is_content_override_active(context)
