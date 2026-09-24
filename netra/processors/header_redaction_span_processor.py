"""Redact sensitive HTTP headers recorded by upstream OpenTelemetry instrumentations.

Netra's own HTTP instrumentations redact headers at the source, through the
sanitizers in :mod:`netra.instrumentation.http.headers`. The upstream OTel
instrumentations (django, flask, falcon, starlette, tornado, asgi, wsgi) do not
know about that list: when a user opts in to header capture with
``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_*`` they record each header as a
``http.request.header.<name>`` / ``http.response.header.<name>`` attribute and
only mask what ``OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SANITIZE_FIELDS``
names, which is empty by default.

This processor applies the same list -- built-in headers plus
``NETRA_REDACT_HEADERS`` -- to those attributes before export, so one list
protects both kinds of span.
"""

import logging
from typing import FrozenSet, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from netra.instrumentation.http.headers import REDACTED, get_sensitive_headers

logger = logging.getLogger(__name__)

# The semantic-convention prefixes the upstream instrumentations record headers
# under (``opentelemetry.util.http.normalise_{request,response}_header_name``).
_HEADER_ATTRIBUTE_PREFIXES = ("http.request.header.", "http.response.header.")


def _normalised_sensitive_headers() -> FrozenSet[str]:
    """Return the sensitive header names in attribute-key form.

    The upstream instrumentations lower-case header names and replace ``-``
    with ``_`` when building the attribute key, so ``x-api-key`` is recorded
    as ``http.request.header.x_api_key``.
    """
    return frozenset(name.replace("-", "_") for name in get_sensitive_headers())


def _header_name(key: str) -> Optional[str]:
    """Return the header part of a header attribute key, or ``None`` for any other key."""
    for prefix in _HEADER_ATTRIBUTE_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


class HeaderRedactionSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Replace sensitive ``http.{request,response}.header.*`` attribute values with ``[REDACTED]``.

    Must be registered before the exporting span processor: every processor's
    ``on_end`` receives the same span, so redacting here is what the exporter
    sees.
    """

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """No-op; header attributes are redacted once the span has ended."""
        return

    def on_end(self, span: ReadableSpan) -> None:
        """Redact sensitive header attributes on the ended span.

        Args:
            span: The ended span.
        """
        try:
            attributes = getattr(span, "_attributes", None)
            if not attributes:
                return

            sensitive = _normalised_sensitive_headers()
            for key in [key for key in attributes if _header_name(key) in sensitive]:
                attributes[key] = (REDACTED,)
        except Exception:
            logger.exception("Error redacting header attributes on span")

    def shutdown(self) -> None:
        """No resources to release."""
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered, so flushing always succeeds."""
        return True
