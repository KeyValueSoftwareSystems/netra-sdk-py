import logging
from typing import Optional

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from netra.config import Config
from netra.session_manager import ATTR_SESSION_ID, ATTR_TENANT_ID, ATTR_USER_ID, SessionManager

logger = logging.getLogger(__name__)


def _resolve_baggage(
    key: str,
    current_context: otel_context.Context,
    parent_context: Optional[otel_context.Context],
) -> Optional[str]:
    """Read one baggage value, falling back to the span's declared parent context.

    A span started with an explicit ``context=`` is *not* created inside that
    context: the SDK fires ``on_start`` before making it current, so the ambient
    context here belongs to whichever task happened to create the span, which may
    carry no session baggage at all. LiveKit does exactly this — every
    ``agent_turn`` is parented onto a context snapshotted when the session started
    — so a turn triggered from outside the session's task tree would otherwise be
    the one span in the trace missing ``netra.session_id``.

    The fallback is per key and ambient-first, so it can only add a value that was
    missing, never change one that was already resolved: ``Netra.set_session_id()``
    is process-wide and may be called at any point, and it must still win over a
    parent context that was snapshotted earlier.

    Args:
        key: The baggage key to read.
        current_context: The ambient context, which takes precedence.
        parent_context: The parent context the SDK passed to ``on_start``, if any.

    Returns:
        The baggage value, or ``None`` when neither context carries *key* as a
        non-empty string. Non-string values are skipped rather than returned:
        W3C baggage is string-valued, every writer in this SDK sets strings, and
        both consumers here — ``Span.set_attribute`` and ``custom_keys.split`` —
        accept nothing else.
    """
    for context in (current_context, parent_context):
        if context is None:
            continue
        value = baggage.get_baggage(key, context)
        if isinstance(value, str) and value:
            return value
    return None


class SessionSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """OpenTelemetry span processor that automatically adds session attributes to spans."""

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """
        Add session attributes to span when it starts and store current span.

        Args:
            span: The span to start.
            parent_context: The parent context of the span. Consulted for session
                baggage the ambient context does not carry — see ``_resolve_baggage``.
        """
        try:
            # Store the current span in SessionManager
            SessionManager.set_current_span(span)

            ctx = otel_context.get_current()
            session_id = _resolve_baggage("session_id", ctx, parent_context)
            user_id = _resolve_baggage("user_id", ctx, parent_context)
            tenant_id = _resolve_baggage("tenant_id", ctx, parent_context)
            custom_keys = _resolve_baggage("custom_keys", ctx, parent_context)

            span.set_attribute("library.name", Config.LIBRARY_NAME)
            span.set_attribute("library.version", Config.LIBRARY_VERSION)
            span.set_attribute("sdk.name", Config.SDK_NAME)

            if session_id:
                span.set_attribute(ATTR_SESSION_ID, session_id)
            if user_id:
                span.set_attribute(ATTR_USER_ID, user_id)
            if tenant_id:
                span.set_attribute(ATTR_TENANT_ID, tenant_id)
            if custom_keys:
                for key in custom_keys.split(","):
                    value = _resolve_baggage(f"custom.{key}", ctx, parent_context)
                    if value:
                        span.set_attribute(f"{Config.LIBRARY_NAME}.custom.{key}", value)

            # Add entity attributes from SessionManager
            entity_attributes = SessionManager.get_current_entity_attributes()
            for attr_key, attr_value in entity_attributes.items():
                span.set_attribute(attr_key, attr_value)

        except Exception as e:
            logger.exception(f"Error setting span attributes: {e}")

    def on_end(self, span: trace.Span) -> None:
        """
        End span.

        Args:
            span: The span to end.
        """
        return

    def force_flush(self, timeout_millis: int = 30000) -> None:
        """
        Force flush span.

        Args:
            timeout_millis: The timeout in milliseconds.
        """
        return

    def shutdown(self) -> None:
        """
        Shutdown the processor.
        """
        return
