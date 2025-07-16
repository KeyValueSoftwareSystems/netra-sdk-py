import logging
from typing import Optional

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from netra.config import Config
from netra.session_manager import SessionManager

logger = logging.getLogger(__name__)


class SessionSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """OpenTelemetry span processor that automatically adds session attributes to spans."""

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Add session attributes to span when it starts and store current span."""
        try:
            # Store the current span in SessionManager
            SessionManager.set_current_span(span)

            ctx = otel_context.get_current()
            session_id = baggage.get_baggage("session_id", ctx)
            user_id = baggage.get_baggage("user_id", ctx)
            tenant_id = baggage.get_baggage("tenant_id", ctx)
            custom_keys = baggage.get_baggage("custom_keys", ctx)

            span.set_attribute("library.name", Config.LIBRARY_NAME)
            span.set_attribute("library.version", Config.LIBRARY_VERSION)
            span.set_attribute("sdk.name", Config.SDK_NAME)

            if session_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.session_id", session_id)
            if user_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.user_id", user_id)
            if tenant_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.tenant_id", tenant_id)
            if custom_keys:
                for key in custom_keys.split(","):
                    value = baggage.get_baggage(f"custom.{key}", ctx)
                    if value:
                        span.set_attribute(f"{Config.LIBRARY_NAME}.custom.{key}", value)

            # Add entity attributes from SessionManager
            entity_attributes = SessionManager.get_current_entity_attributes()
            for attr_key, attr_value in entity_attributes.items():
                span.set_attribute(attr_key, attr_value)

        except Exception as e:
            logger.exception(f"Error setting span attributes: {e}")

    def on_end(self, span: trace.Span) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> None:
        pass

    def shutdown(self) -> None:
        pass
