"""
Session management for PromptOps SDK.
Handles automatic session and user ID management for applications.
"""

from opentelemetry import context as otel_context
from opentelemetry import baggage
from opentelemetry.sdk.trace import SpanProcessor


class SessionManager:
    """Manages session and user context for applications."""

    @staticmethod
    def set_session_context(session_key: str, value: str) -> None:
        """
        Set session context attributes in the current OpenTelemetry baggage.

        Args:
            session_key: Key to set in baggage (session_id, user_id, user_account_id, or custom_attributes)
            value: Value to set for the key
        """
        ctx = otel_context.get_current()
        if session_key == "session_id":
            ctx = baggage.set_baggage("session_id", value, ctx)
        elif session_key == "user_id":
            ctx = baggage.set_baggage("user_id", value, ctx)
        elif session_key == "user_account_id":
            ctx = baggage.set_baggage("user_account_id", value, ctx)
        elif session_key == "custom_attributes":
            custom_keys = list(value.keys())
            ctx = baggage.set_baggage("custom_keys", ",".join(custom_keys), ctx)
            for key, value in value.items():
                ctx = baggage.set_baggage(f"custom.{key}", str(value), ctx)
        otel_context.attach(ctx)


class SessionSpanProcessor(SpanProcessor):
    """OpenTelemetry span processor that automatically adds session attributes to spans."""

    def on_start(self, span, parent_context):
        """Add session attributes to span when it starts."""
        ctx = otel_context.get_current()
        session_id = baggage.get_baggage("session_id", ctx)
        user_id = baggage.get_baggage("user_id", ctx)
        user_account_id = baggage.get_baggage("user_account_id", ctx)
        custom_keys = baggage.get_baggage("custom_keys", ctx)

        if session_id:
            span.set_attribute("combat.session_id", session_id)
        if user_id:
            span.set_attribute("combat.user_id", user_id)
        if user_account_id:
            span.set_attribute("combat.user_account_id", user_account_id)
        if custom_keys:
            for key in custom_keys.split(","):
                value = baggage.get_baggage(f"custom.{key}", ctx)
                if value:
                    span.set_attribute(f"combat.custom.{key}", value)

    def on_end(self, span):
        pass

    def force_flush(self, timeout_millis=30000):
        pass

    def shutdown(self):
        pass
