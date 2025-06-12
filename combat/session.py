"""
Session management for PromptOps SDK.
Handles automatic session and user ID management for applications.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import get_current_span

from .config import Config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session and user context for applications."""

    @staticmethod
    def set_session_context(session_key: str, value: Union[str, Dict[str, str]]) -> None:
        """
        Set session context attributes in the current OpenTelemetry baggage.

        Args:
            session_key: Key to set in baggage (session_id, user_id, user_account_id, or custom_attributes)
            value: Value to set for the key
        """
        try:
            ctx = otel_context.get_current()
            if isinstance(value, str) and value:
                if session_key == "session_id":
                    ctx = baggage.set_baggage("session_id", value, ctx)
                elif session_key == "user_id":
                    ctx = baggage.set_baggage("user_id", value, ctx)
                elif session_key == "user_account_id":
                    ctx = baggage.set_baggage("user_account_id", value, ctx)
            elif isinstance(value, dict) and value:
                if session_key == "custom_attributes":
                    custom_keys = list(value.keys())
                    ctx = baggage.set_baggage("custom_keys", ",".join(custom_keys), ctx)
                    for key, val in value.items():
                        ctx = baggage.set_baggage(f"custom.{key}", str(val), ctx)
            otel_context.attach(ctx)
        except Exception as e:
            logger.exception(f"Failed to set session context for key={session_key}: {e}")

    @staticmethod
    def set_custom_event(name: str, attributes: Dict[str, Any]) -> None:
        """
        Add an event to the current span.

        Args:
            name: Name of the event (e.g., 'pii_detection', 'error', etc.)
            attributes: Dictionary of attributes associated with the event
        """
        try:
            current_span = get_current_span()
            timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

            if not current_span or not current_span.is_recording():
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(f"{Config.LIBRARY_NAME}.{name}") as span:
                    span.add_event(name=name, attributes=attributes, timestamp=timestamp_ns)
            else:
                # Add event to current span
                current_span.add_event(name=name, attributes=attributes, timestamp=timestamp_ns)
        except Exception as e:
            logger.exception(f"Failed to add custom event: {name} - {e}")


class SessionSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """OpenTelemetry span processor that automatically adds session attributes to spans."""

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Add session attributes to span when it starts."""
        try:
            ctx = otel_context.get_current()
            session_id = baggage.get_baggage("session_id", ctx)
            user_id = baggage.get_baggage("user_id", ctx)
            user_account_id = baggage.get_baggage("user_account_id", ctx)
            custom_keys = baggage.get_baggage("custom_keys", ctx)

            span.set_attribute("library.name", Config.LIBRARY_NAME)
            span.set_attribute("library.version", Config.LIBRARY_VERSION)
            span.set_attribute("sdk.name", Config.SDK_NAME)

            if session_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.session_id", session_id)
            if user_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.user_id", user_id)
            if user_account_id:
                span.set_attribute(f"{Config.LIBRARY_NAME}.user_account_id", user_account_id)
            if custom_keys:
                for key in custom_keys.split(","):
                    value = baggage.get_baggage(f"custom.{key}", ctx)
                    if value:
                        span.set_attribute(f"{Config.LIBRARY_NAME}.custom.{key}", value)
        except Exception as e:
            logger.exception(f"Error setting span attributes: {e}")

    def on_end(self, span: trace.Span) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> None:
        pass

    def shutdown(self) -> None:
        pass
