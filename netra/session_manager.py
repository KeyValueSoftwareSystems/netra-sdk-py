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

from .config import Config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session and user context for applications."""

    # Class variable to track the current span
    _current_span: Optional[trace.Span] = None

    @classmethod
    def set_current_span(cls, span: Optional[trace.Span]) -> None:
        """
        Set the current span for the session manager.

        Args:
            span: The current span to store
        """
        cls._current_span = span

    @classmethod
    def get_current_span(cls) -> Optional[trace.Span]:
        """
        Get the current span.

        Returns:
            The stored current span or None if not set
        """
        return cls._current_span

    @staticmethod
    def set_session_context(session_key: str, value: Union[str, Dict[str, str]]) -> None:
        """
        Set session context attributes in the current OpenTelemetry baggage.

        Args:
            session_key: Key to set in baggage (session_id, user_id, tenant_id, or custom_attributes)
            value: Value to set for the key
        """
        try:
            ctx = otel_context.get_current()
            if isinstance(value, str) and value:
                if session_key == "session_id":
                    ctx = baggage.set_baggage("session_id", value, ctx)
                elif session_key == "user_id":
                    ctx = baggage.set_baggage("user_id", value, ctx)
                elif session_key == "tenant_id":
                    ctx = baggage.set_baggage("tenant_id", value, ctx)
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
            current_span = SessionManager.get_current_span()
            timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

            if current_span:
                # Set the event in the current span.
                current_span.add_event(name=name, attributes=attributes, timestamp=timestamp_ns)
            else:
                # Fallback to creating a new span.
                ctx = otel_context.get_current()
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(f"{Config.LIBRARY_NAME}.{name}", context=ctx) as span:
                    span.add_event(name=name, attributes=attributes, timestamp=timestamp_ns)
        except Exception as e:
            logger.exception(f"Failed to add custom event: {name} - {e}")
