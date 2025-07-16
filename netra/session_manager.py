"""
Session management for PromptOps SDK.
Handles automatic session and user ID management for applications.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace

from .config import Config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session and user context for applications."""

    # Class variable to track the current span
    _current_span: Optional[trace.Span] = None

    # Class variables to track separate entity stacks
    _workflow_stack: List[str] = []
    _task_stack: List[str] = []
    _agent_stack: List[str] = []

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

    @classmethod
    def push_entity(cls, entity_type: str, entity_name: str) -> None:
        """
        Push an entity onto the appropriate entity stack.

        Args:
            entity_type: Type of entity (workflow, task, agent)
            entity_name: Name of the entity
        """
        if entity_type == "workflow":
            cls._workflow_stack.append(entity_name)
        elif entity_type == "task":
            cls._task_stack.append(entity_name)
        elif entity_type == "agent":
            cls._agent_stack.append(entity_name)

    @classmethod
    def pop_entity(cls, entity_type: str) -> Optional[str]:
        """
        Pop the most recent entity from the specified entity stack.

        Args:
            entity_type: Type of entity (workflow, task, agent)

        Returns:
            Entity name or None if stack is empty
        """
        if entity_type == "workflow" and cls._workflow_stack:
            return cls._workflow_stack.pop()
        elif entity_type == "task" and cls._task_stack:
            return cls._task_stack.pop()
        elif entity_type == "agent" and cls._agent_stack:
            return cls._agent_stack.pop()
        return None

    @classmethod
    def get_current_entity_attributes(cls) -> Dict[str, str]:
        """
        Get current entity attributes for span annotation.

        Returns:
            Dictionary of entity attributes to add to spans
        """
        attributes = {}

        # Add current workflow if exists
        if cls._workflow_stack:
            attributes[f"{Config.LIBRARY_NAME}.workflow.name"] = cls._workflow_stack[-1]

        # Add current task if exists
        if cls._task_stack:
            attributes[f"{Config.LIBRARY_NAME}.task.name"] = cls._task_stack[-1]

        # Add current agent if exists
        if cls._agent_stack:
            attributes[f"{Config.LIBRARY_NAME}.agent.name"] = cls._agent_stack[-1]

        return attributes

    @classmethod
    def clear_entity_stacks(cls) -> None:
        """Clear all entity stacks."""
        cls._workflow_stack.clear()
        cls._task_stack.clear()
        cls._agent_stack.clear()

    @classmethod
    def get_stack_info(cls) -> Dict[str, List[str]]:
        """
        Get information about all current stacks.

        Returns:
            Dictionary containing all stack contents
        """
        return {
            "workflows": cls._workflow_stack.copy(),
            "tasks": cls._task_stack.copy(),
            "agents": cls._agent_stack.copy(),
        }

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
