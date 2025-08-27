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
    _span_stack: List[str] = []

    # Span registry: name -> stack of spans (most-recent last)
    _spans_by_name: Dict[str, List[trace.Span]] = {}

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
    def register_span(cls, name: str, span: trace.Span) -> None:
        """
        Register a span under a given name. Supports nested spans with the same name via a stack.
        """
        try:
            stack = cls._spans_by_name.get(name)
            if stack is None:
                cls._spans_by_name[name] = [span]
            else:
                stack.append(span)
        except Exception:
            logger.exception("Failed to register span '%s'", name)

    @classmethod
    def unregister_span(cls, name: str, span: trace.Span) -> None:
        """
        Unregister a span for a given name. Safe if not present.
        """
        try:
            stack = cls._spans_by_name.get(name)
            if not stack:
                return
            # Remove the last matching instance (normal case)
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] is span:
                    stack.pop(i)
                    break
            if not stack:
                cls._spans_by_name.pop(name, None)
        except Exception:
            logger.exception("Failed to unregister span '%s'", name)

    @classmethod
    def get_span_by_name(cls, name: str) -> Optional[trace.Span]:
        """
        Get the most recently registered span with the given name.
        """
        stack = cls._spans_by_name.get(name)
        if stack:
            return stack[-1]
        return None

    @classmethod
    def push_entity(cls, entity_type: str, entity_name: str) -> None:
        """
        Push an entity onto the appropriate entity stack.

        Args:
            entity_type: Type of entity (workflow, task, agent, span)
            entity_name: Name of the entity
        """
        if entity_type == "workflow":
            cls._workflow_stack.append(entity_name)
        elif entity_type == "task":
            cls._task_stack.append(entity_name)
        elif entity_type == "agent":
            cls._agent_stack.append(entity_name)
        elif entity_type == "span":
            cls._span_stack.append(entity_name)

    @classmethod
    def pop_entity(cls, entity_type: str) -> Optional[str]:
        """
        Pop the most recent entity from the specified entity stack.

        Args:
            entity_type: Type of entity (workflow, task, agent, span)

        Returns:
            Entity name or None if stack is empty
        """
        if entity_type == "workflow" and cls._workflow_stack:
            return cls._workflow_stack.pop()
        elif entity_type == "task" and cls._task_stack:
            return cls._task_stack.pop()
        elif entity_type == "agent" and cls._agent_stack:
            return cls._agent_stack.pop()
        elif entity_type == "span" and cls._span_stack:
            return cls._span_stack.pop()
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

        # Add current span if exists
        if cls._span_stack:
            attributes[f"{Config.LIBRARY_NAME}.span.name"] = cls._span_stack[-1]

        return attributes

    @classmethod
    def clear_entity_stacks(cls) -> None:
        """Clear all entity stacks."""
        cls._workflow_stack.clear()
        cls._task_stack.clear()
        cls._agent_stack.clear()
        cls._span_stack.clear()

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
            "spans": cls._span_stack.copy(),
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

    @classmethod
    def set_attribute_on_target_span(cls, attr_key: str, attr_value: Any, span_name: Optional[str] = None) -> None:
        """
        Best-effort setter to annotate the active span with the provided attribute.

        If span_name is provided, we look up that span via SessionManager's registry and set
        the attribute on that span explicitly. Otherwise, we annotate the active span.
        We first try the OpenTelemetry current span; if that's invalid, we fall back to
        the SDK-managed current span from `SessionManager`.
        """
        try:
            # Convert attribute value to a JSON-safe string representation
            try:
                if isinstance(attr_value, str):
                    attr_str = attr_value
                else:
                    import json

                    attr_str = json.dumps(attr_value)
            except Exception:
                attr_str = str(attr_value)

            # If a target span name is provided, use the registry for explicit lookup
            if span_name is not None:
                target = cls.get_span_by_name(span_name)
                if target is None:
                    logger.debug("No span found with name '%s' to set attribute %s", span_name, attr_key)
                    return
                target.set_attribute(attr_key, attr_str)
                return

            # Otherwise annotate the active span
            current_span = trace.get_current_span()
            has_valid_current = getattr(current_span, "is_recording", None) is not None and current_span.is_recording()
            candidate = current_span if has_valid_current else cls.get_current_span()
            if candidate is None:
                logger.debug("No active span found to set attribute %s", attr_key)
                return
            candidate.set_attribute(attr_key, attr_str)
        except Exception as e:
            logger.exception("Failed setting attribute %s: %s", attr_key, e)
