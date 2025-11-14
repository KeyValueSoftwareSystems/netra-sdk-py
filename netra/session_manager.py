"""
Session management for PromptOps SDK.
Handles automatic session and user ID management for applications.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace

from netra.config import Config
from netra.utils import process_content_for_max_len

logger = logging.getLogger(__name__)


class ConversationType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


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

    # Global stack of active spans in creation order (oldest first, newest last)
    # Maintained for spans registered via SessionManager (e.g., SpanWrapper)
    _active_spans: List[trace.Span] = []

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
            # Track globally as active
            cls._active_spans.append(span)
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
            # Also remove from global active list (remove last matching instance)
            for i in range(len(cls._active_spans) - 1, -1, -1):
                if cls._active_spans[i] is span:
                    cls._active_spans.pop(i)
                    break
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
    def set_session_context(
        session_key: str,
        value: Union[str, Dict[str, str]],
        attach_globally: bool = False,
    ) -> None:
        """
        Set session context attributes in OpenTelemetry baggage.

        Behavior:
        - Adds values to baggage on the current context.
        - To avoid context token mismatch errors, this method only attaches the
          modified context when it is safe (no active recording span), unless
          forced via attach_globally=True.
        - When not attaching (e.g., inside an active span), it annotates the
          current span with equivalent attributes for visibility.

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

    @staticmethod
    def add_conversation(conversation_type: ConversationType, role: str, content: Any) -> None:
        """
        Append a conversation entry and set span attribute 'conversation' as an array.

        Stored attribute format:
        conversation: [
          { "type": "input/output/system", "field_name": "sample_name", "value": "sample_value" },
          ...
        ]
        """

        # Hard runtime validation of input types and values
        if not isinstance(conversation_type, ConversationType):
            logger.error(
                "add_conversation: conversation_type must be a ConversationType enum value (input, output, system)"
            )
            return
        normalized_type = conversation_type.value

        if not isinstance(role, str):
            logger.error("add_conversation: role must be a string")
            return

        if not isinstance(content, (str, dict)):
            logger.error("add_conversation: content must be a string or dict")
            return

        if not role:
            logger.error("add_conversation: role must be a non-empty string")
            return

        if not content:
            logger.error("add_conversation: content must not be empty")
            return

        try:

            # Get active recording span
            span = trace.get_current_span()
            if not (span and getattr(span, "is_recording", lambda: False)()):
                logger.warning("No active span to add conversation attribute.")
                return

            # Load existing conversation (JSON string -> list)
            existing: List[Dict[str, Any]] = []
            raw_data = None

            try:
                attrs = getattr(span, "_attributes", None)
                if attrs is not None and hasattr(attrs, "get"):
                    raw_data = attrs.get("conversation")
            except Exception:
                logger.exception("Failed to retrieve conversation attribute")

            if raw_data:
                try:
                    import json

                    parsed: Any = None
                    if isinstance(raw_data, str):
                        parsed = json.loads(raw_data)
                    if isinstance(parsed, list):
                        existing = parsed
                except Exception:
                    existing = []

            # Enforce per-entry content length limit without breaking the entire conversation structure
            max_len = Config.CONVERSATION_MAX_LEN
            processed_content = process_content_for_max_len(content, max_len)

            # Create a conversation entry
            entry: Dict[str, Any] = {"type": normalized_type, "role": role, "content": processed_content}

            # Add format based on processed value type for backend parsing
            if isinstance(processed_content, str):
                entry["format"] = "text"
            elif isinstance(processed_content, dict):
                entry["format"] = "json"
            existing.append(entry)

            # Bypass global attribute value truncation by writing directly to the span's
            # private attribute store. We intentionally avoid span.set_attribute here.
            try:
                import json

                payload = json.dumps(existing, default=str)
                attrs = getattr(span, "_attributes", None)
                attrs["conversation"] = payload  # type: ignore[index]
            except Exception:
                logger.exception("Failed to set conversation attribute directly on span")
        except Exception as e:
            logger.exception("Failed to add conversation attribute: %s", e)

    @staticmethod
    def set_attribute_on_active_span(attr_key: str, attr_value: Any) -> None:
        """
        Set an attribute strictly on the currently active OpenTelemetry span.

        - Does not fall back to any root span.
        - Does not mutate global baggage/context.
        - If no active recording span is present, logs a warning and returns.
        """
        try:
            span = trace.get_current_span()
            if span and getattr(span, "is_recording", lambda: False)():
                # Convert attr_value to a JSON-safe string if needed
                try:
                    if isinstance(attr_value, str):
                        v = attr_value
                    else:
                        import json

                        v = json.dumps(attr_value)
                except Exception:
                    v = str(attr_value)
                span.set_attribute(attr_key, v)
            else:
                logger.warning("No active span to set attribute '%s'", attr_key)
        except Exception:
            logger.exception("Failed to set attribute '%s' on active span", attr_key)
