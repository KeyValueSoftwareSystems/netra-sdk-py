import contextvars
import logging
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace

from netra.config import Config, get_conversation_max_len
from netra.utils import process_content_for_max_len, serialize_value

logger = logging.getLogger(__name__)


NETRA_USER_INPUT = "netra.user.input"
NETRA_USER_OUTPUT = "netra.user.output"

# Canonical span-attribute keys for session identity fields.
# Used by both set_session_context (active span) and SessionSpanProcessor (descendant spans).
ATTR_SESSION_ID = f"{Config.LIBRARY_NAME}.session_id"
ATTR_USER_ID = f"{Config.LIBRARY_NAME}.user_id"
ATTR_TENANT_ID = f"{Config.LIBRARY_NAME}.tenant_id"

_SESSION_ATTR_KEYS: Dict[str, str] = {
    "session_id": ATTR_SESSION_ID,
    "user_id": ATTR_USER_ID,
    "tenant_id": ATTR_TENANT_ID,
}

# Entity stacks live in the OpenTelemetry *context* (not plain class variables
# or ContextVars) so they propagate across thread boundaries: the threading
# instrumentation re-attaches the active OTel context inside worker threads,
# which lets spans created there inherit the parent workflow's entity names.
#
# Each key maps to an immutable tuple of frames, where a frame is
# ``(frame_id, entity_name)``. ``push_entity`` appends a frame tagged with a
# unique ``frame_id`` and returns that id as the token; ``pop_entity`` removes
# the frame with the matching id. This is deliberately NOT modelled as OTel
# ``attach``/``detach``: detach requires strict LIFO ordering, which the
# deferred pop of streaming/generator spans cannot guarantee (a generator may
# be finished out of creation order, or abandoned entirely). Removing a frame
# by id is order-independent, so interleaved or partially-consumed generators
# cannot corrupt or leak another entity's context.
_ENTITY_STACK_KEYS: Dict[str, str] = {
    "workflow": "netra.workflow_stack",
    "task": "netra.task_stack",
    "agent": "netra.agent_stack",
    "span": "netra.span_stack",
}

# A single frame on an entity stack: an opaque per-push id plus the entity name.
_EntityFrame = Tuple[object, str]

# Current span and span registries are per-thread execution bookkeeping that
# must NOT be shared across threads. ContextVars give thread-isolated,
# copy-on-write storage (rebind, never mutate-in-place) so concurrent workers
# cannot corrupt or observe each other's state.
_current_span_var: "contextvars.ContextVar[Optional[trace.Span]]" = contextvars.ContextVar(
    "netra_current_span", default=None
)
_spans_by_name_var: "contextvars.ContextVar[Dict[str, Tuple[trace.Span, ...]]]" = contextvars.ContextVar(
    "netra_spans_by_name", default={}
)
_active_spans_var: "contextvars.ContextVar[Tuple[trace.Span, ...]]" = contextvars.ContextVar(
    "netra_active_spans", default=()
)


def _read_entity_frames(entity_type: str) -> Tuple[_EntityFrame, ...]:
    """Read the current entity-stack frames for ``entity_type`` from the OTel context.

    The stack is stored under the OTel context key mapped in
    ``_ENTITY_STACK_KEYS`` as an immutable tuple of ``(frame_id, name)`` frames.
    Reading tolerates a missing or malformed value (returns an empty stack)
    because the key may be absent in a freshly propagated worker context.

    Args:
        entity_type: The entity kind to read (``"workflow"``, ``"task"``,
            ``"agent"`` or ``"span"``).

    Returns:
        Tuple[_EntityFrame, ...]: The frames oldest-first (top of stack last).
        Empty tuple if ``entity_type`` is unknown or no frames are set.
    """
    key = _ENTITY_STACK_KEYS.get(entity_type)
    if key is None:
        return ()
    value = otel_context.get_value(key)
    return cast(Tuple[_EntityFrame, ...], value) if isinstance(value, tuple) else ()


def _current_entity_name(entity_type: str) -> Optional[str]:
    """Return the name on top of the entity stack for ``entity_type``.

    Args:
        entity_type: The entity kind to inspect (``"workflow"``, ``"task"``,
            ``"agent"`` or ``"span"``).

    Returns:
        Optional[str]: The most recently pushed (still-open) entity name, or
        ``None`` if the stack is empty or ``entity_type`` is unknown.
    """
    frames = _read_entity_frames(entity_type)
    return frames[-1][1] if frames else None


# The baggage keys that carry session identity. ``SessionSpanProcessor.on_start``
# reads exactly these names off the ambient context, so every writer must go
# through ``_build_session_context`` rather than calling ``set_baggage`` inline —
# otherwise the global setter and the scoped attach can drift apart. Derived from
# _SESSION_ATTR_KEYS so the baggage keys and the span-attribute keys stay in step.
_SESSION_BAGGAGE_KEYS: Tuple[str, ...] = tuple(_SESSION_ATTR_KEYS)


def _build_session_context(
    ctx: otel_context.Context,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Optional[otel_context.Context]:
    """Return *ctx* with the supplied session fields set as baggage.

    Args:
        ctx: The context to derive from. Not mutated.
        session_id: Session identifier, or ``None`` to leave unset.
        user_id: User identifier, or ``None`` to leave unset.
        tenant_id: Tenant identifier, or ``None`` to leave unset.

    Returns:
        A new ``Context`` carrying the supplied fields, or ``None`` when every
        field was ``None`` or empty — signalling that there is nothing to attach.
    """
    values = {"session_id": session_id, "user_id": user_id, "tenant_id": tenant_id}
    changed = False
    for key in _SESSION_BAGGAGE_KEYS:
        value = values[key]
        if isinstance(value, str) and value:
            ctx = baggage.set_baggage(key, value, ctx)
            changed = True
    return ctx if changed else None


class ConversationType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class SessionManager:
    """Manages session and user context for applications."""

    @classmethod
    def set_current_span(cls, span: Optional[trace.Span]) -> None:
        """
        Set the current span for the session manager.

        Stored in a thread-isolated ContextVar so parallel workers do not
        overwrite each other's current span.

        Args:
            span: The current span to store
        """
        _current_span_var.set(span)

    @classmethod
    def get_current_span(cls) -> Optional[trace.Span]:
        """
        Get the current span.

        Returns:
            The stored current span or None if not set
        """
        return _current_span_var.get()

    @classmethod
    def register_span(cls, name: str, span: trace.Span) -> None:
        """
        Register a span under a given name. Supports nested spans with the same name via a stack.

        Uses copy-on-write on ContextVars (rebind rather than mutate-in-place)
        so registrations are isolated per thread.

        Args:
            name: The name of the span to register
            span: The span to register
        """
        try:
            by_name = _spans_by_name_var.get()
            new_by_name = dict(by_name)
            new_by_name[name] = by_name.get(name, ()) + (span,)
            _spans_by_name_var.set(new_by_name)
            _active_spans_var.set(_active_spans_var.get() + (span,))
        except Exception:
            logger.exception("Failed to register span '%s'", name)

    @classmethod
    def unregister_span(cls, name: str, span: trace.Span) -> None:
        """
        Unregister a span for a given name. Safe if not present.

        Args:
            name: The name of the span to unregister
            span: The span to unregister
        """
        try:
            by_name = _spans_by_name_var.get()
            stack = by_name.get(name)
            if stack:
                # Remove the last matching instance (normal case)
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] is span:
                        remaining = list(stack)
                        del remaining[i]
                        new_by_name = dict(by_name)
                        if remaining:
                            new_by_name[name] = tuple(remaining)
                        else:
                            new_by_name.pop(name, None)
                        _spans_by_name_var.set(new_by_name)
                        break
            # Also remove from active list (remove last matching instance)
            active = _active_spans_var.get()
            for i in range(len(active) - 1, -1, -1):
                if active[i] is span:
                    remaining_active = list(active)
                    del remaining_active[i]
                    _active_spans_var.set(tuple(remaining_active))
                    break
        except Exception:
            logger.exception("Failed to unregister span '%s'", name)

    @classmethod
    def get_trace_id(cls) -> Optional[str]:
        """
        Return the trace ID of the currently active span.

        Returns:
            str: 32-character lowercase hex trace ID, or None if no active span exists.
        """
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
        return None

    @classmethod
    def get_span_by_name(cls, name: str) -> Optional[trace.Span]:
        """
        Get the most recently registered span with the given name.

        Args:
            name: The name of the span to get

        Returns:
            The most recently registered span with the given name, or None if not found
        """
        stack = _spans_by_name_var.get().get(name)
        if stack:
            return stack[-1]
        return None

    @classmethod
    def push_entity(cls, entity_type: str, entity_name: str) -> Optional[object]:
        """
        Push an entity onto the appropriate entity stack.

        The stack is stored in the OpenTelemetry context (so it propagates into
        worker threads). The push appends a uniquely-tagged frame and returns
        that tag as an opaque token. The caller MUST pass the token to
        :meth:`pop_entity`, which removes *that specific frame* — order
        independently, so a deferred/interleaved pop cannot disturb another
        entity's context.

        Args:
            entity_type: Type of entity (workflow, task, agent, span)
            entity_name: Name of the entity

        Returns:
            An opaque token to pass to :meth:`pop_entity`, or ``None`` if
            ``entity_type`` is unknown.
        """
        key = _ENTITY_STACK_KEYS.get(entity_type)
        if key is None:
            return None
        # A fresh object() is a process-unique identity for this exact push, so
        # pop_entity can find and remove this frame even if frames are removed
        # out of order (interleaved streaming spans).
        frame_id: object = object()
        frames = _read_entity_frames(entity_type)
        otel_context.attach(otel_context.set_value(key, frames + ((frame_id, entity_name),)))
        return frame_id

    @classmethod
    def pop_entity(cls, entity_type: str, token: Optional[object] = None) -> Optional[str]:
        """
        Remove the entity frame identified by ``token`` from its stack.

        The frame is located by identity, so it is removed correctly regardless
        of the order in which concurrent/deferred frames are popped. Rebinds the
        entity key in the current OTel context to the reduced stack (it does not
        ``detach``, which would require strict LIFO ordering).

        Args:
            entity_type: Type of entity (workflow, task, agent, span)
            token: The token returned by the matching ``push_entity``. When
                ``None`` (legacy callers), the top frame is removed instead.

        Returns:
            The name of the frame that was removed, or None if nothing matched.
        """
        key = _ENTITY_STACK_KEYS.get(entity_type)
        if key is None:
            return None
        frames = _read_entity_frames(entity_type)
        if not frames:
            return None

        if token is None:
            # Legacy path: no token to match, remove the most recent frame.
            removed_name = frames[-1][1]
            new_frames = frames[:-1]
        else:
            index = next((i for i in range(len(frames) - 1, -1, -1) if frames[i][0] is token), None)
            if index is None:
                # Frame already removed (or belongs to a context we can't see);
                # do not touch the stack.
                return None
            removed_name = frames[index][1]
            new_frames = frames[:index] + frames[index + 1 :]

        otel_context.attach(otel_context.set_value(key, new_frames))
        return removed_name

    @classmethod
    def get_current_entity_attributes(cls) -> Dict[str, str]:
        """
        Get current entity attributes for span annotation.

        Returns:
            Dictionary of entity attributes to add to spans
        """
        attributes = {}

        for entity_type, attr_suffix in (
            ("workflow", "workflow.name"),
            ("task", "task.name"),
            ("agent", "agent.name"),
            ("span", "span.name"),
        ):
            name = _current_entity_name(entity_type)
            if name is not None:
                attributes[f"{Config.LIBRARY_NAME}.{attr_suffix}"] = name

        return attributes

    @classmethod
    def clear_entity_stacks(cls) -> None:
        """Clear all entity stacks in the current context.

        Rebinds every entity stack to empty in the current OTel context. This is
        a blunt reset intended for test isolation, not the normal push/pop
        lifecycle (which removes frames individually by id).
        """
        ctx = otel_context.get_current()
        for key in _ENTITY_STACK_KEYS.values():
            ctx = otel_context.set_value(key, (), ctx)
        otel_context.attach(ctx)

    @classmethod
    def get_stack_info(cls) -> Dict[str, List[str]]:
        """
        Get information about all current stacks.

        Returns:
            Dictionary containing all stack contents
        """
        return {
            "workflows": [name for _, name in _read_entity_frames("workflow")],
            "tasks": [name for _, name in _read_entity_frames("task")],
            "agents": [name for _, name in _read_entity_frames("agent")],
            "spans": [name for _, name in _read_entity_frames("span")],
        }

    @staticmethod
    def set_session_context(
        session_key: str,
        value: Union[str, Dict[str, str]],
        attach_globally: bool = False,
    ) -> None:
        """
        Set a session identity attribute (session_id, user_id, or tenant_id).

        This does two things atomically:
          1. Sets the value as OTel baggage so other spans inherit it via
             ``SessionSpanProcessor.on_start``.
          2. Sets the corresponding span attribute on the currently active span
             (if one exists), so the caller's span also carries the value.

        The attach is deliberately never detached: ``Netra.set_session_id()`` and
        friends are documented as process-sticky, and existing users rely on the
        session id outliving the call that set it. For a scoped session id that
        is restored on exit — what instrumentation wants — use
        :meth:`attach_session_context` or :meth:`session_scope` instead.

        Args:
            session_key: Key to set (``"session_id"``, ``"user_id"``, or ``"tenant_id"``)
            value: Value to set for the key
        """
        try:
            if not (isinstance(value, str) and value):
                return

            attr_key = _SESSION_ATTR_KEYS.get(session_key)
            if attr_key is None:
                return

            # Propagate to descendant spans via baggage
            ctx = _build_session_context(otel_context.get_current(), **{session_key: value})
            if ctx is not None:
                otel_context.attach(ctx)

            # Stamp the active span immediately
            span = trace.get_current_span()
            if span and getattr(span, "is_recording", lambda: False)():
                span.set_attribute(attr_key, value)
        except Exception as e:
            logger.exception(f"Failed to set session context for key={session_key}: {e}")

    @staticmethod
    def attach_session_context(
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[object]:
        """Attach session baggage to the current OTel context and return its token.

        Unlike :meth:`set_session_context`, the caller owns the returned token and
        MUST detach it — in the same context it was attached in, as OTel requires.
        Prefer :meth:`session_scope` where the scope is lexical.

        Args:
            session_id: Session identifier to put in baggage, if any.
            user_id: User identifier to put in baggage, if any.
            tenant_id: Tenant identifier to put in baggage, if any.

        Returns:
            The token to pass to ``opentelemetry.context.detach``, or ``None``
            when no field was supplied — nothing was attached, so there is
            nothing to detach and callers need no emptiness branch.
        """
        ctx = _build_session_context(
            otel_context.get_current(),
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if ctx is None:
            return None
        # Declared as ``object`` so the public signature does not leak OTel's
        # Token generic; callers only ever hand it back to ``otel_context.detach``.
        token: object = otel_context.attach(ctx)
        return token

    @staticmethod
    @contextmanager
    def session_scope(
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Iterator[None]:
        """Scoped form of :meth:`attach_session_context`.

        Detaches on exit, including when the body raises.

        Args:
            session_id: Session identifier to put in baggage, if any.
            user_id: User identifier to put in baggage, if any.
            tenant_id: Tenant identifier to put in baggage, if any.

        Yields:
            None. The session baggage is active for the duration of the block.
        """
        # Attaches directly rather than via attach_session_context() so the token
        # keeps its concrete OTel type here; both paths share _build_session_context,
        # which is what keeps the baggage keys from drifting.
        ctx = _build_session_context(
            otel_context.get_current(),
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        if ctx is None:
            yield
            return
        token = otel_context.attach(ctx)
        try:
            yield
        finally:
            otel_context.detach(token)

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
    def add_conversation(cls, conversation_type: ConversationType, role: str, content: Any) -> None:
        """
        Append a conversation entry and set span attribute 'conversation' as an array.

        Args:
            conversation_type: Type of conversation (input, output, system)
            role: Role of the participant (e.g., 'user', 'assistant', 'system')
            content: Content of the conversation entry
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

            # Get active recording span - first try OTel context, then fallback to SessionManager
            span = trace.get_current_span()
            if not (span and getattr(span, "is_recording", lambda: False)()):
                # Fallback: use the most recent active span from SessionManager
                active_spans = _active_spans_var.get()
                if not active_spans:
                    logger.warning("No active span to add conversation attribute.")
                    return

                # Find the most recent *recording* span (the last item can be a finished span)
                recording_span: Optional[trace.Span] = None
                for span in reversed(active_spans):
                    try:
                        if span and getattr(span, "is_recording", lambda: False)():
                            recording_span = span
                            break
                    except Exception:
                        continue

                if recording_span is None:
                    logger.warning("No active span to add conversation attribute.")
                    return
                span = recording_span

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
            max_len = get_conversation_max_len()
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

    @classmethod
    def set_input(cls, value: Any) -> None:
        """Set the ``input`` attribute on the current active span.

        Accepts any value. Dicts and lists are JSON-serialised; primitives are
        converted with ``str()``. The result is truncated to the active config's
        attribute max length.

        Args:
            value: The input value to record.
        """
        try:
            serialized = serialize_value(value)
            cls.set_attribute_on_active_span(NETRA_USER_INPUT, serialized)
        except Exception:
            logger.exception("SessionManager.set_input: failed to set input attribute")

    @classmethod
    def set_output(cls, value: Any) -> None:
        """Set the ``output`` attribute on the current active span.

        Accepts any value. Dicts and lists are JSON-serialised; primitives are
        converted with ``str()``. The result is truncated to the active config's
        attribute max length.

        Args:
            value: The output value to record.
        """
        try:
            serialized = serialize_value(value)
            cls.set_attribute_on_active_span(NETRA_USER_OUTPUT, serialized)
        except Exception:
            logger.exception("SessionManager.set_output: failed to set output attribute")

    @classmethod
    def set_root_input(cls, value: Any) -> None:
        """Set the ``input`` attribute on the root span of the current trace.

        The root span is the oldest span registered via :meth:`register_span`.
        If no such span exists, falls back to the current active OTel span.

        Args:
            value: The input value to record.
        """
        try:
            serialized = serialize_value(value)
            cls.set_attribute_on_root_span(NETRA_USER_INPUT, serialized)
        except Exception:
            logger.exception("SessionManager.set_root_input: failed to set input attribute")

    @classmethod
    def set_root_output(cls, value: Any) -> None:
        """Set the ``output`` attribute on the root span of the current trace.

        The root span is the oldest span registered via :meth:`register_span`.
        If no such span exists, falls back to the current active OTel span.

        Args:
            value: The output value to record.
        """
        try:
            serialized = serialize_value(value)
            cls.set_attribute_on_root_span(NETRA_USER_OUTPUT, serialized)
        except Exception:
            logger.exception("SessionManager.set_root_output: failed to set output attribute")

    @classmethod
    def set_root_output_stream(cls, value: Any) -> Any:
        """Wrap a stream so that the accumulated output is set on the root span when iteration ends.

        The stream is wrapped transparently — the user should iterate over the returned object
        instead of the original stream.  On exhaustion (or garbage collection), the output is
        automatically written to the ``netra.user.output`` attribute of the root span for the
        current trace, which is then promoted to ``output`` by the export pipeline.

        Supports both sync iterables and async iterables.

        Args:
            value: The stream to wrap.  May be a Netra-instrumented wrapper or any generic iterable.

        Returns:
            A wrapped stream proxy.  Returns *value* unchanged if no active trace context
            exists or if *value* is not iterable, so callers can always reassign safely::

                stream = Netra.set_root_output_stream(stream)
        """
        try:
            from netra.instrumentation.stream_utils import wrap_stream_for_root_output
            from netra.processors.root_span_processor import RootSpanProcessor

            root_span = RootSpanProcessor.get_root_span(trace.get_current_span())
            if not root_span:
                logger.warning("SessionManager.set_root_output_stream: no root span found for current trace")
                return value
            return wrap_stream_for_root_output(value, root_span)
        except Exception:
            logger.exception("SessionManager.set_root_output_stream: failed to wrap stream")
            return value

    @classmethod
    def set_attribute_on_root_span(cls, attr_key: str, attr_value: Any) -> None:
        """Set an attribute on the root span of the current trace.


        Args:
            attr_key: Key for the attribute to set
            attr_value: Value for the attribute to set
        """
        try:
            from netra.processors.root_span_processor import RootSpanProcessor

            span_ctx = trace.get_current_span().get_span_context()
            if not span_ctx.is_valid:
                logger.warning("set_attribute_on_root_span called outside any active span context")
                return

            trace_id = span_ctx.trace_id
            root_span = RootSpanProcessor.get_root_span_by_trace_id(trace_id)
            if not root_span:
                # Format as 32-character zero-padded lowercase hex
                logger.warning(f"Cannot find root span for trace_id: {trace_id:032x}")
                return
            root_span.set_attribute(attr_key, attr_value)
        except Exception:
            logger.exception("Failed to set attribute '%s' on root span", attr_key)

    @staticmethod
    def record_exception(
        exception: BaseException,
        attributes: Optional[Dict[str, Any]] = None,
        escaped: Optional[bool] = False,
    ) -> None:
        """Record a caught exception on the currently active span.

        Adds a standard OTel exception event to the span and marks its status
        as ERROR.  Intended to be called from within user exception-handling
        blocks where the exception would otherwise not propagate to the SDK's
        automatic capture logic.

        Args:
            exception: The exception instance to record.
            attributes: Optional extra attributes to attach to the exception
                event.
        """
        try:
            span = trace.get_current_span()
            if not (span and getattr(span, "is_recording", lambda: False)()):
                logger.warning("record_exception: no active recording span to record exception on")
                return

            span.record_exception(exception, attributes=attributes, escaped=escaped)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
            span.set_attribute(f"{Config.LIBRARY_NAME}.error_message", str(exception))
        except Exception:
            logger.exception("Failed to record exception on active span")

    @staticmethod
    def set_attribute_on_active_span(attr_key: str, attr_value: Any) -> None:
        """
        Set an attribute strictly on the currently active OpenTelemetry span.

        Args:
            attr_key: Key for the attribute to set
            attr_value: Value for the attribute to set
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
