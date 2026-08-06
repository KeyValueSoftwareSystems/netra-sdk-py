"""Normalises the shape of livekit-agents' trace into Netra's conventions.

The trace half of this package's two span processors: it rewrites what LiveKit
puts *on* a span — the ``lk.*`` attributes, the conversation events, the
classification markers a span's name implies. The audio half,
``audio_processor.py``, uses spans only as timing boundaries for captured PCM
and shares none of this module's machinery.

INVARIANT for anything added here: ``on_end`` must never mutate the span that is
ending. By the time it runs, ``BatchSpanProcessor`` — registered earlier in the
chain — has already queued that span, and the exporter serialises it on another
thread. ``on_end`` may only mutate *other* spans that are still recording, which is
exactly what the parent-ward content propagation below does.
"""

from __future__ import annotations

import itertools
import logging
import threading
import weakref
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.util.types import Attributes

from netra.instrumentation.livekit.call_span import end_call_span_parenting
from netra.instrumentation.livekit.utils import (
    AGENT_SESSION_SPAN_NAME,
    ATTRIBUTE_MAP,
    AUDIO_TYPE_BY_SPAN_NAME,
    CHAT_CTX_ATTRIBUTE,
    CONVERSATION_MAP,
    EVENT_CHOICE,
    EVENT_ROLE,
    GEN_AI_COMPLETION_CONTENT,
    GEN_AI_COMPLETION_ROLE,
    GEN_AI_PROMPT_CONTENT,
    GEN_AI_PROMPT_ROLE,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_CHARACTER_COUNT,
    IO_FROM_CHILD_SPAN_NAMES,
    LIVEKIT_SCOPE_NAME,
    MAX_CONVERSATION_MESSAGES_PER_SIDE,
    NETRA_AUDIO_TYPE,
    NETRA_CONVERSATION_TRUNCATED,
    NETRA_ENTITY_TYPE,
    NETRA_ENTITY_TYPE_BY_NAME,
    NETRA_SPAN_TYPE,
    NETRA_USAGE_SOURCE,
    TTS_METRICS_ATTRIBUTE,
    USAGE_SOURCE_FRAMEWORK,
    ConversationSide,
    as_attribute_text,
    content_of_choice_event,
    content_of_event,
    conversation_from_attributes,
    is_absent,
    is_usage_attribute,
    is_zero_usage,
    messages_for_parent,
    messages_from_chat_ctx,
    netra_span_type_for,
    role_of_choice_event,
    tts_pricing_attributes_from,
)

logger = logging.getLogger(__name__)

SetAttributeFunc = Callable[[str, Any], None]

# The indexed key pair to write for each side of the conversation convention.
_KEYS_BY_SIDE: Dict[ConversationSide, Tuple[str, str]] = {
    ConversationSide.PROMPT: (GEN_AI_PROMPT_ROLE, GEN_AI_PROMPT_CONTENT),
    ConversationSide.COMPLETION: (GEN_AI_COMPLETION_ROLE, GEN_AI_COMPLETION_CONTENT),
}

# Instance attribute holding a span's ``_ConversationRecorder``. Stored on the span
# itself so the registry in ``SpanMappingProcessor`` can stay a
# ``WeakValueDictionary`` keyed on span id: the span's own lifetime then decides how
# long the entry lives, with no risk of the processor pinning finished spans in
# memory.
_RECORDER_FIELD = "_netra_livekit_recorder"


def _is_livekit_span(span: Any) -> bool:
    """Whether *span* was produced by livekit-agents' own instrumentation.

    Args:
        span: The span to test.

    Returns:
        True only for spans whose instrumentation scope is ``livekit-agents``.
    """
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == LIVEKIT_SCOPE_NAME


def _class_level_writer(span: Span) -> SetAttributeFunc:
    """Return a writer that bypasses every instance-level wrapper on *span*.

    Args:
        span: The span to write to.

    Returns:
        A single-attribute writer calling ``type(span).set_attributes`` directly.
    """
    class_set_attributes = type(span).set_attributes

    def write(key: str, value: Any) -> None:
        """Write one attribute straight to the class method.

        Args:
            key: The attribute name.
            value: The attribute value.
        """
        class_set_attributes(span, {key: value})

    return write


def _write_tts_pricing(span: Span, metrics_payload: Any) -> None:
    """Lift the priceable fields out of LiveKit's TTS metrics blob into Netra keys.

    Writes through ``span.set_attribute`` — the outermost wrapper — so the model
    reaches the rest of the processor chain and the character count takes the
    usage branch, which stamps ``netra.usage.source`` on it like every other
    framework-reported usage number.

    Args:
        span: The LiveKit span the metrics were written on (``tts_request``).
        metrics_payload: The value of ``lk.tts_metrics``.
    """
    pricing = tts_pricing_attributes_from(metrics_payload)
    if pricing.model is not None:
        span.set_attribute(GEN_AI_REQUEST_MODEL, pricing.model)
    if pricing.character_count is not None:
        span.set_attribute(GEN_AI_USAGE_CHARACTER_COUNT, pricing.character_count)


class _ConversationRecorder:
    """Appends messages to one span's indexed gen_ai prompt/completion sequences.

    The single place an indexed conversation attribute is written, so every source
    that contributes to a span — mapped ``lk.*`` attributes, an expanded chat
    context, conversation events, and a child span's content — advances the same
    counters and cannot overwrite another source's entries. One instance per
    LiveKit span, created in ``SpanMappingProcessor.on_start``.
    """

    __slots__ = ("_span", "_next_index", "_truncated")

    def __init__(self, span: Span) -> None:
        """Start both index sequences at zero for *span*.

        Args:
            span: The span whose conversation this records.
        """
        self._span = span
        self._next_index: Dict[ConversationSide, Iterator[int]] = {
            ConversationSide.PROMPT: itertools.count(),
            ConversationSide.COMPLETION: itertools.count(),
        }
        self._truncated = False

    def append(self, side: ConversationSide, role: str, content: Any) -> None:
        """Append one message to the given side of the conversation.

        Writes through ``span.set_attribute`` — the outermost wrapper — so the
        values reach ``SpanIOProcessor``, which assembles them into
        ``input``/``output``.

        Silently stops at ``MAX_CONVERSATION_MESSAGES_PER_SIDE`` and marks the
        span instead — see that constant for why an unbounded sequence is not
        merely wasteful but destructive. The cap lives here, rather than at each
        call site, so it covers every source that feeds a recorder: mapped
        ``lk.*`` attributes, an expanded chat context, conversation events, and a
        child span's propagated content.

        Args:
            side: Which indexed sequence to append to.
            role: The conversation role to stamp alongside the text.
            content: The message text.
        """
        # The budget is read off the counter itself rather than a separate
        # decrement: ``next()`` on an ``itertools.count`` is atomic, and a span's
        # attributes can be written from more than one thread (a child ending on
        # another thread propagates content up through here).
        index = next(self._next_index[side])
        if index >= MAX_CONVERSATION_MESSAGES_PER_SIDE:
            self._mark_truncated()
            return

        role_key, content_key = _KEYS_BY_SIDE[side]
        self._span.set_attribute(role_key.format(index=index), role)
        self._span.set_attribute(content_key.format(index=index), as_attribute_text(content))

    def _mark_truncated(self) -> None:
        """Record on the span that the conversation was cut short by the cap.

        Written at most once. The guard is not synchronised: two threads racing
        here both write the same value, so the only cost is a duplicate write.
        """
        if self._truncated:
            return
        self._truncated = True
        self._span.set_attribute(NETRA_CONVERSATION_TRUNCATED, True)

    def append_attribute(self, key: str, value: Any) -> bool:
        """Route an ``lk.*`` conversation-content attribute into the sequences.

        Args:
            key: The LiveKit attribute name being written.
            value: The value being written.

        Returns:
            True when *key* belongs to the conversation convention — whether or not
            it carried a value — so the caller knows not to fall through to
            ``ATTRIBUTE_MAP``.
        """
        if key == CHAT_CTX_ATTRIBUTE:
            messages = messages_from_chat_ctx(value)
            # Keep the newest turns. ``append`` caps the sequence either way, but it
            # can only drop what arrives last, so feeding it the whole context
            # oldest-first would preserve the opening of the call and discard the
            # turns this span is actually about. The full context stays on the span
            # verbatim as ``lk.chat_ctx``.
            if len(messages) > MAX_CONVERSATION_MESSAGES_PER_SIDE:
                self._mark_truncated()
                messages = messages[-MAX_CONVERSATION_MESSAGES_PER_SIDE:]
            for role, content in messages:
                self.append(ConversationSide.PROMPT, role, content)
            return True

        target = CONVERSATION_MAP.get(key)
        if target is None:
            return False
        if not is_absent(value):
            self.append(target.side, target.role, value)
        return True

    def append_event(self, name: str, attributes: Attributes) -> None:
        """Route a LiveKit conversation event into the sequences.

        Args:
            name: The event name LiveKit passed to ``add_event``.
            attributes: The event attributes. Events that are not conversation
                content, or that carry no text, contribute nothing.
        """
        if name == EVENT_CHOICE:
            content = content_of_choice_event(attributes)
            if content:
                self.append(ConversationSide.COMPLETION, role_of_choice_event(attributes), content)
            return

        role = EVENT_ROLE.get(name)
        if role is None:
            return
        content = content_of_event(attributes)
        if content:
            self.append(ConversationSide.PROMPT, role, content)

    def append_child_conversation(self, child: ReadableSpan) -> None:
        """Append a finished child span's conversation content.

        Args:
            child: The span that has ended directly beneath this one.
        """
        conversation = conversation_from_attributes(child.attributes)
        # A child no LLM-aware instrumentation touched carries an ``input`` that is
        # not a conversation at all — an HTTP envelope, a SQL statement — and must
        # not be copied up as if it were one.
        allow_raw_io = conversation.carries_gen_ai or _is_livekit_span(child)
        for message in messages_for_parent(conversation, allow_raw_io=allow_raw_io):
            self.append(message.side, message.role, message.content)


class SpanMappingProcessor(SpanProcessor):  # type: ignore[misc]
    """Mirrors LiveKit's ``lk.*`` attributes and conversation events into Netra keys.

    Additive throughout: an ``lk.*`` attribute is never deleted or rewritten, and a
    conversation event is always still recorded on the span. The one exception is
    a zero token count, which is dropped rather than mirrored — see
    ``is_zero_usage``.

    Conversation content — LiveKit's own ``lk.*`` attributes, its serialised chat
    contexts, and its conversation events — all land in the indexed
    ``gen_ai.prompt.*``/``gen_ai.completion.*`` pair that ``SpanIOProcessor``
    assembles into ``input``/``output``. That is the convention every other Netra
    instrumentation emits, so a voice turn renders like any other span.

    Two spans carry no conversation content of their own and inherit it from a
    direct child when that child ends — see ``IO_FROM_CHILD_SPAN_NAMES`` and
    ``on_end``.

    Two values are additionally *derived* rather than mirrored: the model and the
    character count that price a TTS call, which LiveKit reports only inside the
    opaque ``lk.tts_metrics`` JSON blob — see ``_write_tts_pricing``.
    """

    def __init__(self) -> None:
        """Create the registry of spans awaiting content from a child."""
        # Weak values: an entry costs nothing once the span itself is collected, so
        # a span that somehow never ends cannot leak. Guarded by a lock because
        # spans can start and end on threads other than the agent's event loop.
        self._io_parents: "weakref.WeakValueDictionary[int, Span]" = weakref.WeakValueDictionary()
        self._io_parents_lock = threading.Lock()

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Stamp the Netra markers and install the mapping wrappers on a LiveKit span.

        Args:
            span: The span that was started.
            parent_context: The parent context (unused).
        """
        try:
            if not _is_livekit_span(span):
                return
            self._stamp_markers(span)

            recorder = _ConversationRecorder(span)
            setattr(span, _RECORDER_FIELD, recorder)
            self._wrap_set_attribute(span, recorder)
            self._wrap_add_event(span, recorder)

            if span.name in IO_FROM_CHILD_SPAN_NAMES:
                self._register_io_parent(span)
        except Exception:
            logger.warning("netra.livekit: span mapping could not be installed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """Copy a finished span's conversation content up to its parent, if wanted,
        and close the call span when a session ends.

        Deliberately *not* gated on ``_is_livekit_span``: the child holding the
        content is usually the provider's own span (``openai.chat`` and friends),
        which belongs to another instrumentation scope entirely.

        Never touches *this* span — see the module docstring. It only appends to a
        still-recording parent, which the exporter has not seen yet, and ends the
        call span, which is a different span again.

        Args:
            span: The span that has ended.
        """
        try:
            self._propagate_content_to_parent(span)
        except Exception:
            logger.debug("netra.livekit: content propagation to the parent span failed", exc_info=True)
        try:
            self._deregister_io_parent(span)
        except Exception:
            logger.debug("netra.livekit: span could not be deregistered", exc_info=True)
        try:
            self._close_call_span(span)
        except Exception:
            logger.debug("netra.livekit: call span could not be closed", exc_info=True)

    @staticmethod
    def _close_call_span(span: ReadableSpan) -> None:
        """End the ``livekit-call`` span wrapping *span*, when *span* ends a session.

        ``agent_session`` ending is LiveKit's own authoritative "the call is over"
        signal: it is emitted on all five close reasons and needs no method wrap, so
        it is the primary end path for the call span (``wrap_aclose`` is the
        fallback). The call span is looked up by *span*'s parent span id, which is
        the call span's own span id — an exact match, so a job running two sessions
        cannot have one session's close end the other's call span.

        Args:
            span: The span that has ended.
        """
        if span.name != AGENT_SESSION_SPAN_NAME or not _is_livekit_span(span):
            return

        parent = getattr(span, "parent", None)
        end_call_span_parenting(getattr(parent, "span_id", None))

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush.

        Args:
            timeout_millis: Maximum time to wait (unused).

        Returns:
            Always True.
        """
        return True

    def shutdown(self) -> None:
        """No-op shutdown."""

    @staticmethod
    def _stamp_markers(span: Span) -> None:
        """Write the Netra classification markers a LiveKit span's name implies.

        Args:
            span: The LiveKit span to stamp.
        """
        span.set_attribute(NETRA_SPAN_TYPE, netra_span_type_for(span.name))

        entity_type = NETRA_ENTITY_TYPE_BY_NAME.get(span.name)
        if entity_type is not None:
            span.set_attribute(NETRA_ENTITY_TYPE, entity_type)

        audio_type = AUDIO_TYPE_BY_SPAN_NAME.get(span.name)
        if audio_type is not None:
            span.set_attribute(NETRA_AUDIO_TYPE, audio_type)

    def _register_io_parent(self, span: Span) -> None:
        """Record *span* as one whose conversation content arrives from a child.

        Args:
            span: The LiveKit span to register.
        """
        context = span.get_span_context()
        if context is None:
            return
        with self._io_parents_lock:
            self._io_parents[context.span_id] = span

    def _propagate_content_to_parent(self, span: ReadableSpan) -> None:
        """Append *span*'s conversation content to its parent's gen_ai sequences.

        Args:
            span: The span that has ended.
        """
        parent_context = span.parent
        if parent_context is None or not self._io_parents:
            return
        with self._io_parents_lock:
            parent = self._io_parents.get(parent_context.span_id)
        if parent is None or not parent.is_recording():
            return

        recorder: Optional[_ConversationRecorder] = getattr(parent, _RECORDER_FIELD, None)
        if recorder is None:
            return
        recorder.append_child_conversation(span)

    def _deregister_io_parent(self, span: ReadableSpan) -> None:
        """Drop *span* from the registry if it was awaiting content from a child.

        Args:
            span: The span that has ended.
        """
        if span.name not in IO_FROM_CHILD_SPAN_NAMES:
            return
        context = span.get_span_context()
        if context is None:
            return
        with self._io_parents_lock:
            self._io_parents.pop(context.span_id, None)

    @staticmethod
    def _wrap_set_attribute(span: Span, recorder: _ConversationRecorder) -> None:
        """Wrap ``span.set_attribute`` so mapped ``lk.*`` writes also write Netra keys.

        Chains through the previously-installed wrapper rather than the class
        method, so writes still pass down through ``SpanIOProcessor`` and
        ``InstrumentationSpanProcessor``. ``set_attributes`` (plural) is wrapped
        too because the OTel SDK writes it straight to ``_attributes`` without
        going through ``set_attribute`` — LiveKit uses it, e.g. for the
        ``gen_ai.*`` request attributes on ``llm_request`` and for
        ``lk.user_transcript`` on ``user_turn``.

        Args:
            span: The LiveKit span to wrap.
            recorder: The span's conversation recorder.
        """
        previous: SetAttributeFunc = span.set_attribute
        if "set_attribute" not in vars(span):
            # Nothing has wrapped this span, so ``previous`` is the raw SDK method
            # — and from opentelemetry-sdk 1.41 that method is implemented as
            # ``self.set_attributes({key: value})``, which resolves the plural
            # wrapper installed below and recurses until RecursionError. In Netra's
            # own pipeline this branch never runs (``InstrumentationSpanProcessor``
            # always wraps first, and terminates its own writes at the class
            # method for exactly this reason); it keeps the processor correct when
            # it is registered on a provider by itself.
            previous = _class_level_writer(span)

        def map_attribute(key: str, value: Any) -> None:
            """Write *key* through, then write whatever Netra key it implies.

            Args:
                key: The attribute name LiveKit is writing.
                value: The attribute value LiveKit is writing.
            """
            if is_usage_attribute(key):
                if is_zero_usage(value):
                    return
                previous(key, value)
                # Marks whose accounting this is, so the backend can prefer a
                # provider span's tokens over the framework's for the same call.
                previous(NETRA_USAGE_SOURCE, USAGE_SOURCE_FRAMEWORK)
                return

            previous(key, value)

            if key == TTS_METRICS_ATTRIBUTE:
                # Not conversation content and not in ATTRIBUTE_MAP: the blob
                # is forwarded as-is and its priceable fields are lifted out.
                _write_tts_pricing(span, value)
                return

            if recorder.append_attribute(key, value):
                return

            target = ATTRIBUTE_MAP.get(key)
            if target is None or is_absent(value):
                return
            previous(target, value)

        def patched_set_attribute(key: str, value: Any) -> None:
            """Map *key* onto its Netra keys, falling back to a plain write.

            Args:
                key: The attribute name LiveKit is writing.
                value: The attribute value LiveKit is writing.
            """
            try:
                map_attribute(key, value)
            except Exception:
                logger.debug("netra.livekit: attribute mapping failed for %s", key, exc_info=True)
                try:
                    previous(key, value)
                except Exception:
                    logger.debug("netra.livekit: set_attribute failed for %s", key, exc_info=True)

        def patched_set_attributes(attributes: Mapping[str, Any]) -> None:
            """Route a bulk write through the single-attribute mapping.

            Args:
                attributes: The attributes LiveKit is writing.
            """
            for key, value in (attributes or {}).items():
                patched_set_attribute(key, value)

        setattr(span, "set_attribute", patched_set_attribute)
        setattr(span, "set_attributes", patched_set_attributes)

    @staticmethod
    def _wrap_add_event(span: Span, recorder: _ConversationRecorder) -> None:
        """Wrap ``span.add_event`` so conversation events become attributes.

        LiveKit emits conversation content as span *events*
        (``_chat_ctx_to_otel_events`` for the request, ``gen_ai.choice`` for the
        reply), which a ``set_attribute`` wrapper structurally cannot see — so
        LiveKit spans would otherwise export with empty ``input``/``output``.

        Assigning ``span.add_event`` shadows the class method, because
        ``add_event`` is not a dunder and attribute lookup hits the instance dict.

        Args:
            span: The LiveKit span to wrap.
            recorder: The span's conversation recorder.
        """
        original = span.add_event

        def patched_add_event(
            name: str,
            attributes: Attributes = None,
            timestamp: Optional[int] = None,
        ) -> None:
            """Record the event as conversation content, then forward it verbatim.

            Args:
                name: The event name LiveKit passed to ``add_event``.
                attributes: The event attributes, if any.
                timestamp: The event timestamp, if any. Forwarded untouched.
            """
            try:
                recorder.append_event(name, attributes)
            except Exception:
                logger.debug("netra.livekit: event -> attribute mapping failed for %s", name, exc_info=True)
            # ALWAYS forward: the user's event must be recorded whatever happens
            # on our side.
            original(name, attributes, timestamp)

        setattr(span, "add_event", patched_add_event)
