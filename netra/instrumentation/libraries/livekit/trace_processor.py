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

from netra.exporters.utils import set_span_parent
from netra.instrumentation.libraries.livekit.call_span import (
    agent_name_of,
    call_id_of,
    end_call_span_parenting,
    failure_status_of,
)
from netra.instrumentation.libraries.livekit.utils import (
    AGENT_SESSION_SPAN_NAME,
    AGENT_TURN_SPAN_NAME,
    ATTRIBUTE_MAP,
    AUDIO_TYPE_BY_SPAN_NAME,
    CHAT_CTX_ATTRIBUTE,
    CONVERSATION_MAP,
    EVENT_CHOICE,
    EVENT_ROLE,
    GEN_AI_AUDIO_DURATION,
    GEN_AI_COMPLETION_CONTENT,
    GEN_AI_COMPLETION_ROLE,
    GEN_AI_PROMPT_CONTENT,
    GEN_AI_PROMPT_ROLE,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_USAGE_CHARACTER_COUNT,
    GEN_AI_USAGE_COMPLETION_TOKENS,
    GEN_AI_USAGE_PROMPT_TOKENS,
    IO_FROM_CHILD_SPAN_NAMES,
    LIVEKIT_SCOPE_NAME,
    MAX_CONVERSATION_MESSAGES_PER_SIDE,
    NETRA_AGENT_NAME,
    NETRA_AUDIO_TYPE,
    NETRA_CONVERSATION_TRUNCATED,
    NETRA_ENTITY_TYPE,
    NETRA_ENTITY_TYPE_BY_NAME,
    NETRA_SPAN_TYPE,
    NETRA_USAGE_SOURCE,
    SPEAKING_SPAN_NAMES,
    TTS_METRICS_ATTRIBUTE,
    USAGE_SOURCE_FRAMEWORK,
    USER_TURN_SPAN_NAME,
    AudioPricingAttributes,
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
    stt_pricing_attributes_from,
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


# The zero point for the accumulation in ``_write_usage``: nothing carried over.
_NO_USAGE = AudioPricingAttributes(None, None, None, None, None)


def _write_tts_pricing(span: Span, metrics_payload: Any) -> None:
    """Lift the priceable fields out of LiveKit's TTS metrics blob into Netra keys.

    LiveKit writes ``lk.tts_metrics`` once per ``tts_request``, as one complete
    blob, so the values are set rather than accumulated.

    Written through ``span.set_attribute`` — the outermost wrapper — so the model
    reaches the rest of the processor chain and the character count takes the usage
    branch, which stamps ``netra.usage.source`` on it like every other
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
    _write_usage(span, pricing, previous=_NO_USAGE)


def _write_stt_pricing(span: Span, pricing: AudioPricingAttributes) -> None:
    """Add one STT metrics sample to the running usage totals on a ``user_turn`` span.

    Accumulated rather than set, because LiveKit reports transcription usage
    *incrementally*: a streaming STT emits ``RECOGNITION_USAGE`` on every final
    transcript and resets its counter after each one, so a turn with three finals
    arrives here three times, each carrying only the audio since the last. Setting
    would keep the last fragment and discard the rest of the turn.

    The model is set, not accumulated — every sample in a turn reports the same one.

    Args:
        span: The still-recording ``user_turn`` span.
        pricing: One ``STTMetrics`` sample.
    """
    if pricing.model is not None:
        span.set_attribute(GEN_AI_REQUEST_MODEL, pricing.model)
    _write_usage(span, pricing, previous=_usage_on(span))


def _write_usage(span: Span, pricing: AudioPricingAttributes, *, previous: AudioPricingAttributes) -> None:
    """Write the usage fields a TTS and an STT call report alike, added to *previous*.

    Writes through ``span.set_attribute`` — the outermost wrapper — so the token
    counts take the usage branch of ``map_attribute``, which stamps
    ``netra.usage.source`` on them like every other framework-reported number.

    Args:
        span: The span to write to.
        pricing: The values LiveKit reported in this sample.
        previous: The totals already on the span, or ``_NO_USAGE`` for a value
            reported once and in full.
    """
    for key, reported, carried in (
        (GEN_AI_USAGE_PROMPT_TOKENS, pricing.prompt_tokens, previous.prompt_tokens),
        (GEN_AI_USAGE_COMPLETION_TOKENS, pricing.completion_tokens, previous.completion_tokens),
        (GEN_AI_AUDIO_DURATION, pricing.audio_duration, previous.audio_duration),
    ):
        if reported is None:
            continue
        span.set_attribute(key, reported + (carried or 0))


def _usage_on(span: Span) -> AudioPricingAttributes:
    """Read the usage totals already written on *span*.

    Args:
        span: The span to read back from.

    Returns:
        The totals, each ``None`` when the span carries no such value yet.
    """
    attributes: Mapping[str, Any] = getattr(span, "attributes", None) or {}
    prompt_tokens = _numeric(attributes.get(GEN_AI_USAGE_PROMPT_TOKENS))
    completion_tokens = _numeric(attributes.get(GEN_AI_USAGE_COMPLETION_TOKENS))

    return _NO_USAGE._replace(
        prompt_tokens=None if prompt_tokens is None else int(prompt_tokens),
        completion_tokens=None if completion_tokens is None else int(completion_tokens),
        audio_duration=_numeric(attributes.get(GEN_AI_AUDIO_DURATION)),
    )


def _numeric(value: Any) -> Optional[float]:
    """Read a value back off a span as a number, or None if it is not one.

    Args:
        value: The attribute value.

    Returns:
        The value as a float, or ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# Instance attribute holding the id of the call a ``user_turn`` span belongs to.
# Stashed on the span at ``on_start`` because ``on_end`` is handed a
# ``ReadableSpan`` and no context, so the key it must be deregistered under has to
# travel on the span itself — the same reason ``_RECORDER_FIELD`` does.
_CALL_ID_FIELD = "_netra_livekit_call_id"

# The ``user_turn`` span currently recording in each call, keyed by call id — the
# span id of that call's ``livekit-call`` span, as ``call_id_scope`` attaches it.
# STT usage arrives on the session's ``metrics_collected`` event, which carries no
# span and no context, so this is how ``wrappers.py`` finds the turn it belongs to.
#
# Keyed on the call and NOT on the trace, because a trace does not identify a call:
# see ``_CALL_ID_KEY`` for why two sessions in one job share a trace id, and why
# filing turns under one would bill one caller's audio to the other.
#
# Weak values for the same reason as ``_io_parents``: a span that somehow never
# ends cannot leak. One turn records at a time per session, so a call id identifies
# exactly one candidate.
_user_turn_spans: "weakref.WeakValueDictionary[int, Span]" = weakref.WeakValueDictionary()
_user_turn_lock = threading.Lock()


def _register_user_turn_span(span: Span, parent_context: Optional[otel_context.Context]) -> None:
    """Make *span* the turn STT usage is attributed to in its call.

    Args:
        span: A starting ``user_turn`` span.
        parent_context: The context the span is being created in, carrying the call
            id ``wrap_start`` attached. A span with no call id in scope is not
            registered at all: it belongs to no call this package opened, so
            nothing is subscribed to its session's metrics and no usage will ever
            be looked up for it.
    """
    call_id = call_id_of(parent_context)
    if call_id is None:
        return
    setattr(span, _CALL_ID_FIELD, call_id)
    with _user_turn_lock:
        _user_turn_spans[call_id] = span


def _deregister_user_turn_span(span: ReadableSpan) -> None:
    """Drop *span* from the user-turn registry, if it is still the registered one.

    Gated on identity because ``on_end`` can run after the next turn has already
    registered itself — a stale end must not evict the turn that replaced it.

    Args:
        span: The span that has ended.
    """
    if span.name != USER_TURN_SPAN_NAME:
        return
    call_id = getattr(span, _CALL_ID_FIELD, None)
    if call_id is None:
        return
    context = span.get_span_context()
    if context is None:
        return
    with _user_turn_lock:
        registered = _user_turn_spans.get(call_id)
        if registered is not None and registered.get_span_context().span_id == context.span_id:
            del _user_turn_spans[call_id]


def record_stt_usage(call_id: int, metrics_payload: Any) -> None:
    """Add one LiveKit ``STTMetrics`` sample to the recording ``user_turn`` span.

    A sample whose turn has already ended is dropped rather than carried onto the
    next one: an interrupted turn can close before its final metrics arrive, and
    billing the following turn for the previous one's audio is worse than losing
    the sample. A sample for a call with no registered turn is dropped for the same
    reason — including one arriving on a session whose current call is not the one
    the sample was measured on.

    The lookup and the accumulating write are held under one lock: the write is a
    read-modify-write of the totals on the span, so two samples landing at once
    would otherwise lose one. Nothing on the write path reads the registry, so the
    lock cannot be re-entered.

    Args:
        call_id: The call the metrics belong to — the ``livekit-call`` span's own
            span id, as ``call_id_of_session`` reports it.
        metrics_payload: A serialised ``STTMetrics`` mapping or JSON string.
    """
    pricing = stt_pricing_attributes_from(metrics_payload)

    with _user_turn_lock:
        span = _user_turn_spans.get(call_id)
        if span is None or not span.is_recording():
            logger.debug("netra.livekit: no recording user_turn span for STT usage in call %x", call_id)
            return
        _write_stt_pricing(span, pricing)


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


def _stamp_agent_name(span: Span, parent_context: Optional[otel_context.Context]) -> None:
    """Name the dispatched agent on a starting ``agent_turn`` span.

    LiveKit puts the name on ``agent_session`` and ``job_entrypoint`` only, so a
    turn has to be told. The value travels on the context ``wrap_start`` attached:
    every ``agent_turn`` is opened with ``AgentSession._root_span_context``
    (``voice/agent_activity.py``), which is snapshotted inside ``start()`` and so
    carries it — the same route ``user_turn`` takes to its call id.

    Args:
        span: A starting ``agent_turn`` span.
        parent_context: The context the span is being created in. A turn with no
            agent name in scope is left unstamped: the worker declared none, or the
            turn belongs to no session this package wrapped.
    """
    agent_name = agent_name_of(parent_context)
    if agent_name is None:
        return
    span.set_attribute(NETRA_AGENT_NAME, agent_name)


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

        # trace_id -> SpanContext of the agent_session span. Used to reparent
        # orphaned speaking spans (user_speaking/agent_speaking) whose ambient OTel
        # context lost the parent due to a context propagation issue in livekit-agents.
        self._session_span_contexts: Dict[int, Any] = {}
        self._session_span_contexts_lock = threading.Lock()

        # Deferred call-span close: if speaking spans are still open when
        # agent_session ends, the root close is deferred until they all end.
        # trace_id -> count of open speaking spans in that trace.
        self._open_speaking_counts: Dict[int, int] = {}
        # trace_id -> (call_span_id, Optional[Status]) for deferred closes.
        self._pending_closes: Dict[int, Tuple[int, Any]] = {}
        self._speaking_lock = threading.Lock()

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

            if span.name == AGENT_SESSION_SPAN_NAME:
                self._register_session_span(span)

            if span.name in SPEAKING_SPAN_NAMES:
                self._reparent_if_orphaned(span)
                self._increment_speaking_count(span)

            recorder = _ConversationRecorder(span)
            setattr(span, _RECORDER_FIELD, recorder)
            self._wrap_set_attribute(span, recorder)
            self._wrap_add_event(span, recorder)

            if span.name in IO_FROM_CHILD_SPAN_NAMES:
                self._register_io_parent(span)
            if span.name == USER_TURN_SPAN_NAME:
                _register_user_turn_span(span, parent_context)
            if span.name == AGENT_TURN_SPAN_NAME:
                _stamp_agent_name(span, parent_context)
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
            _deregister_user_turn_span(span)
        except Exception:
            logger.debug("netra.livekit: user turn span could not be deregistered", exc_info=True)
        try:
            self._deregister_session_span(span)
        except Exception:
            logger.debug("netra.livekit: session span could not be deregistered", exc_info=True)
        try:
            self._decrement_speaking_count(span)
        except Exception:
            logger.debug("netra.livekit: speaking span count could not be decremented", exc_info=True)
        try:
            self._close_call_span(span)
        except Exception:
            logger.debug("netra.livekit: call span could not be closed", exc_info=True)

    def _close_call_span(self, span: ReadableSpan) -> None:
        """End the ``livekit-call`` span wrapping *span*, when *span* ends a session.

        ``agent_session`` ending is LiveKit's own authoritative "the call is over"
        signal: it is emitted on all five close reasons and needs no method wrap, so
        it is the primary end path for the call span (``wrap_aclose`` is the
        fallback). The call span is looked up by *span*'s parent span id, which is
        the call span's own span id — an exact match, so a job running two sessions
        cannot have one session's close end the other's call span.

        If speaking spans are still open (rare: happens when ``_aclose_impl`` raises
        before ending them), the close is deferred until the last one ends.

        A session that ended in error closes its call span in error too: the call
        span is the trace root, so it is where trace-level health is read from.

        Args:
            span: The span that has ended.
        """
        if span.name != AGENT_SESSION_SPAN_NAME or not _is_livekit_span(span):
            return

        parent = getattr(span, "parent", None)
        call_span_id = getattr(parent, "span_id", None)
        if call_span_id is None:
            return

        span_ctx = span.context if hasattr(span, "context") else None
        trace_id = getattr(span_ctx, "trace_id", None) if span_ctx else None

        status = failure_status_of(span)

        if trace_id is not None:
            with self._speaking_lock:
                count = self._open_speaking_counts.get(trace_id, 0)
                if count > 0:
                    self._pending_closes[trace_id] = (call_span_id, status)
                    logger.debug(
                        "netra.livekit: deferring call span close — %d speaking span(s) still open",
                        count,
                    )
                    return

        end_call_span_parenting(call_span_id, status=status)

    # ------------------------------------------------------------------
    # Session span registry — for reparenting orphaned speaking spans
    # ------------------------------------------------------------------

    def _register_session_span(self, span: Span) -> None:
        """Record the ``agent_session`` span's context for reparenting lookups.

        Args:
            span: The ``agent_session`` span that just started.
        """
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return
        with self._session_span_contexts_lock:
            self._session_span_contexts[ctx.trace_id] = ctx

    def _deregister_session_span(self, span: ReadableSpan) -> None:
        """Remove the ``agent_session`` mapping when its span ends.

        Args:
            span: The span that ended (only acts on ``agent_session``).
        """
        if not _is_livekit_span(span) or span.name != AGENT_SESSION_SPAN_NAME:
            return
        ctx = span.context if hasattr(span, "context") else getattr(span, "_context", None)
        if ctx is None:
            ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
        if ctx is None or not getattr(ctx, "is_valid", False):
            return
        with self._session_span_contexts_lock:
            self._session_span_contexts.pop(ctx.trace_id, None)

    def _reparent_if_orphaned(self, span: Span) -> None:
        """Reparent a speaking span under ``agent_session`` if it has no valid parent.

        LiveKit's ``_update_user_state`` creates ``user_speaking`` without an explicit
        OTel context, relying on the ambient context. Normally the ambient context has
        ``user_turn`` as the current span (via ``use_span`` in audio_recognition.py),
        so the span is correctly parented. In edge cases (``claim_user_turn``, callback
        racing), the ambient may have no span, leaving ``user_speaking`` orphaned.

        This method only acts on spans that are truly parentless — it never overrides
        a valid parent, preserving the correct ``agent_turn``/``user_turn`` hierarchy.

        Args:
            span: A ``user_speaking`` or ``agent_speaking`` span that just started.
        """
        parent = getattr(span, "parent", None)
        if parent is not None and getattr(parent, "is_valid", False):
            return

        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return

        with self._session_span_contexts_lock:
            session_ctx = self._session_span_contexts.get(ctx.trace_id)

        if session_ctx is None:
            return

        try:
            set_span_parent(span, session_ctx)
            logger.debug(
                "netra.livekit: reparented orphaned %s span under agent_session (trace=%032x)",
                span.name,
                ctx.trace_id,
            )
        except Exception:
            logger.debug("netra.livekit: failed to reparent %s span", span.name, exc_info=True)

    # ------------------------------------------------------------------
    # Deferred call-span close — speaking span counting
    # ------------------------------------------------------------------

    def _increment_speaking_count(self, span: Span) -> None:
        """Track that a speaking span opened, keyed by trace_id.

        All spans in a call share the same trace_id regardless of their parent,
        so keying by trace_id works whether the span is under agent_turn, user_turn,
        or agent_session.

        Args:
            span: The speaking span that just started.
        """
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return

        with self._speaking_lock:
            self._open_speaking_counts[ctx.trace_id] = self._open_speaking_counts.get(ctx.trace_id, 0) + 1

    def _decrement_speaking_count(self, span: ReadableSpan) -> None:
        """Track that a speaking span closed, releasing a deferred close if needed.

        Args:
            span: The span that ended (only acts on speaking spans).
        """
        if not _is_livekit_span(span) or span.name not in SPEAKING_SPAN_NAMES:
            return

        span_ctx = span.context if hasattr(span, "context") else None
        if span_ctx is None:
            span_ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
        if span_ctx is None or not getattr(span_ctx, "is_valid", False):
            return

        trace_id = span_ctx.trace_id
        pending_entry = None
        with self._speaking_lock:
            count = self._open_speaking_counts.get(trace_id, 0)
            if count > 0:
                count -= 1
                if count == 0:
                    self._open_speaking_counts.pop(trace_id, None)
                    pending_entry = self._pending_closes.pop(trace_id, None)
                else:
                    self._open_speaking_counts[trace_id] = count

        if pending_entry is not None:
            call_span_id, status = pending_entry
            logger.debug("netra.livekit: releasing deferred call span close (last speaking span ended)")
            end_call_span_parenting(call_span_id, status=status)

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
