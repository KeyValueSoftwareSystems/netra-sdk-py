"""wrapt wrappers for LiveKit's ``AgentSession`` lifecycle and audio hooks.

The session id is attached as OTel baggage *around* ``AgentSession.start`` so
that the ``agent_session`` root span — created inside ``start()`` — carries it,
then detached so the caller's context is restored.

When audio capture is enabled, ``_after_start`` constructs the sender and hooks,
patches the session's audio I/O, and registers the hooks for the
``AudioSpanProcessor`` to find.  ``_before_close`` tears everything down.

Nothing in here may change the behaviour of the user's application: the hook calls
the wrapped function even if our own logic raises, and exceptions raised by the
user's code propagate untouched.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Tuple

from netra.config import get_active_config
from netra.session_manager import SessionManager

if TYPE_CHECKING:
    from livekit.agents import AgentSession
    from livekit.rtc import AudioFrame

    from netra.instrumentation.livekit.audio_sender import AudioChunkSender

logger = logging.getLogger(__name__)

WrappedAsync = Callable[..., Awaitable[Any]]


# ---------------------------------------------------------------------------
# Session-id resolution
# ---------------------------------------------------------------------------


def _resolve_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    """Derive the Netra session id for an ``AgentSession.start`` call.

    Prefers the LiveKit room SID — the id LiveKit itself identifies the session by —
    and falls back to the room name when there is no job context to read it from.

    Args:
        kwargs: The keyword arguments ``start()`` was called with.

    Returns:
        The session id, or ``None`` when neither source yields one — in which case
        the session simply carries no Netra session id.
    """
    return _room_sid_from_job_context() or _room_name(kwargs)


def _room_sid_from_job_context() -> Optional[str]:
    """Read the room SID off the job assignment, or None if it is unavailable.

    Taken from ``JobContext.job.room.sid`` rather than ``rtc.Room.sid``: the latter
    is an *async* property that only resolves once the room is connected, and in the
    usual entrypoint ``session.start()`` runs before ``ctx.connect()`` — awaiting it
    here would stall the user's agent, and in console mode (no real room) it would
    never resolve. The job assignment carries the same server-issued SID
    synchronously, before connect, which is what lets the ``agent_session`` root
    span be stamped with it.

    Returns:
        The room SID, or ``None`` outside a job (eval mode, direct library use) or
        when livekit-agents does not expose one.
    """
    try:
        from livekit.agents import get_job_context

        job_context = get_job_context(required=False)
    except Exception:
        logger.debug("netra.livekit: could not read the job context", exc_info=True)
        return None

    if job_context is None:
        return None

    try:
        sid = getattr(getattr(job_context.job, "room", None), "sid", None)
    except Exception:
        logger.debug("netra.livekit: could not read the room sid off the job", exc_info=True)
        return None

    if isinstance(sid, str) and sid:
        return sid
    return None


def _room_name(kwargs: Dict[str, Any]) -> Optional[str]:
    """Read the room name from ``AgentSession.start``'s ``room`` kwarg.

    ``room`` is keyword-only and defaults to ``NOT_GIVEN``, so it MUST NOT be read
    positionally and MUST be checked with LiveKit's ``is_given`` before touching
    ``room.name``.

    Args:
        kwargs: The keyword arguments ``start()`` was called with.

    Returns:
        The room name, or ``None`` when it cannot be determined.
    """
    room = kwargs.get("room")
    if room is None:
        return None

    try:
        from livekit.agents.utils import is_given

        if not is_given(room):
            return None
    except Exception:
        logger.debug("netra.livekit: could not check room kwarg with is_given", exc_info=True)
        return None

    name = getattr(room, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


# ---------------------------------------------------------------------------
# Session-span helpers
# ---------------------------------------------------------------------------


def _session_span(instance: Any) -> Optional[Any]:
    """Return the live ``agent_session`` span, or ``None`` once it is gone."""
    return getattr(instance, "_session_span", None)


def _trace_id_of(session_span: Optional[Any]) -> Optional[int]:
    """Read the trace id off the ``agent_session`` span."""
    if session_span is None:
        return None
    try:
        span_context = session_span.get_span_context()
    except Exception:
        logger.debug("netra.livekit: could not read the session span context", exc_info=True)
        return None
    if span_context is None or not span_context.trace_id:
        return None
    return int(span_context.trace_id)


# ---------------------------------------------------------------------------
# Audio hook manager
# ---------------------------------------------------------------------------


class AudioHookManager:
    """Coordinates audio streaming using active OTel span context.

    All frames are sent.  Frames inside a ``user_speaking`` or
    ``agent_speaking`` span carry that span's ID; all other frames are
    sent with an empty ``span_id``.

    Lifecycle:
        1. ``attach(session)`` patches audio I/O after ``session.start()``.
        2. ``AudioSpanProcessor`` pushes span start/end events so frames
           carry the correct span_id.
        3. Per-frame callbacks stream every frame via HTTP.
        4. ``end_all()`` closes any still-open span recordings on session close.
    """

    def __init__(self, sender: "AudioChunkSender | None" = None) -> None:
        self._sender = sender

        self._current_user_span_id: Optional[str] = None
        self._current_user_trace_id: Optional[str] = None

        self._current_agent_span_id: Optional[str] = None
        self._current_agent_trace_id: Optional[str] = None

        self._session_trace_id: Optional[str] = None

    def attach(self, session: "AgentSession") -> None:
        """Patch both audio input and output on *session*.

        Must be called after ``session.start()`` so that ``session.input``
        and ``session.output`` are initialised.
        """
        try:
            from opentelemetry import context
            from opentelemetry import trace as otrace

            span = otrace.get_current_span(context.get_current())
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                self._session_trace_id = format(ctx.trace_id, "032x")
        except Exception:
            pass
        self._wrap_audio_input(session)
        self._wrap_audio_output(session)

    # -- SpanProcessor callbacks --------------------------------------------

    def on_user_speaking_start(self, trace_id: str, span_id: str) -> None:
        self._current_user_span_id = span_id
        self._current_user_trace_id = trace_id
        logger.debug("netra.audio: user_speaking started — span_id=%s", span_id)

    def on_agent_speaking_start(self, trace_id: str, span_id: str) -> None:
        self._current_agent_span_id = span_id
        self._current_agent_trace_id = trace_id
        logger.debug("netra.audio: agent_speaking started — span_id=%s", span_id)

    def on_agent_speaking_end(self) -> None:
        if self._current_agent_span_id is not None and self._sender is not None:
            self._sender.mark_audio_end(kind="agent", span_id=self._current_agent_span_id)
        self._current_agent_span_id = None
        self._current_agent_trace_id = None

    def on_user_speaking_end(self) -> None:
        if self._current_user_span_id is not None and self._sender is not None:
            self._sender.mark_audio_end(kind="user", span_id=self._current_user_span_id)
        self._current_user_span_id = None
        self._current_user_trace_id = None

    # -- Per-frame callbacks ------------------------------------------------

    def on_user_frame(self, frame: "AudioFrame") -> None:
        """Enqueue one incoming user audio frame.

        Stamps capture-time metadata at the earliest point we see the frame.
        """
        if self._sender is None:
            return
        timestamp_ns = time.time_ns()
        trace_id = self._current_user_trace_id or self._session_trace_id or ""
        self._sender.enqueue(
            frame,
            kind="user",
            span_id=self._current_user_span_id or "",
            trace_id=trace_id,
            timestamp_ns=timestamp_ns,
        )

    def on_agent_frame(self, frame: "AudioFrame") -> None:
        """Enqueue one outgoing agent audio frame."""
        if self._sender is None:
            return
        timestamp_ns = time.time_ns()
        trace_id = self._current_agent_trace_id or self._session_trace_id or ""
        self._sender.enqueue(
            frame,
            kind="agent",
            span_id=self._current_agent_span_id or "",
            trace_id=trace_id,
            timestamp_ns=timestamp_ns,
        )

    def on_agent_flush(self) -> None:
        logger.debug("netra.audio: agent flush")

    def end_all(self) -> None:
        """End any open span recordings.  Called on session close."""
        if self._sender is not None:
            if self._current_user_span_id is not None:
                self._sender.mark_audio_end(kind="user", span_id=self._current_user_span_id)
            if self._current_agent_span_id is not None:
                self._sender.mark_audio_end(kind="agent", span_id=self._current_agent_span_id)
        self._current_user_span_id = None
        self._current_user_trace_id = None
        self._current_agent_span_id = None
        self._current_agent_trace_id = None

    # -- Session I/O patching -----------------------------------------------

    def _wrap_audio_input(self, session: "AgentSession") -> None:
        """Replace the user audio stream with a proxy that intercepts frames."""
        session_input = getattr(session, "input", None)
        if session_input is None:
            logger.warning("netra.audio: session.input not available — user audio not captured")
            return

        audio_input = getattr(session_input, "audio", None)
        if audio_input is None:
            logger.warning("netra.audio: session.input.audio not available — user audio not captured")
            return

        leaf = _leaf_audio_input(audio_input)
        proxy = _AudioInputProxy(leaf, self)

        if leaf is audio_input:
            try:
                session_input.audio = proxy
                logger.debug("netra.audio: replaced session.input.audio with proxy")
            except (AttributeError, TypeError):
                _patch_anext(leaf, self)
        else:
            parent = _parent_of_leaf(audio_input)
            if parent is not None:
                try:
                    parent.source = proxy
                    logger.debug("netra.audio: replaced leaf source with proxy")
                except (AttributeError, TypeError):
                    _patch_anext(leaf, self)
            else:
                _patch_anext(leaf, self)

    def _wrap_audio_output(self, session: "AgentSession") -> None:
        """Wrap agent ``capture_frame`` and ``flush`` to intercept outgoing audio."""
        audio_output = getattr(getattr(session, "output", None), "audio", None)
        if audio_output is None:
            logger.warning("netra.audio: session.output.audio not available — agent audio not captured")
            return

        hooks = self
        original_capture = audio_output.capture_frame
        original_flush = audio_output.flush

        @functools.wraps(original_capture)
        async def traced_capture(frame: "AudioFrame") -> None:
            try:
                hooks.on_agent_frame(frame)
            except Exception as exc:
                logger.debug("netra.audio: agent frame hook failed: %s", exc)
            await original_capture(frame)

        @functools.wraps(original_flush)
        def traced_flush() -> Any:
            try:
                hooks.on_agent_flush()
            except Exception as exc:
                logger.debug("netra.audio: agent flush hook failed: %s", exc)
            return original_flush()

        audio_output.capture_frame = traced_capture
        audio_output.flush = traced_flush
        logger.debug("netra.audio: wrapped capture_frame + flush")


# ---------------------------------------------------------------------------
# Audio input proxy
# ---------------------------------------------------------------------------


class _AudioInputProxy:
    """Transparent proxy around an async audio iterator.

    Class-level ``__aiter__`` / ``__anext__`` ensures interception works
    with ``async for`` (which resolves ``__anext__`` on the *type*, not the
    instance).
    """

    def __init__(self, original: Any, hooks: AudioHookManager) -> None:
        self._original = original
        self._hooks = hooks

    def __aiter__(self) -> "_AudioInputProxy":
        return self

    async def __anext__(self) -> "AudioFrame":
        frame: "AudioFrame" = await self._original.__anext__()
        try:
            self._hooks.on_user_frame(frame)
        except Exception as exc:
            logger.debug("netra.audio: user frame hook failed: %s", exc)
        return frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Audio input helpers
# ---------------------------------------------------------------------------


def _leaf_audio_input(audio_input: Any) -> Any:
    """Walk the ``.source`` chain to find the innermost audio source."""
    current = audio_input
    while getattr(current, "source", None) is not None:
        current = current.source
    return current


def _parent_of_leaf(audio_input: Any) -> Optional[Any]:
    """Return the object whose ``.source`` is the leaf, or ``None``."""
    current = audio_input
    while True:
        child = getattr(current, "source", None)
        if child is None:
            return None
        if getattr(child, "source", None) is None:
            return current
        current = child


def _patch_anext(leaf: Any, hooks: AudioHookManager) -> None:
    """Last-resort fallback: patch ``__anext__`` on the instance."""
    original = leaf.__anext__

    @functools.wraps(original)
    async def traced() -> "AudioFrame":
        frame = await original()
        try:
            hooks.on_user_frame(frame)
        except Exception as exc:
            logger.debug("netra.audio: user frame hook (fallback) failed: %s", exc)
        return frame

    leaf.__anext__ = traced
    logger.debug("netra.audio: fell back to __anext__ instance patch")


# ---------------------------------------------------------------------------
# Session lifecycle hooks (called from wrapt wrappers)
# ---------------------------------------------------------------------------


async def _after_start(instance: Any, session_id: Optional[str]) -> None:
    """Per-session wiring that runs once ``start()`` has returned.

    When audio capture is enabled, constructs an ``AudioChunkSender`` and
    ``AudioHookManager``, patches the session's audio I/O, and registers the
    hooks for the ``AudioSpanProcessor`` to find.
    """
    span = _session_span(instance)
    trace_id = _trace_id_of(span)
    if trace_id is None:
        logger.debug(
            "netra.livekit: no agent_session span after start(); session-scoped wiring skipped "
            "(session_id=%s). Spans still flow normally",
            session_id,
        )
        return

    logger.debug(
        "netra.livekit: agent session started session_id=%s trace_id=%032x",
        session_id,
        trace_id,
    )

    cfg = get_active_config()
    if cfg is None or not cfg.audio_capture_enabled:
        return

    try:
        from netra.instrumentation.livekit.audio_sender import AudioChunkSender
        from netra.instrumentation.livekit.processors import register_audio_hooks

        audio_url = cfg.audio_endpoint()
        if not audio_url:
            return

        api_key = cfg.api_key or ""
        auth_headers = {k: v for k, v in (cfg.headers or {}).items() if k.lower() in ("x-api-key", "authorization")}

        max_queue = max(1, cfg.audio_buffer_bytes // cfg.audio_batch_bytes) if cfg.audio_batch_bytes > 0 else 64

        sender = AudioChunkSender(
            url=audio_url,
            session_id=session_id or "",
            api_key=api_key,
            auth_headers=auth_headers,
            batch_interval=cfg.audio_batch_interval_ms / 1000.0,
            max_queue_size=max_queue,
        )

        hooks = AudioHookManager(sender=sender)

        await sender.start()
        hooks.attach(instance)

        instance._netra_audio_sender = sender
        instance._netra_audio_hooks = hooks

        register_audio_hooks(trace_id, hooks)
        logger.debug("netra.audio: audio capture attached for trace_id=%032x", trace_id)
    except Exception:
        logger.warning("netra.livekit: audio capture setup failed", exc_info=True)


async def _before_close(instance: Any) -> None:
    """Per-session teardown that runs *before* LiveKit closes the session.

    Must run before the wrapped call: ``_aclose_impl`` ends ``_session_span``
    and only then emits ``close``, after which the span is ``None`` and its
    trace id — our state key — is unreachable.

    Idempotent: the second call sees ``_session_span`` as ``None``.
    """
    span = _session_span(instance)
    trace_id = _trace_id_of(span)
    if trace_id is None:
        logger.debug("netra.livekit: session close with no live agent_session span; nothing to tear down")
        return

    logger.debug("netra.livekit: agent session closing trace_id=%032x", trace_id)

    hooks: Optional[AudioHookManager] = getattr(instance, "_netra_audio_hooks", None)
    sender: Optional["AudioChunkSender"] = getattr(instance, "_netra_audio_sender", None)

    try:
        from netra.instrumentation.livekit.processors import unregister_audio_hooks

        unregister_audio_hooks(trace_id)
    except Exception:
        pass

    if hooks is not None:
        try:
            hooks.end_all()
        except Exception:
            logger.debug("netra.audio: hooks.end_all() failed", exc_info=True)

    if sender is not None:
        try:
            await sender.end_session()
        except Exception:
            logger.debug("netra.audio: sender.end_session() failed", exc_info=True)

        if span is not None:
            try:
                span.set_attribute("netra.audio.sent_bytes", sender.stats.bytes_sent)
                span.set_attribute("netra.audio.sent_chunks", sender.stats.chunks_sent)
                span.set_attribute("netra.audio.dropped_frames", sender.stats.frames_dropped)
                span.set_attribute("netra.audio.errors", sender.stats.errors)
                span.set_attribute("netra.audio.circuit_tripped", sender.stats.circuit_tripped)
            except Exception:
                logger.debug("netra.audio: failed to stamp stats on session span", exc_info=True)

    instance._netra_audio_sender = None
    instance._netra_audio_hooks = None


# ---------------------------------------------------------------------------
# wrapt wrapper functions (public — referenced from __init__.py)
# ---------------------------------------------------------------------------


async def wrap_start(
    wrapped: WrappedAsync,
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Attach the session id around ``AgentSession.start``.

    The attach happens **before** the await, because the ``agent_session`` span is
    created inside ``start()`` and ``SessionSpanProcessor.on_start`` reads baggage
    at that moment. Attaching afterwards would leave the trace's root span as the
    one span missing ``netra.session_id``.

    The detach happens in ``finally``, in the same task, as OTel requires. Every
    LiveKit task that produces spans for this session is created *during*
    ``start()`` and snapshots the context at creation, so those tasks keep the
    baggage for their whole lifetime while the caller's context is restored.

    Documented consequence: the session id is scoped to the LiveKit session's task
    tree, not the whole job. Code running in the entrypoint task *after*
    ``await session.start(...)`` carries no session id; a user who wants that
    calls ``Netra.set_session_id()``, which is process-wide by design.

    Once ``start()`` has returned, ``_after_start`` runs the per-session wiring —
    audio capture included. It is awaited *after* the detach so its own failures
    cannot leak session context, and it is isolated so it can never surface in the
    user's ``start()`` call.

    Args:
        wrapped: LiveKit's ``AgentSession.start``.
        instance: The ``AgentSession``, needed by ``_after_start`` to reach the
            session span and the session's audio I/O.
        args: Positional arguments (``agent``).
        kwargs: Keyword arguments, including the keyword-only ``room``.

    Returns:
        Whatever ``start()`` returns, untouched.
    """
    session_id = _resolve_session_id(kwargs)

    scope = ExitStack()
    if session_id is not None:
        try:
            scope.enter_context(SessionManager.session_scope(session_id=session_id))
        except Exception:
            logger.warning("netra.livekit: could not attach session context", exc_info=True)

    try:
        result = await wrapped(*args, **kwargs)
    finally:
        try:
            scope.close()
        except Exception:
            logger.debug("netra.livekit: session context detach failed", exc_info=True)

    try:
        await _after_start(instance, session_id)
    except Exception as exc:
        logger.warning("netra.livekit: post-start wiring failed: %s", exc)

    return result


async def wrap_aclose(
    wrapped: WrappedAsync,
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Run per-session teardown before LiveKit closes the session.

    Wraps ``_aclose_impl`` rather than ``aclose``: ``aclose()`` covers only the
    ``USER_INITIATED`` close reason, while the other four — including
    ``PARTICIPANT_DISCONNECTED``, i.e. the caller hanging up — reach
    ``_aclose_impl`` directly. Wrapping ``aclose`` would mean the teardown never
    runs on a normal phone call.

    Args:
        wrapped: LiveKit's ``AgentSession._aclose_impl``.
        instance: The ``AgentSession``.
        args: Positional arguments.
        kwargs: Keyword arguments, including the ``reason``.

    Returns:
        Whatever ``_aclose_impl`` returns, untouched.
    """
    try:
        await _before_close(instance)
    except Exception as exc:
        logger.warning("netra.livekit: pre-close teardown failed: %s", exc)

    return await wrapped(*args, **kwargs)
