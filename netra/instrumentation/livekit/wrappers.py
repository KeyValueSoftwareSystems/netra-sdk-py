"""wrapt wrappers for LiveKit's ``AgentSession`` lifecycle.

Three things hang off the session's lifecycle, and this module is where all of
them are bolted on:

* **the Netra session id** — the LiveKit room SID, falling back to the room name
  — attached as OTel baggage *around* ``AgentSession.start`` so the spans created
  inside it carry the id, then detached so the caller's context is restored. See
  :func:`wrap_start` and :func:`_resolve_session_id`;
* **the ``livekit-call`` span** — Netra's own root span for the call, opened
  before ``start()`` runs and left open until the session closes, so LiveKit's
  ``agent_session`` and everything under it nests inside it. The span itself, and
  the trace re-rooting it performs, live in ``call_span.py``; this module only
  decides when it opens and closes;
* **call-audio capture** — started once ``start()`` has returned and torn down
  before the session closes. The capture itself lives in ``audio_capture.py``;
  this module only decides when it begins and ends.

Nothing in here may change the behaviour of the user's application: every hook
runs the wrapped function whether or not our own logic succeeded, and exceptions
raised by the user's code propagate untouched.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from opentelemetry import trace

from netra.config import get_active_config
from netra.instrumentation.livekit.audio_capture import (
    finish_audio_capture_close,
    prepare_audio_capture_close,
    start_audio_capture,
)
from netra.instrumentation.livekit.call_span import (
    call_id_of_session,
    call_id_scope,
    end_call_span_of_session,
    start_call_span,
)
from netra.instrumentation.livekit.trace_processor import record_stt_usage
from netra.instrumentation.livekit.utils import STT_METRICS_TYPE
from netra.session_manager import SessionManager

if TYPE_CHECKING:
    from livekit.agents import AgentSession
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# The wrapt quadruple is (wrapped, instance, args, kwargs). ``instance`` is a
# livekit AgentSession, which cannot be imported at module scope — the SDK must
# stay importable with livekit-agents absent.
WrappedAsync = Callable[..., Awaitable[Any]]

# LiveKit's ``AgentSession`` event carrying every plugin's metrics.
_METRICS_EVENT = "metrics_collected"

# Instance attribute marking a session whose metrics this package already listens
# to. One subscription per session, however many times ``start()`` is called on it:
# a second listener would record every STT sample twice and double the audio
# duration and token counts the call is billed on.
_METRICS_SUBSCRIBED_FIELD = "_netra_livekit_metrics_subscribed"


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


def _session_span(instance: "AgentSession") -> Optional["Span"]:
    """Return the live ``agent_session`` span, or ``None`` once it is gone.

    Args:
        instance: The ``AgentSession``.

    Returns:
        LiveKit's own session span while the session is open.
    """
    return getattr(instance, "_session_span", None)


def _trace_id_of(session_span: Optional["Span"]) -> Optional[int]:
    """Read the trace id off the ``agent_session`` span.

    Args:
        session_span: The session span, or ``None``.

    Returns:
        The trace id, or ``None`` when there is no usable span context. This is
        the key every per-session resource is filed under, so ``None`` means the
        session gets no session-scoped wiring at all.
    """
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
# STT usage
# ---------------------------------------------------------------------------


def _subscribe_stt_usage(instance: "AgentSession") -> None:
    """Route the session's STT metrics onto the ``user_turn`` span they belong to.

    LiveKit puts LLM and TTS metrics on the spans they describe, but not the STT
    ones: ``telemetry/trace_types.py`` has no ``ATTR_STT_METRICS``, so the audio
    duration and token counts a transcription is billed on reach the SDK only as
    ``metrics_collected`` events. ``trace_processor.record_stt_usage`` matches them
    back to the recording turn by call id.

    Subscribed at most once per session, and the listener resolves its call *at
    event time* rather than capturing one. Both halves are about a ``start()`` that
    is retried after failing — a second listener would record every sample twice,
    and a captured call id would keep routing to the abandoned call.

    The listener is never removed, and needs no removal: a sample can only land on
    a turn that is registered and still recording, and by the time a call is over
    every turn it opened has ended and deregistered itself. A listener outliving
    its call therefore drops what it is handed rather than misattributing it, and
    it dies with the session either way.

    Args:
        instance: The ``AgentSession`` that is starting.
    """
    if getattr(instance, _METRICS_SUBSCRIBED_FIELD, False):
        return

    def on_metrics(event: Any) -> None:
        """Record one ``metrics_collected`` event if it is an STT one.

        Args:
            event: LiveKit's ``MetricsCollectedEvent``.
        """
        try:
            call_id = call_id_of_session(instance)
            if call_id is None:
                return
            metrics = getattr(event, "metrics", None)
            dump = getattr(metrics, "model_dump", None)
            payload = dump() if callable(dump) else metrics
            # Matched on the discriminator rather than by ``isinstance``: the same
            # event carries LLM, TTS, VAD and EOU metrics too, and the SDK must
            # stay importable with livekit-agents absent.
            if isinstance(payload, Mapping) and payload.get("type") == STT_METRICS_TYPE:
                record_stt_usage(call_id, payload)
        except Exception:
            logger.debug("netra.livekit: STT usage could not be recorded", exc_info=True)

    _listen_for_metrics(instance, on_metrics)

    try:
        setattr(instance, _METRICS_SUBSCRIBED_FIELD, True)
    except Exception:
        # A session that cannot hold the marker cannot hold the call span either,
        # so ``call_id_of_session`` returns ``None`` for it and every listener it
        # accumulates is inert. Nothing is double-counted; the feature is simply
        # off for that session.
        logger.debug("netra.livekit: could not mark the session as subscribed", exc_info=True)


def _listen_for_metrics(instance: "AgentSession", handler: Callable[[Any], None]) -> None:
    """Subscribe *handler* to the session's ``metrics_collected`` event.

    Subscribed through ``rtc.EventEmitter``, the base class, rather than
    ``AgentSession.on``: the override logs a "metrics_collected is deprecated"
    warning on every subscription, and an SDK has no business putting that in the
    user's logs for a listener the user did not add. The replacement LiveKit points
    at — ``session_usage_updated`` — reports cumulative session totals, which
    cannot be attributed to a single turn.

    If the base class is not reachable, the subscription still goes through the
    session itself: one deprecation line in the log is a smaller cost than a call
    that does not price.

    Args:
        instance: The ``AgentSession`` to subscribe to.
        handler: The callback to register.
    """
    try:
        from livekit import rtc
    except ImportError:
        logger.debug("netra.livekit: livekit.rtc is unavailable; subscribing through the session", exc_info=True)
        instance.on(_METRICS_EVENT, handler)
        return

    rtc.EventEmitter.on(instance, _METRICS_EVENT, handler)


# ---------------------------------------------------------------------------
# Session lifecycle hooks
# ---------------------------------------------------------------------------


async def _after_start(instance: "AgentSession", session_id: Optional[str]) -> None:
    """Run the per-session wiring, now that ``start()`` has returned.

    Args:
        instance: The ``AgentSession`` that has started.
        session_id: The Netra session id resolved for it, if any.
    """
    trace_id = _trace_id_of(_session_span(instance))
    if trace_id is None:
        logger.debug(
            "netra.livekit: no agent_session span after start(); session-scoped wiring skipped "
            "(session_id=%s). Spans still flow normally",
            session_id,
        )
        return

    logger.debug("netra.livekit: agent session started session_id=%s trace_id=%032x", session_id, trace_id)

    config = get_active_config()
    if config is None or not config.audio_capture_enabled:
        return

    await start_audio_capture(instance, config=config, session_id=session_id or "", trace_id=trace_id)


async def _before_close(instance: "AgentSession") -> None:
    """Prepare audio teardown *before* LiveKit closes the session.

    Stops forwarding new agent frames but leaves the sender open so LiveKit's
    ``clear_buffer`` / ``playback_finished`` during ``_aclose_impl`` can still
    report how much of the utterance was heard. The drain runs in
    :func:`_after_close`.

    Also snapshots ``_session_span`` here: LiveKit ends it inside
    ``_aclose_impl``, and the finish half still needs the object (and its
    trace id) to stamp delivery stats.

    Args:
        instance: The ``AgentSession`` that is closing.
    """
    session_span = _session_span(instance)
    trace_id = _trace_id_of(session_span)
    if trace_id is None:
        logger.debug("netra.livekit: session close with no live agent_session span; nothing to tear down")
        return

    setattr(instance, "_netra_pending_audio_close", (trace_id, session_span))
    logger.debug("netra.livekit: agent session closing trace_id=%032x", trace_id)
    await prepare_audio_capture_close(trace_id)


async def _after_close(instance: "AgentSession") -> None:
    """Drain audio after LiveKit has closed (and reported playback position)."""
    pending = getattr(instance, "_netra_pending_audio_close", None)
    if pending is None:
        return
    try:
        delattr(instance, "_netra_pending_audio_close")
    except AttributeError:
        pass
    trace_id, session_span = pending
    await finish_audio_capture_close(trace_id, session_span=session_span)


# ---------------------------------------------------------------------------
# wrapt wrapper functions (public — referenced from __init__.py)
# ---------------------------------------------------------------------------


async def wrap_start(
    wrapped: WrappedAsync,
    instance: "AgentSession",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Open the call span and attach the session id around ``AgentSession.start``.

    Both happen **before** the await, and in that order. The session id must be
    attached first because ``SessionSpanProcessor.on_start`` reads baggage at span
    creation, so anything created afterwards — the ``livekit-call`` span included —
    carries ``netra.session_id``; attaching later would leave the trace's root span
    as the one span missing it. The call span must then be made current before
    ``start()`` runs, because LiveKit creates ``agent_session`` inside ``start()``
    and it has to land underneath.

    Both are unwound through one ``ExitStack``, in this same coroutine, as OTel
    requires of context tokens — innermost (the call span) first. Every LiveKit
    task that produces spans for this session is created *during* ``start()`` and
    snapshots the context at creation, so those tasks keep both the baggage and the
    call span as an ancestor for their whole lifetime while the caller's context is
    restored.

    Documented consequence: the session id is scoped to the LiveKit session's task
    tree, not the whole job. Code running in the entrypoint task *after*
    ``await session.start(...)`` carries no session id; a user who wants that
    calls ``Netra.set_session_id()``, which is process-wide by design.

    Args:
        wrapped: LiveKit's ``AgentSession.start``.
        instance: The ``AgentSession``, needed by ``_after_start`` to reach the
            session span and the session's audio I/O, and the handle the call span
            is stored on.
        args: Positional arguments (``agent``).
        kwargs: Keyword arguments, including the keyword-only ``room``.

    Returns:
        Whatever ``start()`` returns, untouched.
    """
    session_id = _resolve_session_id(kwargs)

    try:
        # ``with`` rather than a bare ``close()`` so the unwinding sees the
        # exception: that is what lets ``use_span`` record a failing start() on the
        # call span. Each attach is isolated, so a failure degrades to a missing
        # session id or an un-nested call span instead of propagating into the
        # user's start() call — and the detaches run in this same coroutine, in
        # reverse order of attachment, as OTel requires of context tokens.
        with ExitStack() as scope:
            if session_id is not None:
                try:
                    scope.enter_context(SessionManager.session_scope(session_id=session_id))
                except Exception:
                    logger.warning("netra.livekit: could not attach session context", exc_info=True)

            call_span = None
            try:
                call_span = start_call_span(instance, session_id=session_id)
            except Exception:
                logger.warning("netra.livekit: could not open the call span", exc_info=True)

            if call_span is not None:
                try:
                    # end_on_exit=False: the span outlives start() by the whole
                    # call. Exception recording is left on, so a start() that
                    # raises marks the call span before it is ended below.
                    scope.enter_context(trace.use_span(call_span, end_on_exit=False))
                except Exception:
                    logger.warning("netra.livekit: could not make the call span current", exc_info=True)

                try:
                    # Attached alongside the call span, and for the same span of
                    # time, so every LiveKit task created inside start() carries
                    # the call id for the whole call. It is what tells this call's
                    # ``user_turn`` spans from a concurrent session's in the same
                    # job — which share a trace id, so the trace cannot.
                    scope.enter_context(call_id_scope(call_span))
                except Exception:
                    logger.warning("netra.livekit: could not attach the call id", exc_info=True)

            # Subscribed before start() so no metrics can be missed: the STT stream
            # is created inside it. Idempotent, and it reads the call it belongs to
            # off the session on each event, so it needs neither the call span here
            # nor a second subscription if start() is retried.
            try:
                _subscribe_stt_usage(instance)
            except Exception:
                logger.warning(
                    "netra.livekit: could not subscribe to session metrics; STT spans will carry "
                    "no audio duration or token counts and will not price",
                    exc_info=True,
                )

            result = await wrapped(*args, **kwargs)
    except BaseException:
        # A session that never started will never be closed, so neither end path
        # would ever run and the call span would be left open — and an unended span
        # is never exported.
        try:
            end_call_span_of_session(instance)
        except Exception:
            logger.debug("netra.livekit: could not end the call span after a failed start", exc_info=True)
        raise

    # Awaited after the detach so its own failures cannot leak session context,
    # and isolated so they can never surface in the user's start() call.
    try:
        await _after_start(instance, session_id)
    except Exception:
        logger.warning("netra.livekit: post-start wiring failed", exc_info=True)

    return result


async def wrap_aclose(
    wrapped: WrappedAsync,
    instance: "AgentSession",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Run per-session teardown around LiveKit closing the session.

    Wraps ``_aclose_impl`` rather than ``aclose``: ``aclose()`` covers only the
    ``USER_INITIATED`` close reason, while the other four — including
    ``PARTICIPANT_DISCONNECTED``, i.e. the caller hanging up — reach
    ``_aclose_impl`` directly. Wrapping ``aclose`` would mean the teardown never
    runs on a normal phone call.

    Audio teardown is split across the wrapped call on purpose:

    * **before** — stop capturing new agent frames while ``_session_span`` (and
      its trace id) still exist to key the coordinator;
    * **after** — drain the sender once LiveKit has run ``clear_buffer`` /
      ``playback_finished`` inside ``_aclose_impl``, so mid-speech disconnects
      trim to what the caller heard instead of the full buffered TTS.

    Ending the call span must still run *last*: LiveKit ends ``agent_session``
    inside ``_aclose_impl``, so ending the call span beforehand would close a
    parent before its own child.

    ``SpanMappingProcessor`` normally gets there first, off the ``agent_session``
    span ending — that path needs no method wrap and so survives a LiveKit rename
    of ``_aclose_impl``. This one is the fallback for a provider that never got the
    processors registered; both are idempotent, and whichever runs first wins.

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
    except Exception:
        logger.warning("netra.livekit: pre-close teardown failed", exc_info=True)

    try:
        return await wrapped(*args, **kwargs)
    finally:
        try:
            await _after_close(instance)
        except Exception:
            logger.warning("netra.livekit: post-close audio drain failed", exc_info=True)
        # In ``finally`` so a close that raises still closes the call span rather
        # than abandoning the trace's root.
        try:
            end_call_span_of_session(instance)
        except Exception:
            logger.warning("netra.livekit: could not end the call span", exc_info=True)
