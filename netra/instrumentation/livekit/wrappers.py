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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Tuple

from opentelemetry import trace

from netra.config import get_active_config
from netra.instrumentation.livekit.audio_capture import start_audio_capture, stop_audio_capture
from netra.instrumentation.livekit.call_span import end_call_span_of_session, start_call_span
from netra.session_manager import SessionManager

if TYPE_CHECKING:
    from livekit.agents import AgentSession
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# The wrapt quadruple is (wrapped, instance, args, kwargs). ``instance`` is a
# livekit AgentSession, which cannot be imported at module scope — the SDK must
# stay importable with livekit-agents absent.
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
    """Run the per-session teardown, *before* LiveKit closes the session.

    Ordering is load-bearing: ``_aclose_impl`` ends ``_session_span`` before it
    emits ``close``, after which the span is gone and its trace id — the key
    every per-session resource is filed under — is unreachable.

    Idempotent, in two layers: a second call finds no ``_session_span``, and the
    coordinator registry only hands out a coordinator once.

    Args:
        instance: The ``AgentSession`` that is closing.
    """
    session_span = _session_span(instance)
    trace_id = _trace_id_of(session_span)
    if trace_id is None:
        logger.debug("netra.livekit: session close with no live agent_session span; nothing to tear down")
        return

    logger.debug("netra.livekit: agent session closing trace_id=%032x", trace_id)
    await stop_audio_capture(trace_id, session_span=session_span)


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

    The two halves sit on opposite sides of the wrapped call, and both placements
    are load-bearing. The audio teardown must run *first*, while ``_session_span``
    still exists to key it by. Ending the call span must run *last*: LiveKit ends
    ``agent_session`` inside ``_aclose_impl``, so ending the call span beforehand
    would close a parent before its own child.

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
        # In ``finally`` so a close that raises still closes the call span rather
        # than abandoning the trace's root.
        try:
            end_call_span_of_session(instance)
        except Exception:
            logger.warning("netra.livekit: could not end the call span", exc_info=True)
