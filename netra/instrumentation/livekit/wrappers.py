"""wrapt wrappers for LiveKit's ``AgentSession`` lifecycle.

The session id — the LiveKit room SID, falling back to the room name — is attached
as OTel baggage *around* ``AgentSession.start`` so that the ``agent_session`` root
span, created inside ``start()``, carries it, then detached so the caller's context
is restored. See ``wrap_start`` and ``_resolve_session_id``.

Nothing in here may change the behaviour of the user's application: the hook calls
the wrapped function even if our own logic raises, and exceptions raised by the
user's code propagate untouched.
"""

import logging
from contextlib import ExitStack
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from netra.session_manager import SessionManager

logger = logging.getLogger(__name__)

# The wrapt quadruple: (wrapped, instance, args, kwargs). ``instance`` is a
# livekit AgentSession, which is not importable at module scope — the SDK must
# stay importable with livekit-agents absent.
WrappedAsync = Callable[..., Awaitable[Any]]


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

    Args:
        wrapped: LiveKit's ``AgentSession.start``.
        instance: The ``AgentSession``. Unused; part of the wrapt contract.
        args: Positional arguments (``agent``).
        kwargs: Keyword arguments, including the keyword-only ``room``.

    Returns:
        Whatever ``start()`` returns, untouched.
    """
    session_id = _resolve_session_id(kwargs)

    # ExitStack rather than a bare token so the detach is ordinary context-manager
    # unwinding: it runs in this same coroutine, on both the success and error
    # paths, and an attach failure degrades to "no session id" instead of
    # propagating into the user's start() call.
    scope = ExitStack()
    if session_id is not None:
        try:
            scope.enter_context(SessionManager.session_scope(session_id=session_id))
        except Exception:
            logger.warning("netra.livekit: could not attach session context", exc_info=True)

    try:
        return await wrapped(*args, **kwargs)
    finally:
        try:
            scope.close()
        except Exception:
            logger.debug("netra.livekit: session context detach failed", exc_info=True)
