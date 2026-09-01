"""The Netra-owned span that wraps a whole LiveKit call: ``livekit-call``.

The one span in this package Netra *creates* rather than annotates, and the trace
root for every voice call.

**Why it exists.** livekit-agents roots a job's trace at ``job_entrypoint``, which
ends the moment the user's entrypoint coroutine returns — in the ordinary
entrypoint, moments after ``await session.start(...)``. The call itself then runs
for minutes underneath an already-finished root, with three consequences:

* the root's duration is the entrypoint's, not the call's, so trace-level latency
  is meaningless for voice;
* ``netra.session_id`` is missing from the root, because the session id is only
  resolvable once ``start()`` is called and ``job_entrypoint`` predates that;
* ``netra.trace.llm.call`` is never stamped — ``LlmTraceIdentifierSpanProcessor``
  writes it on the root only while the root is still recording, and the first LLM
  span ends after ``job_entrypoint`` has closed. Voice traces were therefore
  invisible to anything keyed on that marker.

**What it does.** ``livekit-call`` is created inside ``AgentSession.start`` and
ends when the session closes, so it spans the call. It is created in the ambient
context — inheriting the job's trace id rather than starting a fresh trace, which
matters because ``audio_capture`` reads the ambient trace id after ``start()``
returns — and is then rewritten to be the trace root, with ``job_entrypoint``
rewritten to be one of its two children:

    livekit-call            ROOT
    ├── job_entrypoint            (and any work the entrypoint does itself)
    └── agent_session
        └── (LiveKit's tree, untouched)

Rewriting a parent is legal here because both spans are still recording and the
parent is only read at export; ``set_span_parent`` is the same helper the exporter
uses to reparent around dropped spans.

**When it does nothing.** The rewrite happens only when the trace really is rooted
at a live, livekit-scoped ``job_entrypoint`` that has not already been re-rooted —
whether that span is the one current at ``start()`` or sits further up, above
spans the entrypoint opened itself (a ``@workflow`` decorator on the entrypoint is
the common case; those spans keep their parents and ride along under
``job_entrypoint``). ``AgentSession.start()`` is also called outside a job (eval
mode, direct library use, a call started from a trace of the caller's own); there
``livekit-call`` is an ordinary child of whatever is current and no other span is
touched. A root the caller owns stays the root.

**Known limit: the end boundary is not enforced, only the start one.** The
backdated ``start_time`` guarantees ``livekit-call`` begins no later than
``job_entrypoint``. Nothing guarantees the reverse at the other end: the call span
closes when the session closes, while ``job_entrypoint`` — now its child — closes
when the user's entrypoint coroutine returns. An entrypoint that keeps working
after ``await session.start(...)`` for longer than the call lasts therefore ends
*after* its own parent, and the root's duration understates the trace. This is
deliberately not compensated for: holding the root open until the entrypoint
returns would reintroduce the very coupling to entrypoint lifetime that this span
exists to break, and the ordinary LiveKit entrypoint returns long before the call
ends. Assume the root covers the *call*, not every last thing the job does.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from netra.exporters.utils import set_span_parent
from netra.instrumentation.libraries.livekit.utils import (
    CALL_SPAN_NAME,
    DEFAULT_NETRA_SPAN_TYPE,
    ENTITY_TYPE_WORKFLOW,
    JOB_ENTRYPOINT_SPAN_NAME,
    LIVEKIT_SCOPE_NAME,
    LK_JOB_ID_ATTRIBUTE,
    LK_ROOM_NAME_ATTRIBUTE,
    NETRA_ENTITY_TYPE,
    NETRA_SPAN_TYPE,
)
from netra.instrumentation.libraries.livekit.version import __version__
from netra.processors.root_span_processor import RootSpanProcessor

logger = logging.getLogger(__name__)

# The tracer scope for ``livekit-call``. Spelled out rather than derived from
# ``__name__``: the instrument name is the *last* dotted component of the scope
# (``RootInstrumentFilterProcessor._resolve_instrument_name``), so ``__name__``
# here would resolve to the instrument ``call_span``, which is in no allow-list —
# and ``livekit-call`` would be peeled off as a disallowed root span.
_TRACER_NAME = "netra.instrumentation.livekit"

# Instance attribute holding a session's call span. The handle ``wrap_aclose``
# ends the span through, and the reason it must be stored somewhere: the span
# outlives ``start()`` by the whole duration of the call. Mirrors LiveKit's own
# ``AgentSession._session_span``.
CALL_SPAN_FIELD = "_netra_livekit_call_span"

# Written on ``job_entrypoint`` once its trace has been re-rooted. Both a guard
# against a second ``AgentSession`` in the same job re-rooting the trace again,
# and a visible record on the span that its parent was rewritten.
REROOTED_ATTRIBUTE = "netra.livekit.rerooted"

# Context key carrying the id of the call a span belongs to.
#
# **The call id is the call span's own span id, NOT the trace id.** A trace does
# not identify a call: ``livekit-call`` is created in the ambient context and so
# inherits the *job's* trace id rather than minting one, and the re-rooting guard
# above means a second ``AgentSession`` in the same job inherits that same trace id
# again. Two concurrent sessions in one job therefore share a trace id, and
# anything filed under one mixes their calls together — which for the STT usage
# ``SpanMappingProcessor`` files under it would mean billing one caller's audio to
# the other. A call span id is unique per call.
_CALL_ID_KEY = otel_context.create_key("netra-livekit-call-id")

# Context key carrying the name of the agent dispatched to a call.
#
# Attached beside the call id, and for the same reason: LiveKit writes the name on
# ``agent_session`` and ``job_entrypoint`` only, so an ``agent_turn`` has no way to
# reach it except through the context its creator snapshotted. Constant for the
# whole call — it names the worker the job was dispatched to, not the ``Agent``
# instance currently speaking — so a value attached once around ``start()`` stays
# correct for every turn, with no registry to keep fresh or to evict.
_AGENT_NAME_KEY = otel_context.create_key("netra-livekit-agent-name")

# Hard cap on simultaneously-open call spans, mirroring the bound
# ``RootInstrumentFilterProcessor`` puts on its own candidate registry.
#
# Entries leave the registry when the session closes, so a correct process holds
# one per *concurrent* call — a handful even on a thread-executor worker running
# many jobs in one process. The cap therefore never binds in normal operation; it
# only stops a process from accumulating live ``Span`` objects forever when
# sessions are abandoned without ever closing (a job that dies in-process, an
# ``AgentSession`` dropped on the floor). No TTL: a call span legitimately stays
# open for the whole call, so age says nothing about whether it leaked.
_MAX_OPEN_CALL_SPANS = 256

# Statuses for the two ways a call span can be closed by something other than its
# own session ending. Both mean "this call did not end cleanly", and without them
# an abandoned call is indistinguishable at the root from a healthy one.
_SHUTDOWN_STATUS = Status(StatusCode.ERROR, "livekit-call: session never closed before Netra.shutdown()")
_EVICTED_STATUS = Status(StatusCode.ERROR, f"livekit-call: evicted, more than {_MAX_OPEN_CALL_SPANS} open call spans")


class _CallSpanRegistry:
    """Finds a call span from the span id of a child it parents.

    ``SpanMappingProcessor`` is registered once per process but sees the
    ``agent_session`` spans of every concurrent call, so it needs a way from an
    ending ``agent_session`` back to the ``livekit-call`` that wraps it. Keyed on
    the call span's own span id — which is exactly the ending span's parent span
    id — rather than on the trace id: a job that runs two sessions puts two call
    spans in one trace, and a trace-keyed registry would let the second session's
    close end the first session's span.

    Bounded at ``_MAX_OPEN_CALL_SPANS``, oldest first, because every entry pins a
    live ``Span`` and nothing but a session close removes one.

    Locked because ``Netra.shutdown()`` reaches :meth:`pop_all` from whichever
    thread called it, while registration and lookup happen on the agent's event
    loop.
    """

    def __init__(self) -> None:
        """Start with no calls registered."""
        self._by_span_id: "OrderedDict[int, Span]" = OrderedDict()
        self._lock = threading.Lock()

    def register(self, span_id: int, span: Span) -> List[Span]:
        """Record a call span under its own span id, evicting the oldest if full.

        Args:
            span_id: The call span's own span id.
            span: The ``livekit-call`` span.

        Returns:
            The call spans evicted to make room, which the caller owns and must
            end. Returned rather than ended here because ``Span.end()`` runs the
            whole span-processor chain synchronously, and that chain reaches back
            into this registry (``SpanMappingProcessor.on_end`` →
            :func:`end_call_span_parenting` → :meth:`unregister`). Ending under
            ``self._lock`` would make that re-entrant, which a plain
            ``threading.Lock`` does not survive.
        """
        with self._lock:
            self._by_span_id[span_id] = span
            self._by_span_id.move_to_end(span_id)
            evicted: List[Span] = []
            while len(self._by_span_id) > _MAX_OPEN_CALL_SPANS:
                _oldest_span_id, oldest_span = self._by_span_id.popitem(last=False)
                evicted.append(oldest_span)
        return evicted

    def unregister(self, span_id: int) -> Optional[Span]:
        """Remove and return a call span. Idempotent.

        The atomic claim that decides which of the two end paths actually ends the
        span: whoever pops the entry owns the ``end()``.

        Args:
            span_id: The call span's own span id.

        Returns:
            The span that was registered, or ``None`` if it is already gone.
        """
        with self._lock:
            return self._by_span_id.pop(span_id, None)

    def pop_all(self) -> List[Span]:
        """Remove and return every registered call span.

        Returns:
            The call spans that were still open.
        """
        with self._lock:
            spans = list(self._by_span_id.values())
            self._by_span_id.clear()
        return spans


call_spans = _CallSpanRegistry()


# ---------------------------------------------------------------------------
# Starting the call span
# ---------------------------------------------------------------------------


def start_call_span(instance: Any, *, session_id: Optional[str] = None) -> Optional[Span]:
    """Open a ``livekit-call`` span for *instance* and re-root its trace.

    Must be called with the session-id context already attached, so
    ``SessionSpanProcessor`` stamps ``netra.session_id`` on the call span itself.

    Args:
        instance: The ``AgentSession`` the call belongs to. The span is stored on
            it so ``wrap_aclose`` can end it.
        session_id: The resolved Netra session id, for the debug log only — the
            attribute itself comes from the attached context.

    Returns:
        The started span, or ``None`` when it could not be created, in which case
        the trace keeps the shape it has today.
    """
    job_entrypoint = _job_entrypoint_to_reroot(trace.get_current_span())

    tracer = trace.get_tracer(_TRACER_NAME, __version__)
    # Backdated to the job entrypoint's start so the call span fully encloses both
    # of its children. Without it ``job_entrypoint`` — now a child — would begin
    # before its parent.
    start_time = _start_time_of(job_entrypoint) if job_entrypoint is not None else None
    span = tracer.start_span(CALL_SPAN_NAME, start_time=start_time)

    _stamp_markers(span)

    if job_entrypoint is not None:
        _reroot_trace(span, job_entrypoint)

    span_id = _span_id_of(span)
    if span_id is not None:
        # Ended out here rather than inside ``register`` — see its docstring for
        # why that would deadlock. Logged rather than dropped quietly: reaching
        # the cap means calls are being abandoned, which is worth knowing about.
        for evicted in call_spans.register(span_id, span):
            logger.warning(
                "netra.livekit: evicting an open %s span; more than %d calls started without ever closing",
                CALL_SPAN_NAME,
                _MAX_OPEN_CALL_SPANS,
            )
            _end_unclaimed(evicted, status=_EVICTED_STATUS)
    _store_on_session(instance, span)

    logger.debug(
        "netra.livekit: opened %s session_id=%s rerooted=%s",
        CALL_SPAN_NAME,
        session_id,
        job_entrypoint is not None,
    )
    return span


@contextmanager
def call_id_scope(call_span: Span) -> Iterator[None]:
    """Attach the id of the call *call_span* opened, for the duration of the block.

    Entered around ``AgentSession.start``, so every LiveKit task created inside it
    — each of which snapshots the context at creation — carries the call id for the
    whole call, exactly as it carries the session id and the call span itself. That
    is what lets ``SpanMappingProcessor.on_start`` tell one call's ``user_turn``
    spans from another's without depending on where LiveKit happens to nest them.

    Args:
        call_span: The ``livekit-call`` span identifying this call.

    Yields:
        ``None``, with the call id attached to the context. A call span with no
        usable span id yields without attaching anything, leaving its spans
        unidentifiable rather than filed under a wrong id.
    """
    call_id = _span_id_of(call_span)
    if call_id is None:
        yield
        return

    token = otel_context.attach(otel_context.set_value(_CALL_ID_KEY, call_id))
    try:
        yield
    finally:
        otel_context.detach(token)


def call_id_of(context: Optional[otel_context.Context] = None) -> Optional[int]:
    """Read the id of the call *context* belongs to.

    Args:
        context: The context to read, or ``None`` for the ambient one. ``on_start``
            is handed ``None`` whenever the span's creator relied on the ambient
            context — the usual case — and the ambient context at that moment is
            the one the span is being parented to.

    Returns:
        The call id, or ``None`` outside a call.
    """
    call_id = otel_context.get_value(_CALL_ID_KEY, context=context)
    return call_id if isinstance(call_id, int) else None


@contextmanager
def agent_name_scope(agent_name: Optional[str]) -> Iterator[None]:
    """Attach the dispatched agent's name for the duration of the block.

    Entered around ``AgentSession.start`` alongside :func:`call_id_scope`, so every
    context LiveKit snapshots inside ``start()`` carries the name — including
    ``AgentSession._root_span_context``, which is the context every ``agent_turn``
    span is created in.

    Args:
        agent_name: The name to attach. ``None`` or empty attaches nothing: LiveKit
            leaves ``job.agent_name`` empty for a worker that declares none, and an
            empty name is worse than an absent one on a span.

    Yields:
        ``None``, with the agent name attached to the context.
    """
    if not agent_name:
        yield
        return

    token = otel_context.attach(otel_context.set_value(_AGENT_NAME_KEY, agent_name))
    try:
        yield
    finally:
        otel_context.detach(token)


def agent_name_of(context: Optional[otel_context.Context] = None) -> Optional[str]:
    """Read the name of the agent dispatched to the call *context* belongs to.

    Args:
        context: The context to read, or ``None`` for the ambient one. As with
            :func:`call_id_of`, ``on_start`` is handed ``None`` whenever the span's
            creator relied on the ambient context.

    Returns:
        The agent name, or ``None`` outside a call and for a worker that declares
        no ``agent_name``.
    """
    agent_name = otel_context.get_value(_AGENT_NAME_KEY, context=context)
    return agent_name if isinstance(agent_name, str) and agent_name else None


def call_id_of_session(instance: Any) -> Optional[int]:
    """Return the id of the call *instance* is currently on.

    The counterpart to :func:`call_id_scope` for code that holds the session rather
    than the context — resolved on each read, not captured, so a session whose
    ``start()`` is retried reports the call it is on *now* rather than a dead one.

    Args:
        instance: The ``AgentSession``.

    Returns:
        The call id, or ``None`` when the session holds no usable call span —
        before one was opened, or after opening one failed.
    """
    return _span_id_of(getattr(instance, CALL_SPAN_FIELD, None))


def _reroot_trace(span: Span, job_entrypoint: Span) -> None:
    """Make *span* the trace root and *job_entrypoint* its child.

    Args:
        span: The freshly started ``livekit-call`` span, currently a child of
            *job_entrypoint*.
        job_entrypoint: LiveKit's live ``job_entrypoint`` span.
    """
    span_context = span.get_span_context()
    set_span_parent(span, None)
    set_span_parent(job_entrypoint, span_context)
    # Marked after the rewrite so a failure above leaves the guard unset and the
    # next session can still try.
    job_entrypoint.set_attribute(REROOTED_ATTRIBUTE, True)
    # ``RootSpanProcessor.on_start`` already recorded ``job_entrypoint`` as this
    # trace's root and records with ``setdefault``, so the move has to be stated.
    RootSpanProcessor.replace_root_span(span)


def _job_entrypoint_to_reroot(current: Any) -> Optional[Span]:
    """The live ``job_entrypoint`` this call should be re-rooted onto, if any.

    Two spans are tested, not one, because the entrypoint is not obliged to await
    ``session.start()`` directly:

    * the span current at ``start()`` — the ordinary entrypoint, where that span
      *is* ``job_entrypoint``;
    * the trace's recorded root — an entrypoint that opens a span of its own
      first, a Netra ``@workflow`` decorator on the entrypoint being the common
      case. Such a span displaces ``job_entrypoint`` from the current-span slot
      without changing what it is: still the root, still ending the moment the
      entrypoint returns. Re-rooting is a statement about the root, so the root
      is what has to be tested; whatever the entrypoint opened in between keeps
      the parent it was created with and rides along under ``job_entrypoint``.

    The current-span test is kept rather than folded into the root one: a job
    process that restored a parent trace context
    (``extract_subprocess_context``) gives ``job_entrypoint`` a remote parent, so
    ``RootSpanProcessor`` never records it as a root and only the current span
    can find it.

    Args:
        current: The span current at ``AgentSession.start()``.

    Returns:
        The span to re-root onto, or ``None`` to leave the trace as it is — no
        job, an already-re-rooted job, or a root the caller owns.
    """
    if _is_live_job_entrypoint(current):
        return current

    root = RootSpanProcessor.get_root_span(current)
    return root if _is_live_job_entrypoint(root) else None


def _is_live_job_entrypoint(span: Any) -> bool:
    """Whether *span* is a live, livekit-scoped, not-yet-re-rooted ``job_entrypoint``.

    Args:
        span: The candidate span, or ``None`` when there is none to test.

    Returns:
        ``True`` only when re-rooting the trace onto it is correct.
    """
    if getattr(span, "name", None) != JOB_ENTRYPOINT_SPAN_NAME:
        return False

    scope = getattr(span, "instrumentation_scope", None)
    if getattr(scope, "name", None) != LIVEKIT_SCOPE_NAME:
        return False

    if not _is_recording(span):
        return False

    # A job that runs a second AgentSession must not re-root the trace twice: the
    # first call span is already the root, and rewriting job_entrypoint's parent
    # again would move it under the second call.
    attributes = getattr(span, "attributes", None) or {}
    return not attributes.get(REROOTED_ATTRIBUTE)


# ---------------------------------------------------------------------------
# Ending the call span
# ---------------------------------------------------------------------------


def end_call_span_of_session(instance: Any) -> None:
    """End the call span belonging to *instance*, if it is still open.

    The fallback end path, used by ``wrap_aclose``. Idempotent.

    Args:
        instance: The ``AgentSession`` that is closing.
    """
    span = getattr(instance, CALL_SPAN_FIELD, None)
    if span is None:
        return
    _end(span)


def end_call_span_parenting(child_parent_span_id: Optional[int], *, status: Optional[Status] = None) -> None:
    """End the call span whose own span id is *child_parent_span_id*, if any.

    The primary end path: called when a livekit ``agent_session`` span ends, which
    is LiveKit's own authoritative "the call is over" signal and reaches us on
    every close reason without depending on a method wrap. A no-op when the ending
    span is not a direct child of a call span. Idempotent.

    Args:
        child_parent_span_id: The parent span id of the ending ``agent_session``.
        status: The status to close the call span with, from
            :func:`failure_status_of` — ``None`` to leave it ``UNSET``, which is a
            call that ended normally.
    """
    if child_parent_span_id is None:
        return
    span = call_spans.unregister(child_parent_span_id)
    if span is None:
        return
    _end_unclaimed(span, status=status)


def failure_status_of(span: Any) -> Optional[Status]:
    """Mirror an ``agent_session`` that ended in error onto its call span.

    The call span is the trace root, so it is what anything keyed on trace-level
    health reads. Left to itself it always closes ``UNSET``, which would make a
    call that died mid-way indistinguishable from a clean one without walking the
    children.

    Only the OTel status is mirrored, not LiveKit's close reason: the reason is
    already on ``agent_session`` under LiveKit's own attribute key, and copying it
    would mean hard-coding a key from a library this package deliberately reads
    through name and scope only.

    Args:
        span: The ending ``agent_session`` span.

    Returns:
        An ``ERROR`` status carrying the child's description, or ``None`` when the
        session did not end in error.
    """
    status = getattr(span, "status", None)
    if getattr(status, "status_code", None) is not StatusCode.ERROR:
        return None
    description = getattr(status, "description", None)
    return Status(StatusCode.ERROR, description or "livekit-call: agent_session ended in error")


def end_all_call_spans() -> None:
    """End every call span still open. Backstop for ``Netra.shutdown()``.

    Must run *before* the tracer provider is flushed and shut down, or the spans
    ended here never reach the exporter — and a process that exits mid-call would
    lose the call's root span, not merely close it late.

    Anything closed here is closed with an ``ERROR`` status: reaching this path at
    all means the process is exiting with a call still in flight, which is not a
    clean end and should not be exported as one.
    """
    spans = call_spans.pop_all()
    if not spans:
        return

    logger.info("netra.livekit: closing %d call span(s) whose session never closed", len(spans))
    for span in spans:
        _end_unclaimed(span, status=_SHUTDOWN_STATUS)


def _end(span: Span) -> None:
    """End *span* exactly once, whichever path gets here first.

    Args:
        span: The call span to end.
    """
    span_id = _span_id_of(span)
    if span_id is None:
        return
    # Popping the registry entry is the atomic claim on the ``end()``: the loser
    # of the race finds nothing and returns.
    if call_spans.unregister(span_id) is None:
        return
    _end_unclaimed(span)


def _end_unclaimed(span: Span, *, status: Optional[Status] = None) -> None:
    """End a span whose registry entry the caller has already claimed.

    Args:
        span: The call span to end.
        status: The status to set before ending, or ``None`` to leave whatever the
            span already carries — which is ``UNSET`` for a clean call, and the
            ``ERROR`` ``trace.use_span`` recorded for a ``start()`` that raised.
    """
    if status is not None:
        try:
            span.set_status(status)
        except Exception:
            logger.debug("netra.livekit: could not set the %s span status", CALL_SPAN_NAME, exc_info=True)
    try:
        span.end()
    except Exception:
        logger.debug("netra.livekit: could not end the %s span", CALL_SPAN_NAME, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stamp_markers(span: Span) -> None:
    """Write the Netra classification markers and job identifiers on the call span.

    Stamped here rather than by ``SpanMappingProcessor``, which is gated on the
    ``livekit-agents`` scope and so never sees this span.

    Args:
        span: The call span to stamp.
    """
    try:
        span.set_attribute(NETRA_SPAN_TYPE, DEFAULT_NETRA_SPAN_TYPE.value)
        # The workflow entity moved here from ``job_entrypoint``: this is now the
        # one span wrapping everything the job does for the call.
        span.set_attribute(NETRA_ENTITY_TYPE, ENTITY_TYPE_WORKFLOW)
    except Exception:
        logger.debug("netra.livekit: could not stamp the call span markers", exc_info=True)

    for key, value in _job_identifiers().items():
        try:
            span.set_attribute(key, value)
        except Exception:
            logger.debug("netra.livekit: could not stamp %s on the call span", key, exc_info=True)


def _job_identifiers() -> Dict[str, str]:
    """Read the job id and room name off the job assignment.

    Read from the job context rather than copied off ``job_entrypoint``'s
    attributes so the call span carries them even when there is no entrypoint span
    to copy from.

    Returns:
        The identifiers that resolved, which is nothing at all outside a job.
    """
    identifiers: Dict[str, str] = {}
    try:
        from livekit.agents import get_job_context

        job_context = get_job_context(required=False)
    except Exception:
        logger.debug("netra.livekit: could not read the job context", exc_info=True)
        return identifiers

    if job_context is None:
        return identifiers

    try:
        job = job_context.job
        job_id = getattr(job, "id", None)
        if isinstance(job_id, str) and job_id:
            identifiers[LK_JOB_ID_ATTRIBUTE] = job_id
        room_name = getattr(getattr(job, "room", None), "name", None)
        if isinstance(room_name, str) and room_name:
            identifiers[LK_ROOM_NAME_ATTRIBUTE] = room_name
    except Exception:
        logger.debug("netra.livekit: could not read the job identifiers", exc_info=True)

    return identifiers


def _store_on_session(instance: Any, span: Span) -> None:
    """Store *span* on *instance* so the close path can find it.

    Args:
        instance: The ``AgentSession``.
        span: The call span.
    """
    try:
        setattr(instance, CALL_SPAN_FIELD, span)
    except Exception:
        # A session that cannot hold the attribute keeps the registry-based end
        # path, which is the one that runs in practice anyway.
        logger.debug("netra.livekit: could not store the call span on the session", exc_info=True)


def _start_time_of(span: Any) -> Optional[int]:
    """Return *span*'s start time in nanoseconds, or ``None``.

    Args:
        span: The span to read.

    Returns:
        The start time, or ``None`` when the span does not expose one — in which
        case the call span simply starts now.
    """
    start_time = getattr(span, "start_time", None)
    return start_time if isinstance(start_time, int) else None


def _span_id_of(span: Any) -> Optional[int]:
    """Return *span*'s own span id, or ``None``.

    Args:
        span: The span to read.

    Returns:
        The span id, or ``None`` when there is no usable span context.
    """
    try:
        span_context = span.get_span_context()
    except Exception:
        return None
    span_id = getattr(span_context, "span_id", None)
    return span_id if isinstance(span_id, int) and span_id else None


def _is_recording(span: Any) -> bool:
    """Whether *span* is still recording.

    Args:
        span: The span to test.

    Returns:
        ``True`` if the span reports that it is recording.
    """
    try:
        return bool(span.is_recording())
    except Exception:
        return False


__all__ = [
    "CALL_SPAN_FIELD",
    "REROOTED_ATTRIBUTE",
    "agent_name_of",
    "agent_name_scope",
    "call_id_of",
    "call_id_of_session",
    "call_id_scope",
    "call_spans",
    "end_all_call_spans",
    "end_call_span_of_session",
    "end_call_span_parenting",
    "failure_status_of",
    "start_call_span",
]
