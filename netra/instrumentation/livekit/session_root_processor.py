"""Re-roots a LiveKit voice trace from ``job_entrypoint`` onto ``agent_session``.

The third of this package's span processors, and the only one that changes a
span's *position* in the trace rather than what is written on it (``trace_processor``)
or read off it (``audio_processor``).

The shape livekit-agents produces, and why it is wrong for a voice trace::

    job_entrypoint    2.3s   <- trace root; no session id, no conversation
    └── agent_session  27.8s  <- the actual call

``job_entrypoint`` wraps the user's entrypoint coroutine, which returns as soon as
it has called ``session.start()`` and ``ctx.connect()``. So the root span ends
seconds into a call that runs for minutes, and every consumer that reads a trace's
identity off its root span — the trace name, duration, session id, input/output —
reads it off the wrong span. Downstream ingestion (Langfuse and friends) sees a
2-second trace whose subtree outlives it.

This processor moves the root to ``agent_session``, which livekit-agents opens
inside ``start()`` and ends when the session closes, and under which every other
LiveKit span already sits. The trace id is untouched: only the parent link is
cleared, so a trace keeps the id it was ingested under.

Clearing ``agent_session``'s parent happens at ``on_start``, and that timing is the
point: it must not wait for export, because the exporter's drop-and-reparent path
can only promote a child whose dropped ancestor is still resolvable — and
``job_entrypoint`` ends (and is evicted from the candidate registry ten minutes
later) long before a call is over.

``job_entrypoint``'s own drop is decided in two steps, because the two things that
decide it become knowable at different times:

* **Marked at its start, provisionally.** A span the entrypoint traced *before*
  ``session.start()`` can end — and, under ``disable_batch``, export — while the
  entrypoint is still running. It can only be reparented past ``job_entrypoint`` if
  the mark already exists when it exports, so waiting for a session to appear would
  trade a stray root for something worse: a span whose parent never arrives.
* **Released at its end, if nothing claimed the root.** A LiveKit job is not
  necessarily a voice job: an entrypoint that opens no ``AgentSession`` at all, or
  one whose ``session.start()`` raised before the session span existed, still
  produced a ``job_entrypoint``, and dropping that would leave the job's own spans
  as parentless siblings with nothing above them. By the time it ends, whether a
  session took the root is settled, so the mark can be released and the job keeps
  the only root it has.

The release is a ``on_end`` hook on a processor registered *after* the exporting
one, so it lands before export only while spans are buffered — the default
``BatchSpanProcessor``. Under ``disable_batch`` (``SimpleSpanProcessor`` exports
synchronously, from an earlier link in the same chain) the release is too late and
a session-less job still fragments, exactly as it would without the hook. That
degrades toward extra roots, never toward a dangling parent, which is the trade
this ordering is chosen for.

Nothing here can change the shape of a *non-job* trace: a session started outside
a LiveKit job — console mode, evals, ``AgentSession`` used as a library, or a
``session.start()`` nested inside the user's own span — has no ``job_entrypoint``
parent, so the re-rooting does not fire and the session span keeps the parent it
was created with.

Two accepted consequences, both inherent to dropping the span that used to hold a
job's spans together rather than defects in how it is done:

* **A voice job can export more than one root.** Anything the entrypoint traced
  *before* ``session.start()`` — a ``@workflow``, a manual span — was a sibling of
  the session under ``job_entrypoint``, and once that is gone it has no parent left
  to point at, so it becomes a second root of the same trace. (Spans from an
  instrumentation that is not on the root allow-list, ``httpx`` and friends, are
  peeled recursively by the exporter instead, so they never surface this way.) The
  alternative — reparenting them onto ``agent_session`` — would file spans that
  *ended* before the session began underneath it, which is a worse lie than a
  sibling root.
* **Only the first session of a job is re-rooted.** livekit-agents allows several
  ``AgentSession``\\ s per job (``JobContext._primary_agent_session``), and each one
  resolves ``job_entrypoint`` as its parent because ``start()`` detaches the
  previous session's context first. Promoting each in turn would leave
  ``RootSpanProcessor`` naming whichever started last, so the LLM-call marker and
  ``Netra.set_attribute_on_root_span`` would write the *first* session's data onto
  the *second* session's span. The trace root is claimed once, by the first session
  to start while no other is live; later sessions keep their parent and reach the
  export as roots through the ordinary drop-and-reparent path, without disturbing
  the bookkeeping. Sessions that run one after another are each re-rooted normally,
  because the mapping is released when the recorded root ends.
"""

import logging
from typing import Any, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from netra.exporters.utils import set_span_parent
from netra.instrumentation.livekit.utils import (
    AGENT_SESSION_SPAN_NAME,
    JOB_ENTRYPOINT_SPAN_NAME,
    LIVEKIT_SCOPE_NAME,
    NETRA_LIVEKIT_REROOTED,
)
from netra.processors.root_instrument_filter_processor import (
    mark_as_root_block_candidate,
    refresh_root_block_candidate_parent,
    unmark_root_block_candidate,
)
from netra.processors.root_span_processor import RootSpanProcessor

logger = logging.getLogger(__name__)


class SessionRootSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Makes ``agent_session`` the root of a LiveKit job's trace, and drops the old root.

    Holds no state, so it needs no lock. Every decision is made from the span in
    hand — its name, its parent, the root already recorded for its trace, and, for
    the one decision that spans both hooks, an attribute written on the span itself
    rather than kept here (see ``NETRA_LIVEKIT_REROOTED``).

    Acts only on the two span names it is written for, and only on spans from the
    ``livekit-agents`` scope. Every other span — including a user span that happens
    to be called ``agent_session`` or ``job_entrypoint`` — passes through untouched.
    """

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Re-root the trace when the span being started is a job's first session.

        Args:
            span: The span that was started.
            parent_context: The context the span was created in. ``None`` when the
                creator passed no explicit context — which is what livekit-agents
                does — in which case the ambient context is the parent, and reading
                it here is correct because ``on_start`` runs synchronously inside
                ``start_span``.
        """
        try:
            if not _is_livekit_span(span):
                return

            if span.name == JOB_ENTRYPOINT_SPAN_NAME:
                # Provisional: see ``on_end``. Marking now — rather than once a
                # session actually claims the root — is what lets a span the
                # entrypoint traced *before* ``session.start()`` be reparented past
                # this one instead of exporting with a parent that never arrives.
                mark_as_root_block_candidate(span)
                return

            if span.name == AGENT_SESSION_SPAN_NAME:
                job_entrypoint = _job_entrypoint_parent(parent_context)
                if job_entrypoint is not None:
                    self._promote_to_trace_root(span, job_entrypoint)
        except Exception:
            logger.warning("netra.livekit: could not re-root the voice trace", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """Release ``job_entrypoint``'s provisional drop when no session replaced it.

        A LiveKit job is not necessarily a voice job. An entrypoint that opens no
        ``AgentSession``, or one whose ``session.start()`` raised before the session
        span existed, still produced a ``job_entrypoint``, and dropping that would
        leave the job's own spans as parentless siblings with nothing above them.
        By the time it ends, whether a session took the root is settled, and
        ``NETRA_LIVEKIT_REROOTED`` is where the promotion recorded the answer.

        Reading the answer off the span rather than off ``RootSpanProcessor`` is
        what makes this correct for a session that closes *before* the entrypoint
        returns: the root mapping is released when the recorded root ends, so a
        short call would otherwise look exactly like a job that opened no session
        at all, and the entrypoint would come back as a second root.

        Args:
            span: The span that has ended.
        """
        try:
            if not _is_livekit_span(span) or span.name != JOB_ENTRYPOINT_SPAN_NAME:
                return

            span_context = span.get_span_context()
            if span_context is None or not span_context.is_valid:
                return

            if _read_bool_attribute(span, NETRA_LIVEKIT_REROOTED):
                return

            unmark_root_block_candidate(span)
            logger.debug("netra.livekit: job_entrypoint opened no agent_session; keeping it as the trace root")
        except Exception:
            logger.debug("netra.livekit: could not settle the job_entrypoint drop decision", exc_info=True)

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
    def _promote_to_trace_root(span: Span, job_entrypoint: trace_api.Span) -> None:
        """Clear *span*'s parent link, record it as its trace's root, drop the old one.

        Clearing the parent keeps the trace id: the id was assigned when
        ``job_entrypoint`` started and lives on the span context, which this does
        not touch. Only the exported parent reference changes, and the SDK reads
        that when it snapshots the span at end — so mutating it now is what the
        exporter sees.

        The ``RootSpanProcessor`` update is not cosmetic. That processor runs
        earlier in the chain (registered by ``netra/tracer.py``, before any
        instrumentation's own processors), so it has already filed
        ``job_entrypoint`` as this trace's root. Left uncorrected, the LLM-call
        marker and ``Netra.set_attribute_on_root_span`` would target a span that is
        about to be dropped, and then nothing at all once it ends a few seconds
        into the call.

        ``job_entrypoint`` was already marked provisionally when it started; this
        promotion is what confirms the mark, and ``on_end`` releases it if no
        promotion ever arrives.

        Args:
            span: The ``agent_session`` span to promote.
            job_entrypoint: The live ``job_entrypoint`` span it was started under,
                dropped at export in its place.
        """
        span_context = span.get_span_context()
        if span_context is None or not span_context.is_valid:
            # Nothing to key the bookkeeping by, and a span in this state is not
            # going to be exported either. Leave the entrypoint alone.
            return

        entrypoint_context = job_entrypoint.get_span_context()
        entrypoint_span_id = entrypoint_context.span_id if entrypoint_context is not None else None
        if _trace_root_claimed_by_another(span_context.trace_id, entrypoint_span_id):
            logger.debug("netra.livekit: a sibling agent_session already roots this trace; leaving this one in place")
            return

        # Confirms the provisional mark ``job_entrypoint`` was given at its own
        # start. Recorded on the span rather than in this processor because
        # ``on_end`` is handed a snapshot that carries only the attribute map.
        job_entrypoint.set_attribute(NETRA_LIVEKIT_REROOTED, True)

        set_span_parent(span, None)
        # The candidate registry snapshotted this span's parent before the line
        # above, and it is the registry — not the span — that the exporter reads.
        # Left stale, a deployment whose ``root_instruments`` excludes livekit would
        # judge the promoted span against a parent link it no longer has.
        refresh_root_block_candidate_parent(span)

        if not RootSpanProcessor.replace_root_span(span):
            logger.debug(
                "netra.livekit: agent_session reparented to the trace root, but the root-span "
                "mapping was not updated; root-span attributes may not land"
            )
            return

        logger.debug("netra.livekit: agent_session promoted to trace root; job_entrypoint will be dropped")


def _is_livekit_span(span: Any) -> bool:
    """Whether *span* was produced by livekit-agents' own instrumentation.

    Args:
        span: The span to test.

    Returns:
        True only for spans whose instrumentation scope is ``livekit-agents``.
    """
    scope = getattr(span, "instrumentation_scope", None)
    return getattr(scope, "name", None) == LIVEKIT_SCOPE_NAME


def _read_bool_attribute(span: ReadableSpan, key: str) -> bool:
    """Read a boolean attribute off an ended span's attribute map.

    Args:
        span: The ended span to read from.
        key: The attribute key.

    Returns:
        True only when the attribute is present and truthy. ``attributes`` may be a
        read-only proxy or ``None``; neither is an error here.
    """
    attributes = getattr(span, "attributes", None)
    if not attributes:
        return False
    return bool(attributes.get(key))


def _job_entrypoint_parent(parent_context: Optional[otel_context.Context]) -> Optional[trace_api.Span]:
    """Return the live ``job_entrypoint`` span the starting span sits under, if any.

    Resolves the *live parent span*, not the parent span context, for two reasons:
    a ``SpanContext`` carries only ids — nothing that says which span it belongs to
    — and the caller needs the span object itself in order to mark it. The parent
    is reachable here, and only here: ``on_start`` runs inside ``start_span``, so it
    is still the current span and still recording.

    Both the name and the instrumentation scope are checked, so a user span named
    ``job_entrypoint`` cannot cause a session to be torn out of its trace position.

    Args:
        parent_context: The context passed to ``on_start``; ``None`` means the
            ambient context, which is where livekit-agents' implicit parenting
            resolves from.

    Returns:
        The parent span when it is livekit-agents' ``job_entrypoint``, else ``None``.
    """
    context = parent_context if parent_context is not None else otel_context.get_current()
    parent_span = trace_api.get_current_span(context)
    if parent_span is None:
        return None
    if getattr(parent_span, "name", None) != JOB_ENTRYPOINT_SPAN_NAME or not _is_livekit_span(parent_span):
        return None
    return parent_span


def _trace_root_claimed_by_another(trace_id: int, entrypoint_span_id: Optional[int]) -> bool:
    """Whether a live span other than ``job_entrypoint`` is on record as this trace's root.

    Answers two questions with one lookup, because they are the same question asked
    at different times: at a session's start, "has a sibling session already taken
    the root?"; at ``job_entrypoint``'s end, "did any session take it?".

    The recorded root being ``job_entrypoint`` itself is the ordinary case — that is
    the entry a promotion exists to replace — and does not count as claimed.

    Args:
        trace_id: The trace to inspect.
        entrypoint_span_id: The ``job_entrypoint`` span's own ID, or ``None`` when
            it could not be read, in which case any recorded root counts as another.

    Returns:
        True when some other span is on record as this trace's root and is still
        recording.
    """
    recorded_root = RootSpanProcessor.get_root_span_by_trace_id(trace_id)
    if recorded_root is None:
        return False

    recorded_context = recorded_root.get_span_context()
    if recorded_context is None:
        return False
    if entrypoint_span_id is not None and recorded_context.span_id == entrypoint_span_id:
        return False

    return bool(recorded_root.is_recording())


__all__ = ["SessionRootSpanProcessor"]
