"""Streams captured call audio to the Netra audio-ingest endpoint.

:class:`SessionAudioCoordinator` hands frames to :meth:`AudioChunkSender.enqueue`
from the agent's event loop; a background task batches them and POSTs raw PCM
with the metadata in ``x-audio-*`` headers. Enqueueing never blocks and never
raises into the agent: a full queue drops the frame and a failing endpoint trips
a circuit breaker for the rest of the call.

Three request shapes reach the endpoint, all defined in ``audio_types``:

**Span chunk** — audio captured while a ``user_speaking``/``agent_speaking`` span
was open. Body is raw PCM; carries ``x-audio-span-id`` and a per-span
``x-audio-seq``, and the final one carries ``x-audio-last`` (plus
``x-audio-heard-ms`` when the utterance was interrupted).

**Noise chunk** — audio captured between speaking spans. Same shape without the
span headers, so it can be laid out on the call timeline but belongs to no turn.

**Session end** — one bodyless request carrying ``x-audio-session-last``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import httpx
from opentelemetry import context as otel_context

from netra.instrumentation.livekit.audio_types import (
    CONTENT_TYPE_PCM,
    DEFAULT_CHANNEL_COUNT,
    DEFAULT_SAMPLE_RATE_HZ,
    HEADER_API_KEY,
    HEADER_BIT_DEPTH,
    HEADER_CHANNELS,
    HEADER_CONTENT_TYPE,
    HEADER_HEARD_MS,
    HEADER_LAST_CHUNK,
    HEADER_ROLE,
    HEADER_SAMPLE_RATE,
    HEADER_SEQUENCE,
    HEADER_SESSION_ID,
    HEADER_SESSION_LAST,
    HEADER_SPAN_ID,
    HEADER_START_MS,
    HEADER_TRACE_ID,
    HEADER_VALUE_TRUE,
    PCM_BIT_DEPTH,
    SpeakerRole,
    pcm_byte_offset_at,
)

if TYPE_CHECKING:
    from livekit.rtc import AudioFrame

logger = logging.getLogger(__name__)

# Defaults for the knobs ``Config`` does not resolve. Every other limit reaches
# the sender from ``Config`` — see ``audio_capture.start_audio_capture``.
DEFAULT_MAX_BATCH_FRAMES = 200
DEFAULT_FLUSH_AT_BYTES = 32768
DEFAULT_MAX_REQUEST_BYTES = 262144

_HTTP_TIMEOUT_SECONDS = 5.0

# Attempts per chunk, total. A chunk POST is safe to repeat: the endpoint keys on
# (session, span, sequence) and the sequence only advances once a chunk has been
# accepted, so a retry re-sends identical bytes under an identical key.
_POST_ATTEMPTS = 2
_RETRY_BASE_DELAY_SECONDS = 0.05

# Consecutive failed chunks after which the rest of the call is abandoned. Audio
# is best-effort: a backend that has been failing this long will not be fixed by
# the next frame, and retrying every 20ms frame for a 10-minute call is worse for
# the agent than sending nothing.
_MAX_CONSECUTIVE_FAILURES = 5

# How long ``end_session`` spends draining, in total, before giving up. It runs
# inline in ``AgentSession._aclose_impl``, so this delays the caller's own session
# teardown — a few seconds of best-effort audio is worth that, half a minute is
# not. A backend too slow to drain inside it has usually tripped the circuit
# already.
_DEFAULT_DRAIN_TIMEOUT_SECONDS = 5.0

_HTTP_STATUS_BAD_REQUEST = 400
_UNAUTHENTICATED_STATUSES = frozenset({401, 403})


# ---------------------------------------------------------------------------
# Queue messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FrameMessage:
    """One captured audio frame awaiting batching."""

    pcm_bytes: bytes
    role: SpeakerRole
    span_id: str
    trace_id: str
    sample_rate_hz: int
    channel_count: int
    timestamp_ns: int


@dataclass(frozen=True)
class _SpanEndMarker:
    """A speaking span closed normally; its recording is complete."""

    role: SpeakerRole
    span_id: str


@dataclass(frozen=True)
class _SpanInterruptMarker:
    """An agent utterance was cut off after *playback_ms* of audible playback."""

    span_id: str
    playback_ms: int


@dataclass(frozen=True)
class _SessionEndMarker:
    """The session is closing; drain everything and stop the loop."""


_QueueMessage = Union[_FrameMessage, _SpanEndMarker, _SpanInterruptMarker, _SessionEndMarker]


# ---------------------------------------------------------------------------
# Sender state
# ---------------------------------------------------------------------------


@dataclass
class _PendingBatch:
    """Frames of one speaker accumulating until a flush condition is met.

    Reused in place across flushes rather than reallocated, so the send loop can
    hold one per :class:`SpeakerRole` in a plain dict with no rebinding.
    """

    role: SpeakerRole
    span_id: str = ""
    trace_id: str = ""
    sample_rate_hz: int = 0
    channel_count: int = 0
    start_ms: int = 0
    frame_count: int = 0
    byte_count: int = 0
    _pcm_parts: List[bytes] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether the batch holds no frames yet."""
        return self.frame_count == 0

    @property
    def pcm_bytes(self) -> bytes:
        """The accumulated frames as one contiguous PCM buffer."""
        return b"".join(self._pcm_parts)

    def frames_within(self, byte_count: int) -> int:
        """Estimate how many accumulated frames fit in the first *byte_count* bytes.

        Used only for the ``frames_sent`` statistic when an interrupt trims the
        batch: pro-rating by the mean frame size is accurate whenever the frames
        are uniformly sized, which is every case livekit-agents produces, and is
        never worse than reporting the untrimmed count.

        Args:
            byte_count: Length of the prefix actually being sent.

        Returns:
            The frame count attributable to that prefix.
        """
        if self.byte_count <= 0:
            return 0
        capped = min(max(byte_count, 0), self.byte_count)
        return round(self.frame_count * capped / self.byte_count)

    def add(self, frame: _FrameMessage) -> None:
        """Append *frame*, adopting its span and format if this is the first one.

        Args:
            frame: The frame to accumulate.
        """
        if self.is_empty:
            self.span_id = frame.span_id
            self.trace_id = frame.trace_id
            self.sample_rate_hz = frame.sample_rate_hz
            self.channel_count = frame.channel_count
            self.start_ms = frame.timestamp_ns // 1_000_000
        self._pcm_parts.append(frame.pcm_bytes)
        self.frame_count += 1
        self.byte_count += len(frame.pcm_bytes)

    def clear(self) -> None:
        """Discard the accumulated frames, keeping the batch's speaker role."""
        self.span_id = ""
        self.trace_id = ""
        self.sample_rate_hz = 0
        self.channel_count = 0
        self.start_ms = 0
        self.frame_count = 0
        self.byte_count = 0
        self._pcm_parts.clear()


@dataclass
class _SpanAudioState:
    """Everything the sender tracks about one speaking span's audio stream.

    One record per span replaces the parallel per-span dictionaries this class
    used to keep, so a span's sequence number, byte position and terminal state
    cannot disagree about which spans exist.

    Attributes:
        role: The speaker the span belongs to.
        trace_id: Hex trace id, so a terminator posted after the batch holding the
            span is gone can still be attributed.
        next_sequence: The number the span's next chunk will carry.
        bytes_consumed: How many PCM bytes of this span have already left the
            pending batch — a *position* in the span's stream, so it counts a
            chunk the sender gave up on as well as an accepted one. Trimming an
            interrupted utterance measures against this; counting bytes actually
            delivered here would make the trim offset drift by whatever was lost.
        is_finalized: Whether the span's terminal chunk has been accepted.
        is_interrupted: Whether the caller cut this utterance short.
        is_end_received: Whether the span-end marker has been processed. Agent
            spans defer their ``is_last`` chunk until either an interrupt marker
            arrives (carrying ``heard_ms``).
    """

    role: SpeakerRole
    trace_id: str = ""
    next_sequence: int = 0
    bytes_consumed: int = 0
    is_finalized: bool = False
    is_interrupted: bool = False
    is_end_received: bool = False


@dataclass
class AudioSenderStats:
    """Delivery counters for one call, stamped onto the ``agent_session`` span.

    The ``sent`` counters record what the endpoint *accepted*: a chunk that
    failed every attempt raises ``errors``, never ``chunks_sent``.

    Attributes:
        chunks_sent: Accepted HTTP requests carrying audio or a terminal marker.
        frames_sent: Captured frames inside those accepted requests.
        bytes_sent: PCM bytes inside those accepted requests.
        frames_dropped: Frames discarded because the queue was full.
        errors: Failed POST attempts, including ones a retry then recovered.
        circuit_tripped: Whether the call gave up on the endpoint entirely.
        total_send_time_ms: Wall-clock spent inside POSTs, for the average below.
    """

    chunks_sent: int = 0
    frames_sent: int = 0
    bytes_sent: int = 0
    frames_dropped: int = 0
    errors: int = 0
    circuit_tripped: bool = False
    total_send_time_ms: float = 0.0

    def __str__(self) -> str:
        """Render the counters as a single log-friendly line."""
        average_ms = self.total_send_time_ms / self.chunks_sent if self.chunks_sent else 0.0
        return (
            f"chunks={self.chunks_sent} frames={self.frames_sent} "
            f"bytes={self.bytes_sent} dropped={self.frames_dropped} "
            f"errors={self.errors} avg_latency={average_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------


class AudioChunkSender:
    """Batches captured frames and POSTs them to the audio-ingest endpoint.

    Single-consumer by construction: :meth:`enqueue` and the marker methods are
    called from the agent's event loop and only hand work to a bounded queue, and
    exactly one background task drains it. Nothing here is safe to call from
    another thread.
    """

    def __init__(
        self,
        *,
        url: str,
        session_id: str,
        api_key: str = "",
        auth_headers: Optional[Dict[str, str]] = None,
        max_batch_frames: int = DEFAULT_MAX_BATCH_FRAMES,
        flush_at_bytes: int = DEFAULT_FLUSH_AT_BYTES,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_queue_frames: int = 0,
    ) -> None:
        """Configure the sender without starting it.

        Args:
            url: Absolute audio-ingest URL, from ``Config.audio_endpoint()``.
            session_id: Identifies the call; sent as ``x-audio-session-id``.
            api_key: Credential sent as ``x-api-key`` when non-empty.
            auth_headers: Further credential headers from the Netra config.
                Applied only where they do not already have a value.
            max_batch_frames: Flush once this many frames have accumulated.
            flush_at_bytes: Target request size — flush once this many PCM bytes
                have accumulated.
            max_request_bytes: Hard ceiling on one request body. A frame that
                would push the batch past it flushes the batch first, so the
                ceiling holds even when it sits just above *flush_at_bytes*.
            max_queue_frames: Bound on frames awaiting batching; further frames
                are dropped rather than queued. 0 means unbounded.
        """
        self._url = url.rstrip("/")
        self._session_id = session_id
        self._api_key = api_key
        self._auth_headers = auth_headers or {}
        self._max_batch_frames = max_batch_frames
        self._flush_at_bytes = flush_at_bytes
        self._max_request_bytes = max(flush_at_bytes, max_request_bytes)

        self._queue: asyncio.Queue[_QueueMessage] = asyncio.Queue(maxsize=max(0, max_queue_frames))
        self._span_states: Dict[str, _SpanAudioState] = {}
        self._send_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._is_closed = False

        self._consecutive_failures = 0
        self._circuit_tripped = False
        self._has_warned_about_drops = False

        self.stats = AudioSenderStats()

    # -- lifecycle ----------------------------------------------------------

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The event loop this sender's queue and task belong to, once started.

        Everything here is bound to that loop, so a shutdown path reaching the
        sender from elsewhere has to drive it through this rather than awaiting
        it directly. ``None`` before :meth:`start`.
        """
        return self._loop

    async def start(self) -> None:
        """Open the HTTP client and start the background send loop."""
        self._loop = asyncio.get_running_loop()
        self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)
        self._send_task = asyncio.create_task(self._run_send_loop(), name="netra-audio-chunk-sender")
        logger.info(
            "netra.audio: sender started -> %s (max_frames=%d, flush_at=%dB, max_request=%dB)",
            self._url,
            self._max_batch_frames,
            self._flush_at_bytes,
            self._max_request_bytes,
        )

    async def end_session(self, *, drain_timeout_seconds: float | None = None) -> None:
        """Drain the queue, close every open span, and signal the session's end.

        Idempotent: a second call returns immediately. Once this has been called
        no further frames are accepted, so a late frame from a task that has not
        noticed the shutdown is dropped rather than queued behind the terminal
        marker it would never get past.

        Args:
            drain_timeout_seconds: Total budget for the whole teardown. The two
                waits inside share one deadline rather than each taking the full
                timeout, because a caller that allowed *n* seconds for the session
                to close means *n* seconds, not 2*n*.
        """
        if self._is_closed:
            return
        self._is_closed = True

        deadline = time.monotonic() + max(
            0.0, drain_timeout_seconds if drain_timeout_seconds is not None else _DEFAULT_DRAIN_TIMEOUT_SECONDS
        )
        await self._enqueue_session_end(deadline)
        if self._send_task is not None:
            await self._await_send_task(deadline)
        if self._client is not None:
            await self._client.aclose()
        logger.info("netra.audio: sender closed — %s", self.stats)

    async def _enqueue_session_end(self, deadline: float) -> None:
        """Get the terminal marker onto the queue, waiting for room if need be.

        ``put_nowait`` is wrong here: on a bounded queue that is currently full
        the marker would be dropped and the send loop would never learn to stop,
        so the drain below would spend its whole timeout before cancelling. No
        producer can refill the queue at this point — ``_is_closed`` is already
        set — so waiting for the consumer to make room terminates.

        Args:
            deadline: ``time.monotonic()`` value the whole teardown must finish by.
        """
        try:
            await asyncio.wait_for(self._queue.put(_SessionEndMarker()), timeout=_seconds_until(deadline))
        except asyncio.TimeoutError:
            logger.warning("netra.audio: could not signal session end before the teardown deadline")

    async def _await_send_task(self, deadline: float) -> None:
        """Wait for the send loop to drain, cancelling it if it overruns.

        The cancellation is awaited rather than merely requested: ``end_session``
        closes the HTTP client next, and a send loop still inside a POST would
        otherwise find the client shut from under it.

        Args:
            deadline: ``time.monotonic()`` value the whole teardown must finish by.
        """
        task = self._send_task
        if task is None:
            return
        try:
            await asyncio.wait_for(task, timeout=_seconds_until(deadline))
        except asyncio.TimeoutError:
            logger.warning("netra.audio: send loop did not drain before the teardown deadline; cancelling")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("netra.audio: send loop ended with an error", exc_info=True)

    # -- producer side (agent event loop) -----------------------------------

    def enqueue(
        self,
        frame: "AudioFrame",
        *,
        role: SpeakerRole,
        trace_id: str,
        span_id: str = "",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Queue one captured frame. Never blocks, never raises into the agent.

        Copies the PCM out of the frame via the public ``frame.data``
        memoryview: LiveKit reuses the underlying buffer for the next frame, so
        holding a reference would corrupt the batch.

        Args:
            frame: The LiveKit frame just captured.
            role: Which speaker produced it.
            trace_id: Hex trace id to attribute the audio to.
            span_id: Hex id of the open speaking span, or ``""`` for audio
                captured between turns.
            timestamp_ns: Capture time, defaulting to now. Passed in by the
                coordinator so the timestamp is taken at capture rather than
                after any queuing delay.
        """
        if self._is_closed or self._circuit_tripped:
            return
        try:
            message = _FrameMessage(
                pcm_bytes=bytes(frame.data),
                role=role,
                span_id=span_id,
                trace_id=trace_id,
                sample_rate_hz=frame.sample_rate,
                channel_count=frame.num_channels,
                timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns(),
            )
        except (AttributeError, TypeError, ValueError):
            # A frame shaped differently from what livekit-agents documents. Not
            # recoverable and not the agent's problem — drop this one frame.
            logger.debug("netra.audio: unreadable audio frame dropped", exc_info=True)
            self.stats.frames_dropped += 1
            return

        if not self._offer(message):
            self.stats.frames_dropped += 1
            self._warn_about_drops_once()

    def mark_audio_end(self, *, role: SpeakerRole, span_id: str) -> None:
        """Signal that the recording for *span_id* is complete.

        Args:
            role: The speaker whose span closed.
            span_id: Hex id of the closed speaking span.
        """
        if self._is_closed or not span_id:
            return
        state = self._span_states.get(span_id)
        if state is not None and state.is_finalized:
            return
        if not self._offer(_SpanEndMarker(role=role, span_id=span_id)):
            logger.debug("netra.audio: queue full; end marker for span=%s dropped", span_id)

    def interrupt_agent_span(self, *, span_id: str, playback_ms: int) -> None:
        """Signal that an agent utterance was cut off *playback_ms* into playback.

        The send loop trims the pending audio for the span to what was heard and
        finalizes it with a single ``is_last`` chunk carrying ``heard_ms``.
        Agent span-end markers defer their terminator specifically so this
        interrupt can be the sole ``is_last`` for the span, avoiding duplicate
        terminators that would cause the backend to process prematurely.

        Args:
            span_id: Hex id of the interrupted ``agent_speaking`` span.
            playback_ms: Milliseconds of the utterance the caller heard.
        """
        if self._is_closed or not span_id:
            return
        if not self._offer(_SpanInterruptMarker(span_id=span_id, playback_ms=playback_ms)):
            logger.debug("netra.audio: queue full; interrupt marker for span=%s dropped", span_id)

    def _offer(self, message: _QueueMessage) -> bool:
        """Hand *message* to the send loop without ever blocking the caller.

        ``asyncio.Queue`` is not thread-safe, and the marker methods are reachable
        from :class:`AudioSpanProcessor`, which OTel invokes on whichever thread
        ends the span — normally the agent's loop thread, but nothing enforces
        that. An off-loop caller is therefore bounced onto the sender's own loop
        instead of corrupting the queue.

        Args:
            message: The message to enqueue.

        Returns:
            True when it was queued or handed to the loop, False when the queue is
            at its bound. The caller decides how a drop is accounted for — a
            dropped frame is a statistic, a dropped marker is not.
        """
        loop = self._loop
        if loop is not None and loop is not _running_loop():
            # Whether the queue had room is not knowable from here; the hop itself
            # succeeding is all this can report.
            loop.call_soon_threadsafe(self._offer_on_loop, message)
            return True
        return self._put_nowait(message)

    def _offer_on_loop(self, message: _QueueMessage) -> None:
        """Enqueue a message that arrived from another thread. Runs on the loop.

        Args:
            message: The message to enqueue.
        """
        if not self._put_nowait(message):
            logger.debug("netra.audio: queue full; cross-thread %s dropped", type(message).__name__)

    def _put_nowait(self, message: _QueueMessage) -> bool:
        """Put *message* on the queue if it has room.

        Args:
            message: The message to enqueue.

        Returns:
            True when it was queued, False when the queue is at its bound.
        """
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            return False
        return True

    def _warn_about_drops_once(self) -> None:
        """Warn that frames are being dropped, at most once per session."""
        if self._has_warned_about_drops:
            return
        self._has_warned_about_drops = True
        logger.warning(
            "netra.audio: queue full, dropping frames (session=%s). This is logged once per session",
            self._session_id,
        )

    # -- consumer side (background task) ------------------------------------

    async def _run_send_loop(self) -> None:
        """Drain the queue until the session ends, with instrumentation muted.

        The loop's own HTTP calls run under ``_SUPPRESS_INSTRUMENTATION_KEY`` so
        Netra's httpx instrumentation does not trace them: every audio chunk
        would otherwise produce a span, inside the very trace the audio belongs
        to.
        """
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

        token = otel_context.attach(otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            await self._consume_queue()
        finally:
            otel_context.detach(token)

    async def _consume_queue(self) -> None:
        """Batch queued frames and post them until the session-end marker."""
        batches = {role: _PendingBatch(role=role) for role in SpeakerRole}

        while True:
            message = await self._queue.get()

            if isinstance(message, _SessionEndMarker):
                await self._drain_batches(batches)
                return

            await self._handle_message(message, batches)

    async def _handle_message(self, message: _QueueMessage, batches: Dict[SpeakerRole, _PendingBatch]) -> None:
        """Dispatch one queued message to its handler.

        Args:
            message: The message the loop dequeued.
            batches: The pending batch for each speaker.
        """
        if isinstance(message, _FrameMessage):
            await self._handle_frame(message, batches[message.role])
        elif isinstance(message, _SpanEndMarker):
            await self._handle_span_end(message, batches[message.role])
        elif isinstance(message, _SpanInterruptMarker):
            await self._handle_span_interrupt(message, batches[SpeakerRole.AGENT])

    async def _handle_frame(self, frame: _FrameMessage, batch: _PendingBatch) -> None:
        """Accumulate one frame, flushing first or after if a boundary is hit.

        Args:
            frame: The frame to accumulate.
            batch: The pending batch for that frame's speaker.
        """
        state = self._span_states.get(frame.span_id) if frame.span_id else None
        if state is not None and state.is_interrupted:
            # Queued before the interrupt was observed but captured after the
            # caller cut in — this audio was never heard.
            return

        # A batch holds one span's audio: the chunk's span id is a single header.
        # It also has to stay under the request ceiling, so a frame that would
        # burst it closes the batch instead of joining it.
        spans_differ = not batch.is_empty and batch.span_id != frame.span_id
        would_overflow = batch.byte_count + len(frame.pcm_bytes) > self._max_request_bytes
        if spans_differ or would_overflow:
            await self._flush(batch)

        batch.add(frame)

        if batch.frame_count >= self._max_batch_frames or batch.byte_count >= self._flush_at_bytes:
            await self._flush(batch)

    async def _handle_span_end(self, marker: _SpanEndMarker, batch: _PendingBatch) -> None:
        """Finalize a speaking span, flushing whatever audio is still pending.

        Agent spans defer their terminator: LiveKit routinely ends the
        ``agent_speaking`` span before it reports an interrupt, and sending
        ``is_last`` at span-end would cause the backend to start processing
        the full audio before the interrupt's ``heard_ms`` arrives. Deferring
        lets the interrupt marker be the single ``is_last`` for interrupted
        spans; uninterrupted ones are finalized at session drain.

        Args:
            marker: The end marker for the span.
            batch: The pending batch for that span's speaker.
        """
        state = self._span_states.get(marker.span_id)
        if state is not None and state.is_finalized:
            return

        if marker.role is SpeakerRole.AGENT:
            if batch.span_id == marker.span_id and not batch.is_empty:
                await self._flush(batch)
            self._state_for(marker.span_id, marker.role).is_end_received = True
            return

        if batch.span_id == marker.span_id and not batch.is_empty:
            await self._flush(batch, is_final=True)
            return

        await self._post_span_terminator(role=marker.role, span_id=marker.span_id)

    async def _handle_span_interrupt(self, marker: _SpanInterruptMarker, batch: _PendingBatch) -> None:
        """Trim an interrupted agent span to the audio heard, then finalize it.

        Agent span-end markers defer their ``is_last``, so this handler is
        normally the one that sends the single terminator for the span — with
        ``heard_ms`` attached. If the deferred finalization happened to run
        first (edge case: the interrupt arrived more than one batch interval
        after the span end), the span is already closed and nothing is sent.

        Args:
            marker: The interrupt marker, carrying the playback position.
            batch: The pending agent batch.
        """
        state = self._state_for(marker.span_id, SpeakerRole.AGENT)
        state.is_interrupted = True
        if state.is_finalized:
            return

        if batch.span_id != marker.span_id or batch.is_empty:
            await self._post_span_terminator(
                role=state.role,
                span_id=marker.span_id,
                heard_ms=marker.playback_ms,
            )
            return

        await self._flush_heard_prefix(batch, marker.playback_ms)

    async def _flush_heard_prefix(self, batch: _PendingBatch, playback_ms: int) -> None:
        """Post only the part of *batch* the caller heard, marked final.

        The heard prefix is measured from the start of the *span*, so whatever
        earlier chunks already consumed of it has to come off the offset before
        the pending batch can be trimmed.

        Args:
            batch: The pending agent batch, known to hold audio for the span.
            playback_ms: Milliseconds of the utterance the caller heard.
        """
        # Read the batch's identity out before any flush: ``_PendingBatch.clear``
        # resets ``span_id``, so a terminator addressed from a cleared batch would
        # carry ``""`` and be silently dropped by ``_post_span_terminator``.
        span_id = batch.span_id
        role = batch.role
        heard_offset = pcm_byte_offset_at(
            playback_ms=playback_ms,
            sample_rate_hz=batch.sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ,
            channel_count=batch.channel_count or DEFAULT_CHANNEL_COUNT,
        )
        already_consumed = self._state_for(span_id, role).bytes_consumed
        remaining = heard_offset - already_consumed

        if remaining <= 0:
            batch.clear()
            await self._post_span_terminator(
                role=role,
                span_id=span_id,
                heard_ms=playback_ms,
            )
            return

        heard_pcm = batch.pcm_bytes[:remaining]
        logger.debug(
            "netra.audio: trimmed interrupted span=%s to %d of %d pending bytes (heard=%dms, consumed=%d)",
            span_id,
            len(heard_pcm),
            batch.byte_count,
            playback_ms,
            already_consumed,
        )
        frame_count = batch.frames_within(len(heard_pcm))
        start_ms = batch.start_ms
        sample_rate_hz = batch.sample_rate_hz
        channel_count = batch.channel_count
        trace_id = batch.trace_id
        batch.clear()
        await self._post_chunk(
            role=role,
            span_id=span_id,
            trace_id=trace_id,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
            pcm=heard_pcm,
            frame_count=frame_count,
            start_ms=start_ms,
            is_last=True,
            heard_ms=playback_ms,
        )

    async def _drain_batches(self, batches: Dict[SpeakerRole, _PendingBatch]) -> None:
        """Send everything still held, then close the session on the wire.

        Args:
            batches: The pending batch for each speaker.
        """
        for batch in batches.values():
            await self._flush(batch, is_final=bool(batch.span_id))
        await self._finalize_deferred_agent_spans()
        await self._finalize_open_spans()
        await self._post_session_terminator()

    async def _finalize_deferred_agent_spans(self) -> None:
        """Send the deferred terminator for agent spans that were not interrupted.

        Agent spans defer their ``is_last`` chunk so that a closely-following
        interrupt marker can be the single terminator carrying ``heard_ms``.
        Spans still open at drain are treated as uninterrupted and finalized here.
        """
        for span_id, state in list(self._span_states.items()):
            if (
                state.role is SpeakerRole.AGENT
                and state.is_end_received
                and not state.is_finalized
                and not state.is_interrupted
            ):
                await self._post_span_terminator(role=state.role, span_id=span_id)

    async def _flush(self, batch: _PendingBatch, *, is_final: bool = False) -> None:
        """Post *batch*'s audio and clear it.

        A final flush is two requests, not one: the audio chunk, then an empty
        chunk carrying ``x-audio-last``. Keeping the terminator separate means
        the span closes the same way whether or not audio happened to be pending
        when it ended.

        Args:
            batch: The batch to send.
            is_final: Whether this closes the batch's span.
        """
        if batch.is_empty and not is_final:
            return

        span_id = batch.span_id
        role = batch.role
        trace_id = batch.trace_id

        if not batch.is_empty:
            await self._post_chunk(
                role=role,
                span_id=span_id,
                trace_id=trace_id,
                sample_rate_hz=batch.sample_rate_hz,
                channel_count=batch.channel_count,
                pcm=batch.pcm_bytes,
                frame_count=batch.frame_count,
                start_ms=batch.start_ms,
                is_last=False,
            )
        batch.clear()

        if is_final and span_id:
            await self._post_span_terminator(role=role, span_id=span_id)

    async def _finalize_open_spans(self) -> None:
        """Close any span that never received an end marker.

        A span left open would leave the endpoint waiting for audio that is
        never coming, so this is a backstop rather than a normal path — hence
        the warning.

        Skipped entirely once the circuit has tripped: every span is open in that
        case, by definition, and ``_trip_circuit`` has already said why once.
        Warning per span would bury it under hundreds of lines.
        """
        if self._circuit_tripped:
            return

        open_span_ids = sorted(span_id for span_id, state in self._span_states.items() if not state.is_finalized)
        for span_id in open_span_ids:
            state = self._span_states[span_id]
            logger.warning(
                "netra.audio: finalizing span left open at session end: span_id=%s role=%s",
                span_id,
                state.role.value,
            )
            await self._post_span_terminator(role=state.role, span_id=span_id)

    # -- requests -----------------------------------------------------------

    async def _post_span_terminator(
        self,
        *,
        role: SpeakerRole,
        span_id: str,
        heard_ms: int = 0,
    ) -> None:
        """Post the empty chunk that closes a span.

        Args:
            role: The speaker the span belongs to.
            span_id: Hex id of the span to close.
            heard_ms: Milliseconds heard, for an interrupted agent span only.
        """
        if not span_id:
            return
        state = self._span_states.get(span_id)
        if state is not None and state.is_finalized:
            return

        await self._post_chunk(
            role=role,
            span_id=span_id,
            trace_id=state.trace_id if state is not None else "",
            sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
            channel_count=DEFAULT_CHANNEL_COUNT,
            pcm=b"",
            frame_count=0,
            start_ms=0,
            is_last=True,
            heard_ms=heard_ms,
        )

    async def _post_session_terminator(self) -> None:
        """Post the bodyless request that marks the whole session complete.

        Skipped once the circuit has tripped: "no further audio will be sent for
        this session" has to include this request, or a session abandoned over a
        rejected credential would still end with one more rejected POST.
        """
        if self._circuit_tripped:
            return

        headers = {
            HEADER_SESSION_ID: self._session_id,
            HEADER_SESSION_LAST: HEADER_VALUE_TRUE,
        }
        self._apply_credentials(headers)
        await self._post(b"", headers)

    async def _post_chunk(
        self,
        *,
        role: SpeakerRole,
        span_id: str,
        trace_id: str,
        sample_rate_hz: int,
        channel_count: int,
        pcm: bytes,
        frame_count: int,
        start_ms: int,
        is_last: bool,
        heard_ms: int = 0,
    ) -> None:
        """Send one chunk and record what it did to the span's state.

        Args:
            role: The speaker the audio came from.
            span_id: Hex id of the speaking span, or ``""`` for between-turn audio.
            trace_id: Hex trace id the audio belongs to.
            sample_rate_hz: Samples per second, per channel.
            channel_count: Interleaved channel count.
            pcm: The body — signed 16-bit little-endian PCM.
            frame_count: How many captured frames the body holds, for the stats.
            start_ms: Epoch milliseconds of the body's first frame.
            is_last: Whether this closes the span.
            heard_ms: Milliseconds heard, for an interrupted agent span only.
        """
        if self._circuit_tripped:
            return

        state = self._state_for(span_id, role, trace_id) if span_id else None
        headers = self._chunk_headers(
            role=role,
            span_id=span_id,
            trace_id=trace_id,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
            start_ms=start_ms,
            is_last=is_last,
            heard_ms=heard_ms,
            state=state,
        )

        accepted = await self._post(pcm, headers)

        logger.debug(
            "netra.audio: chunk span_id=%s role=%s frames=%d bytes=%d last=%s accepted=%s",
            span_id or "(between turns)",
            role.value,
            frame_count,
            len(pcm),
            is_last,
            accepted,
        )

        if state is not None:
            # Advanced whether or not the chunk landed. Both are positions in the
            # span's stream, not delivery counts: a chunk the sender gave up on
            # still occupied its slot, so reusing its number for the *next*,
            # different audio would break the idempotency key the endpoint dedupes
            # on. A gap is how the endpoint learns audio was lost.
            state.next_sequence += 1
            state.bytes_consumed += len(pcm)
            if accepted and is_last:
                state.is_finalized = True

        if not accepted:
            return

        self.stats.chunks_sent += 1
        self.stats.frames_sent += frame_count
        self.stats.bytes_sent += len(pcm)

    def _chunk_headers(
        self,
        *,
        role: SpeakerRole,
        span_id: str,
        trace_id: str,
        sample_rate_hz: int,
        channel_count: int,
        start_ms: int,
        is_last: bool,
        heard_ms: int,
        state: Optional[_SpanAudioState],
    ) -> Dict[str, str]:
        """Build the ``x-audio-*`` headers describing one chunk.

        Args:
            role: The speaker the audio came from.
            span_id: Hex span id, or ``""`` for between-turn audio.
            trace_id: Hex trace id the audio belongs to.
            sample_rate_hz: Samples per second, per channel.
            channel_count: Interleaved channel count.
            start_ms: Epoch milliseconds of the first frame.
            is_last: Whether this closes the span.
            heard_ms: Milliseconds heard, for an interrupted agent span only.
            state: The span's state, or ``None`` for between-turn audio.

        Returns:
            The complete header set for the request.
        """
        headers = {
            HEADER_CONTENT_TYPE: CONTENT_TYPE_PCM,
            HEADER_SESSION_ID: self._session_id,
            HEADER_TRACE_ID: trace_id,
            HEADER_ROLE: role.value,
            HEADER_START_MS: str(start_ms),
            HEADER_SAMPLE_RATE: str(sample_rate_hz or DEFAULT_SAMPLE_RATE_HZ),
            HEADER_CHANNELS: str(channel_count or DEFAULT_CHANNEL_COUNT),
            HEADER_BIT_DEPTH: str(PCM_BIT_DEPTH),
        }
        self._apply_credentials(headers)

        if state is not None:
            headers[HEADER_SPAN_ID] = span_id
            headers[HEADER_SEQUENCE] = str(state.next_sequence)
            if is_last:
                headers[HEADER_LAST_CHUNK] = HEADER_VALUE_TRUE
                if heard_ms > 0:
                    headers[HEADER_HEARD_MS] = str(heard_ms)
        return headers

    def _apply_credentials(self, headers: Dict[str, str]) -> None:
        """Add the configured credential headers, without overwriting any.

        Args:
            headers: The header set being built, mutated in place.
        """
        if self._api_key:
            headers[HEADER_API_KEY] = self._api_key
        for name, value in self._auth_headers.items():
            headers.setdefault(name, value)

    async def _post(self, pcm: bytes, headers: Dict[str, str]) -> bool:
        """POST one request, retrying a transient failure.

        Args:
            pcm: The request body.
            headers: The request headers.

        Returns:
            True when the endpoint accepted the request.
        """
        client = self._client
        if client is None:
            logger.debug("netra.audio: post attempted before start(); dropping chunk")
            return False

        for attempt in range(_POST_ATTEMPTS):
            accepted, is_fatal = await self._post_once(client, pcm, headers, attempt)
            if accepted or is_fatal:
                return accepted
            if attempt < _POST_ATTEMPTS - 1:
                await asyncio.sleep(_retry_delay_seconds(attempt))

        logger.warning("netra.audio: giving up on a chunk after %d attempts", _POST_ATTEMPTS)
        return False

    async def _post_once(
        self,
        client: httpx.AsyncClient,
        pcm: bytes,
        headers: Dict[str, str],
        attempt: int,
    ) -> tuple[bool, bool]:
        """Make one POST attempt and account for its outcome.

        Args:
            client: The open HTTP client.
            pcm: The request body.
            headers: The request headers.
            attempt: 0-based attempt number, for the log line.

        Returns:
            ``(accepted, is_fatal)`` — ``is_fatal`` means retrying cannot help,
            either because the credential was rejected or because the circuit
            breaker has now tripped.
        """
        started_at = time.monotonic()
        try:
            response = await client.post(self._url, content=pcm, headers=headers)
        except httpx.HTTPError as exc:
            self.stats.total_send_time_ms += (time.monotonic() - started_at) * 1000
            self.stats.errors += 1
            logger.warning("netra.audio: chunk POST error (attempt=%d): %s", attempt + 1, exc)
            return False, self._record_failure()

        self.stats.total_send_time_ms += (time.monotonic() - started_at) * 1000

        if response.status_code < _HTTP_STATUS_BAD_REQUEST:
            self._consecutive_failures = 0
            return True, False

        self.stats.errors += 1
        if response.status_code in _UNAUTHENTICATED_STATUSES:
            self._trip_circuit(f"HTTP {response.status_code} — a credential will not become valid mid-call")
            return False, True

        logger.warning(
            "netra.audio: chunk POST rejected (attempt=%d): %d %s",
            attempt + 1,
            response.status_code,
            response.text[:200],
        )
        return False, self._record_failure()

    # -- failure handling ---------------------------------------------------

    def _record_failure(self) -> bool:
        """Count one failure and trip the circuit if the run is long enough.

        Returns:
            True when the circuit is now open, meaning retrying is pointless.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self._trip_circuit(f"{self._consecutive_failures} consecutive failures")
        return self._circuit_tripped

    def _trip_circuit(self, reason: str) -> None:
        """Abandon audio for the rest of the call.

        Args:
            reason: What went wrong, for the operator-facing log line.
        """
        if self._circuit_tripped:
            return
        self._circuit_tripped = True
        self.stats.circuit_tripped = True
        logger.warning(
            "netra.audio: circuit breaker tripped (session=%s): %s. "
            "No further audio will be sent for this session; traces are unaffected",
            self._session_id,
            reason,
        )

    # -- span state ---------------------------------------------------------

    def _state_for(self, span_id: str, role: SpeakerRole, trace_id: str = "") -> _SpanAudioState:
        """Return the state record for *span_id*, creating it on first sight.

        Args:
            span_id: Hex id of a speaking span.
            role: The speaker it belongs to.
            trace_id: Hex trace id, remembered so a later terminator for this
                span can still be attributed once the batch holding it is gone.

        Returns:
            The span's mutable state record.
        """
        state = self._span_states.get(span_id)
        if state is None:
            state = _SpanAudioState(role=role, trace_id=trace_id)
            self._span_states[span_id] = state
        elif trace_id and not state.trace_id:
            state.trace_id = trace_id
        return state


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the loop running on this thread, or ``None`` on a plain thread.

    Returns:
        The current event loop, if there is one.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _seconds_until(deadline: float) -> float:
    """Return the time left before *deadline*, never negative.

    Args:
        deadline: A ``time.monotonic()`` value.

    Returns:
        Seconds remaining. 0.0 once the deadline has passed, which makes the
        ``wait_for`` it is handed to give up immediately rather than restart the
        full budget.
    """
    return max(0.0, deadline - time.monotonic())


def _retry_delay_seconds(attempt: int) -> float:
    """Return the backoff before retrying, exponential with full jitter.

    Args:
        attempt: 0-based number of the attempt that just failed.

    Returns:
        Seconds to wait. Jittered so that a backend recovering from an outage is
        not hit by every concurrent call's sender at the same instant.
    """
    ceiling = _RETRY_BASE_DELAY_SECONDS * (2**attempt)
    return random.uniform(0.0, ceiling)
