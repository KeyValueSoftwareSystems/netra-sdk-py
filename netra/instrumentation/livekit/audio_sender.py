"""Async HTTP sender — streams audio chunks to a backend in real time.

Frames are enqueued from ``AudioHookManager``, batched internally, and
POSTed with raw PCM in the body and metadata in ``x-audio-*`` HTTP headers.

Three request shapes:

**Span chunk** (speech) — body = raw PCM::

    x-audio-session-id, x-audio-trace-id, x-audio-span-id
    x-audio-role: user|agent
    x-audio-start-ms   (epoch ms)
    x-audio-seq        (0-based, per-span)
    x-audio-last: true (final chunk only)
    x-audio-sample-rate / channels / bit-depth

**Noise chunk** (unspanned audio) — same but ``x-audio-span-id``,
``x-audio-seq``, ``x-audio-last`` are omitted.

**Session end** — bodyless::

    x-audio-session-id, x-audio-session-last: true
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from opentelemetry import context as otel_context

if TYPE_CHECKING:
    from livekit.rtc import AudioFrame

logger = logging.getLogger(__name__)

_BATCH_INTERVAL_S = 0.5
_MAX_BATCH_FRAMES = 200
_HTTP_TIMEOUT_S = 5.0
_POST_RETRIES = 2
_MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class _QueueItem:
    """Single frame waiting to be batched."""

    pcm_bytes: bytes
    kind: str
    span_id: str
    trace_id: str
    sample_rate: int
    num_channels: int
    timestamp_ns: int


@dataclass
class _AudioEndMarker:
    """Signals that a span recording is complete."""

    kind: str
    span_id: str


@dataclass
class _PendingBatch:
    """Accumulator for frames of the same kind + span_id until flush."""

    kind: str
    span_id: str = ""
    trace_id: str = ""
    sample_rate: int = 0
    num_channels: int = 0
    start_ms: int = 0
    pcm_parts: list[bytes] = field(default_factory=list)
    frame_count: int = 0

    def add(self, item: _QueueItem) -> None:
        if not self.pcm_parts:
            self.span_id = item.span_id
            self.trace_id = item.trace_id
            self.sample_rate = item.sample_rate
            self.num_channels = item.num_channels
            self.start_ms = item.timestamp_ns // 1_000_000
        self.pcm_parts.append(item.pcm_bytes)
        self.frame_count += 1

    @property
    def pcm_bytes(self) -> bytes:
        return b"".join(self.pcm_parts)


_QueueItemType = _QueueItem | _AudioEndMarker


@dataclass
class AudioSenderStats:
    chunks_sent: int = 0
    frames_sent: int = 0
    bytes_sent: int = 0
    frames_dropped: int = 0
    errors: int = 0
    circuit_tripped: bool = False
    total_send_time_ms: float = 0.0

    def __str__(self) -> str:
        avg = self.total_send_time_ms / self.chunks_sent if self.chunks_sent else 0.0
        return (
            f"chunks={self.chunks_sent} frames={self.frames_sent} "
            f"bytes={self.bytes_sent} dropped={self.frames_dropped} "
            f"errors={self.errors} avg_latency={avg:.1f}ms"
        )


class AudioChunkSender:
    """Streams audio frames to a backend endpoint via HTTP POST.

    Args:
        url: Full audio ingest URL (e.g.
            ``http://localhost:3000/telemetry/v1/audio/chunk``).
            Comes from ``Config.audio_endpoint()``.
        session_id: Session-level identifier — sent as ``x-audio-session-id``.
        api_key: API key for authentication.
        auth_headers: Additional auth headers from the Netra config.
        batch_interval: Seconds between automatic flushes.
        max_batch_frames: Flush when this many frames accumulate.
        max_queue_size: Bounded queue size; frames are dropped when full.
    """

    def __init__(
        self,
        *,
        url: str,
        session_id: str,
        api_key: str = "",
        auth_headers: Optional[Dict[str, str]] = None,
        batch_interval: float = _BATCH_INTERVAL_S,
        max_batch_frames: int = _MAX_BATCH_FRAMES,
        max_queue_size: int = 0,
    ) -> None:
        self._url = url.rstrip("/")
        self._session_id = session_id
        self._api_key = api_key
        self._auth_headers = auth_headers or {}
        self._batch_interval = batch_interval
        self._max_batch_frames = max_batch_frames

        queue_bound = max_queue_size if max_queue_size > 0 else 0
        self._queue: asyncio.Queue[_QueueItemType | None] = asyncio.Queue(maxsize=queue_bound)
        self._seq: Dict[str, int] = {}
        self._sent_span_ids: Set[str] = set()
        self._finalized_span_ids: Set[str] = set()
        self._span_trace_ids: Dict[str, str] = {}
        self._span_roles: Dict[str, str] = {}
        self._known_span_ids: Set[str] = set()
        self._task: Optional[asyncio.Task[None]] = None
        self._client: Any = None
        self._closed = False

        self._consecutive_failures = 0
        self._circuit_tripped = False
        self._drop_warned = False

        self.stats = AudioSenderStats()

    async def start(self) -> None:
        """Start the background send loop."""
        import httpx

        self._client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
        self._task = asyncio.create_task(self._send_loop(), name="netra-audio-chunk-sender")
        logger.info(
            "netra.audio: sender started -> %s (batch=%.1fs, max_frames=%d)",
            self._url,
            self._batch_interval,
            self._max_batch_frames,
        )

    def enqueue(
        self,
        frame: "AudioFrame",
        *,
        kind: str,
        trace_id: str,
        span_id: str = "",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Enqueue a single audio frame for streaming.

        Uses ``bytes(frame.data)`` — the public memoryview API — to copy
        frame data.  LiveKit reuses the underlying buffer so the copy is
        mandatory.

        Drops the frame (never blocks) when the queue is full or the
        circuit breaker has tripped.
        """
        if self._closed or self._circuit_tripped:
            return
        try:
            pcm = bytes(frame.data)
            self._queue.put_nowait(
                _QueueItem(
                    pcm_bytes=pcm,
                    kind=kind,
                    span_id=span_id,
                    trace_id=trace_id,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    timestamp_ns=timestamp_ns if timestamp_ns is not None else time.time_ns(),
                )
            )
        except asyncio.QueueFull:
            self.stats.frames_dropped += 1
            if not self._drop_warned:
                self._drop_warned = True
                logger.warning(
                    "netra.audio: queue full, dropping frames (session=%s). " "This is logged once per session",
                    self._session_id,
                )
        except Exception as exc:
            logger.debug("netra.audio: failed to enqueue audio frame: %s", exc)

    def mark_audio_end(self, kind: str, span_id: str) -> None:
        """Signal that the span recording for *span_id* is complete."""
        if self._closed or not span_id:
            return
        if span_id in self._finalized_span_ids:
            return
        try:
            self._queue.put_nowait(_AudioEndMarker(kind=kind, span_id=span_id))
        except asyncio.QueueFull:
            logger.debug("netra.audio: could not enqueue audio-end marker for span=%s", span_id)

    async def end_session(self) -> None:
        """Flush remaining frames, finalize open spans, signal session end."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("netra.audio: send loop timed out on close")
                self._task.cancel()
        if self._client:
            await self._client.aclose()
        logger.info("netra.audio: sender closed — %s", self.stats)

    # -- background send loop -----------------------------------------------

    async def _send_loop(self) -> None:
        from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

        token = otel_context.attach(otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            await self._send_loop_inner()
        finally:
            otel_context.detach(token)

    async def _send_loop_inner(self) -> None:
        user_batch = _PendingBatch(kind="user")
        agent_batch = _PendingBatch(kind="agent")

        def _batch_for(kind: str) -> _PendingBatch:
            return user_batch if kind == "user" else agent_batch

        def _reset_batch(kind: str) -> None:
            nonlocal user_batch, agent_batch
            if kind == "user":
                user_batch = _PendingBatch(kind="user")
            else:
                agent_batch = _PendingBatch(kind="agent")

        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._batch_interval)
            except asyncio.TimeoutError:
                await self._flush_batch(user_batch)
                await self._flush_batch(agent_batch)
                _reset_batch("user")
                _reset_batch("agent")
                continue

            if item is None:
                for kind in ("user", "agent"):
                    batch = _batch_for(kind)
                    is_last = bool(batch.span_id)
                    await self._flush_batch(batch, is_last=is_last)
                    _reset_batch(kind)
                await self._finalize_open_spans()
                await self._post_session_last()
                break

            if isinstance(item, _AudioEndMarker):
                if item.span_id in self._finalized_span_ids:
                    continue
                batch = _batch_for(item.kind)
                if batch.span_id == item.span_id and batch.pcm_parts:
                    await self._flush_batch(batch, is_last=True)
                    _reset_batch(item.kind)
                else:
                    if batch.span_id == item.span_id:
                        _reset_batch(item.kind)
                    await self._send_span_last(
                        kind=item.kind,
                        span_id=item.span_id,
                        trace_id=self._span_trace_ids.get(item.span_id, ""),
                    )
                continue

            batch = _batch_for(item.kind)

            if batch.pcm_parts and batch.span_id != item.span_id:
                await self._flush_batch(batch)
                _reset_batch(item.kind)
                batch = _batch_for(item.kind)

            batch.add(item)

            if batch.frame_count >= self._max_batch_frames:
                await self._flush_batch(batch)
                _reset_batch(item.kind)

    async def _finalize_open_spans(self) -> None:
        """Send ``x-audio-last`` for every span that was never finalized."""
        open_spans = (self._known_span_ids | self._sent_span_ids) - self._finalized_span_ids
        for span_id in sorted(open_spans):
            kind = self._span_roles.get(span_id, "user")
            logger.warning(
                "netra.audio: finalizing open span without mark_audio_end: span_id=%s role=%s",
                span_id,
                kind,
            )
            await self._send_span_last(
                kind=kind,
                span_id=span_id,
                trace_id=self._span_trace_ids.get(span_id, ""),
            )

    async def _send_span_last(self, *, kind: str, span_id: str, trace_id: str) -> None:
        """Send an empty terminal chunk with ``x-audio-last: true``."""
        if not span_id or span_id in self._finalized_span_ids:
            return
        await self._post_chunk(
            kind=kind,
            span_id=span_id,
            trace_id=trace_id,
            sample_rate=16000,
            num_channels=1,
            pcm=b"",
            frame_count=0,
            is_last=True,
            start_ms=0,
        )

    async def _flush_batch(self, batch: _PendingBatch, *, is_last: bool = False) -> None:
        if batch.frame_count == 0 and not is_last:
            return

        if batch.frame_count > 0:
            if batch.span_id:
                self._known_span_ids.add(batch.span_id)
                self._span_roles[batch.span_id] = batch.kind
                if batch.trace_id:
                    self._span_trace_ids[batch.span_id] = batch.trace_id
            await self._post_chunk(
                kind=batch.kind,
                span_id=batch.span_id,
                trace_id=batch.trace_id,
                sample_rate=batch.sample_rate,
                num_channels=batch.num_channels,
                pcm=batch.pcm_bytes,
                frame_count=batch.frame_count,
                is_last=False,
                start_ms=batch.start_ms,
            )

        if is_last and batch.span_id:
            await self._send_span_last(
                kind=batch.kind,
                span_id=batch.span_id,
                trace_id=batch.trace_id or self._span_trace_ids.get(batch.span_id, ""),
            )

    async def _post_chunk(
        self,
        *,
        kind: str,
        span_id: str,
        trace_id: str,
        sample_rate: int,
        num_channels: int,
        pcm: bytes,
        frame_count: int,
        is_last: bool,
        start_ms: int = 0,
    ) -> None:
        if self._circuit_tripped:
            return

        is_span = bool(span_id)

        headers: Dict[str, str] = {
            "Content-Type": "application/octet-stream",
            "x-audio-session-id": self._session_id,
            "x-audio-trace-id": trace_id,
            "x-audio-role": kind,
            "x-audio-start-ms": str(start_ms),
            "x-audio-sample-rate": str(sample_rate or 16000),
            "x-audio-channels": str(num_channels or 1),
            "x-audio-bit-depth": "16",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        for k, v in self._auth_headers.items():
            headers.setdefault(k, v)

        if is_span:
            headers["x-audio-span-id"] = span_id
            seq = self._seq.get(span_id, 0)
            headers["x-audio-seq"] = str(seq)
            if is_last:
                headers["x-audio-last"] = "true"
            self._known_span_ids.add(span_id)
            self._span_roles[span_id] = kind
            if trace_id:
                self._span_trace_ids[span_id] = trace_id

        self.stats.chunks_sent += 1
        self.stats.frames_sent += frame_count
        self.stats.bytes_sent += len(pcm)

        ok = await self._post_one(pcm, headers)

        if is_span and ok:
            self._seq[span_id] = self._seq.get(span_id, 0) + 1
            self._sent_span_ids.add(span_id)
            if is_last:
                self._finalized_span_ids.add(span_id)

        logger.debug(
            "netra.audio: sent chunk span_id=%s role=%s frames=%d bytes=%d last=%s start_ms=%d",
            span_id or "(noise)",
            kind,
            frame_count,
            len(pcm),
            is_last,
            start_ms,
        )

    async def _post_one(self, pcm: bytes, headers: Dict[str, str]) -> bool:
        """POST to the endpoint with a short retry.  Returns True on 2xx."""
        last_exc: Optional[Exception] = None
        for attempt in range(_POST_RETRIES):
            try:
                t0 = time.monotonic()
                resp = await self._client.post(self._url, content=pcm, headers=headers)
                elapsed_ms = (time.monotonic() - t0) * 1000
                self.stats.total_send_time_ms += elapsed_ms

                if resp.status_code < 400:
                    self._consecutive_failures = 0
                    return True

                self.stats.errors += 1
                if resp.status_code in (401, 403):
                    self._trip_circuit(f"HTTP {resp.status_code} — credentials will not fix themselves mid-call")
                    return False
                self._consecutive_failures += 1
                self._check_circuit()
                logger.warning(
                    "netra.audio: chunk POST failed (attempt=%d): %d %s",
                    attempt + 1,
                    resp.status_code,
                    resp.text[:200],
                )
            except Exception as exc:
                last_exc = exc
                self.stats.errors += 1
                self._consecutive_failures += 1
                self._check_circuit()
                logger.warning(
                    "netra.audio: chunk POST error (attempt=%d): %s",
                    attempt + 1,
                    exc,
                )

            if attempt < _POST_RETRIES - 1:
                await asyncio.sleep(0.05 * (attempt + 1))

        if last_exc:
            logger.warning("netra.audio: giving up after retries")
        return False

    async def _post_session_last(self) -> None:
        """Send a bodyless request with ``x-audio-session-last: true``."""
        headers: Dict[str, str] = {
            "x-audio-session-id": self._session_id,
            "x-audio-session-last": "true",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        for k, v in self._auth_headers.items():
            headers.setdefault(k, v)
        await self._post_one(b"", headers)

    def _trip_circuit(self, reason: str) -> None:
        if not self._circuit_tripped:
            self._circuit_tripped = True
            self.stats.circuit_tripped = True
            logger.warning(
                "netra.audio: circuit breaker tripped (session=%s): %s. "
                "No further audio will be sent for this session",
                self._session_id,
                reason,
            )

    def _check_circuit(self) -> None:
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self._trip_circuit(f"{self._consecutive_failures} consecutive failures")
