"""Tests for the LiveKit call-audio capture pipeline.

The sender is exercised end to end against a recording HTTP server defined in
this module, so the ``x-audio-*`` wire contract the Netra backend depends on is
asserted on real requests rather than on a mock's call args. The coordinator and
the span processor are tested directly, with a stub sender.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Awaitable, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from netra.config import Config
from netra.instrumentation.libraries.livekit.audio_capture import (
    AudioCoordinatorRegistry,
    SessionAudioCoordinator,
    audio_coordinators,
    build_audio_sender,
    start_audio_capture,
    stop_audio_capture,
)
from netra.instrumentation.libraries.livekit.audio_processor import AudioSpanProcessor
from netra.instrumentation.libraries.livekit.audio_sender import AudioChunkSender
from netra.instrumentation.libraries.livekit.audio_types import (
    HEADER_HEARD_MS,
    HEADER_LAST_CHUNK,
    HEADER_PARENT_SPAN_ID,
    HEADER_ROLE,
    HEADER_SEQUENCE,
    HEADER_SESSION_ID,
    HEADER_SESSION_LAST,
    HEADER_SPAN_ID,
    HEADER_TRACE_ID,
    NETRA_AUDIO_DROPPED_FRAMES,
    NETRA_AUDIO_SENT_BYTES,
    NETRA_AUDIO_SENT_CHUNKS,
    SpeakerRole,
    pcm_byte_offset_at,
)

# 24kHz mono 16-bit — what livekit-agents delivers by default.
SAMPLE_RATE_HZ = 24000
BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1000
SAMPLES_PER_FRAME = 480
FRAME_BYTES = SAMPLES_PER_FRAME * 2
FRAME_MS = FRAME_BYTES // BYTES_PER_MS

USER_SPAN_ID = "aaaabbbbccccdddd"
AGENT_SPAN_ID = "1111222233334444"
PARENT_SPAN_ID = "ffffeeeebbbbcccc"
TRACE_ID = "0123456789abcdef0123456789abcdef"

# Large enough that no test hits a batch boundary it did not ask for.
UNBOUNDED_BYTES = 10_000_000
UNBOUNDED_FRAMES = 10_000


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAudioFrame:
    """Minimal stand-in for ``livekit.rtc.AudioFrame``."""

    pcm: bytes
    sample_rate: int = SAMPLE_RATE_HZ
    num_channels: int = 1

    @property
    def data(self) -> memoryview:
        return memoryview(self.pcm)


def make_frame(sample_count: int = SAMPLES_PER_FRAME, value: int = 1000) -> FakeAudioFrame:
    """Build one frame of constant-amplitude PCM."""
    return FakeAudioFrame(pcm=value.to_bytes(2, "little", signed=True) * sample_count)


async def _async_noop(*args: Any, **kwargs: Any) -> None:
    """Stand in for an awaitable the test does not care about."""


@dataclass
class RecordedRequest:
    """One request the ingest server received."""

    headers: Dict[str, str]
    body: bytes

    @property
    def span_id(self) -> Optional[str]:
        return self.headers.get(HEADER_SPAN_ID)

    @property
    def is_last(self) -> bool:
        return self.headers.get(HEADER_LAST_CHUNK) == "true"


@dataclass
class IngestRecorder:
    """Thread-safe record of what the ingest server received."""

    status_code: int = 200
    # Held before replying, so a test can stall the send loop mid-POST the way a
    # degraded backend would. Only the teardown-budget tests set it.
    delay_seconds: float = 0.0
    requests: List[RecordedRequest] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, request: RecordedRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def snapshot(self) -> List[RecordedRequest]:
        with self._lock:
            return list(self.requests)

    def chunks_for(self, span_id: str) -> List[RecordedRequest]:
        return [request for request in self.snapshot() if request.span_id == span_id]

    def bytes_for(self, span_id: str) -> int:
        return sum(len(request.body) for request in self.chunks_for(span_id))


class _IngestHandler(BaseHTTPRequestHandler):
    """Records every POST into the server's recorder and replies with its status."""

    recorder: IngestRecorder

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.recorder.record(
            RecordedRequest(
                headers={name.lower(): value for name, value in self.headers.items()},
                body=body,
            )
        )
        if self.recorder.delay_seconds:
            time.sleep(self.recorder.delay_seconds)
        self.send_response(self.recorder.status_code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr access log."""


@pytest.fixture()
def ingest_server():
    """Serve the audio-ingest endpoint on a random port; yield (url, recorder)."""
    recorder = IngestRecorder()
    handler = type("_BoundIngestHandler", (_IngestHandler,), {"recorder": recorder})

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/telemetry/v1/audio/chunk", recorder
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def build_sender(url: str, **overrides: Any) -> AudioChunkSender:
    """Build a sender that only flushes when a test tells it to."""
    settings: Dict[str, Any] = {
        "url": url,
        "session_id": "session-under-test",
        "api_key": "test-key",
        "max_batch_frames": UNBOUNDED_FRAMES,
        "flush_at_bytes": UNBOUNDED_BYTES,
        "max_request_bytes": UNBOUNDED_BYTES,
    }
    settings.update(overrides)
    return AudioChunkSender(**settings)


def enqueue_frames(
    sender: AudioChunkSender,
    count: int,
    *,
    role: SpeakerRole,
    span_id: str,
    parent_span_id: str = "",
) -> None:
    """Enqueue *count* identical frames for one span."""
    for _ in range(count):
        sender.enqueue(
            make_frame(),
            role=role,
            trace_id=TRACE_ID,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )


def run_call(url: str, scenario: Callable[[AudioChunkSender], Awaitable[None]], **overrides: Any) -> AudioChunkSender:
    """Drive one whole call against the ingest server and return its sender.

    The repo has no pytest-asyncio, so each async scenario gets its own loop.

    Args:
        url: The ingest URL to send to.
        scenario: What the call does between start and close.
        **overrides: Sender settings to override for this call.

    Returns:
        The closed sender, for its statistics.
    """

    async def drive() -> AudioChunkSender:
        sender = build_sender(url, **overrides)
        await sender.start()
        await scenario(sender)
        await sender.end_session()
        return sender

    return asyncio.run(drive())


# ---------------------------------------------------------------------------
# audio_types
# ---------------------------------------------------------------------------


class TestAudioTypes:
    @pytest.mark.parametrize(
        "playback_ms,expected_bytes",
        [
            (0, 0),
            (-100, 0),
            (1, 48),
            (400, 19200),
            (1000, 48000),
        ],
    )
    def test_pcm_byte_offset_converts_playback_time_to_bytes(self, playback_ms: int, expected_bytes: int) -> None:
        offset = pcm_byte_offset_at(playback_ms=playback_ms, sample_rate_hz=SAMPLE_RATE_HZ, channel_count=1)
        assert offset == expected_bytes

    def test_pcm_byte_offset_rounds_down_to_a_whole_sample_frame(self) -> None:
        # 11025Hz stereo: 44.1 bytes/ms, so 7ms is 308.7 bytes — not a frame boundary.
        offset = pcm_byte_offset_at(playback_ms=7, sample_rate_hz=11025, channel_count=2)

        frame_size = 2 * 2
        assert offset % frame_size == 0
        assert offset == 308

    @pytest.mark.parametrize("sample_rate_hz,channel_count", [(0, 1), (24000, 0), (-1, 1)])
    def test_pcm_byte_offset_rejects_an_unplayable_format(self, sample_rate_hz: int, channel_count: int) -> None:
        with pytest.raises(ValueError, match="unplayable PCM format"):
            pcm_byte_offset_at(playback_ms=100, sample_rate_hz=sample_rate_hz, channel_count=channel_count)


# ---------------------------------------------------------------------------
# Sender: wire contract
# ---------------------------------------------------------------------------


class TestAudioChunkSenderWireContract:
    def test_frames_reach_the_endpoint_and_the_session_is_closed(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 10, role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        sender = run_call(url, scenario)

        requests = recorder.snapshot()
        assert recorder.bytes_for(USER_SPAN_ID) == 10 * FRAME_BYTES
        assert sender.stats.frames_sent == 10
        assert sender.stats.errors == 0

        session_end = [r for r in requests if r.headers.get(HEADER_SESSION_LAST) == "true"]
        assert len(session_end) == 1
        assert session_end[0].body == b""
        assert session_end[0].headers[HEADER_SESSION_ID] == "session-under-test"

    def test_every_request_carries_the_session_and_credential(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 3, role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        run_call(url, scenario, auth_headers={"Authorization": "Bearer token"})

        assert recorder.snapshot()
        for request in recorder.snapshot():
            assert request.headers[HEADER_SESSION_ID] == "session-under-test"
            assert request.headers["x-api-key"] == "test-key"
            assert request.headers["authorization"] == "Bearer token"

    def test_sequence_is_zero_based_and_monotonic_per_span(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 6, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            enqueue_frames(sender, 4, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)

        # Two frames per request, so each span spans several sequence numbers.
        run_call(url, scenario, max_batch_frames=2)

        for span_id, expected_chunks in ((USER_SPAN_ID, 3), (AGENT_SPAN_ID, 2)):
            sequences = [int(r.headers[HEADER_SEQUENCE]) for r in recorder.chunks_for(span_id)]
            # Each span sends its data chunks plus one terminator.
            assert sequences == list(range(expected_chunks + 1)), span_id

    def test_a_span_is_terminated_exactly_once(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 3, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            sender.mark_audio_end(role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            # A duplicate end marker must not produce a second terminator.
            sender.mark_audio_end(role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        run_call(url, scenario)

        terminators = [r for r in recorder.chunks_for(USER_SPAN_ID) if r.is_last]
        assert len(terminators) == 1
        assert terminators[0].body == b""
        assert terminators[0].headers[HEADER_ROLE] == "user"

    def test_a_span_left_open_is_terminated_at_session_end(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            # No mark_audio_end: the session closes with the span still recording.
            enqueue_frames(sender, 2, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)

        run_call(url, scenario)

        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1

    def test_audio_outside_a_span_carries_no_span_headers(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 4, role=SpeakerRole.USER, span_id="")

        run_call(url, scenario)

        chunks = [r for r in recorder.snapshot() if r.body]
        assert chunks
        for chunk in chunks:
            assert HEADER_SPAN_ID not in chunk.headers
            assert HEADER_PARENT_SPAN_ID not in chunk.headers
            assert HEADER_SEQUENCE not in chunk.headers
            assert HEADER_LAST_CHUNK not in chunk.headers
            assert chunk.headers[HEADER_TRACE_ID] == TRACE_ID

    def test_span_chunks_carry_the_parent_span_id(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(
                sender,
                3,
                role=SpeakerRole.USER,
                span_id=USER_SPAN_ID,
                parent_span_id=PARENT_SPAN_ID,
            )
            sender.mark_audio_end(
                role=SpeakerRole.USER,
                span_id=USER_SPAN_ID,
                parent_span_id=PARENT_SPAN_ID,
            )

        run_call(url, scenario)

        for request in recorder.chunks_for(USER_SPAN_ID):
            assert request.headers[HEADER_PARENT_SPAN_ID] == PARENT_SPAN_ID

    def test_span_chunks_omit_parent_when_the_speaking_span_is_a_root(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 2, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            sender.mark_audio_end(role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        run_call(url, scenario)

        for request in recorder.chunks_for(USER_SPAN_ID):
            assert HEADER_PARENT_SPAN_ID not in request.headers

    def test_a_request_body_never_exceeds_the_configured_ceiling(self, ingest_server) -> None:
        url, recorder = ingest_server
        ceiling = FRAME_BYTES * 3

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 10, role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        run_call(url, scenario, flush_at_bytes=ceiling, max_request_bytes=ceiling)

        bodies = [len(r.body) for r in recorder.chunks_for(USER_SPAN_ID)]
        assert bodies, "expected at least one chunk"
        assert max(bodies) <= ceiling
        assert sum(bodies) == 10 * FRAME_BYTES


# ---------------------------------------------------------------------------
# Sender: interrupts
# ---------------------------------------------------------------------------


class TestAudioChunkSenderInterrupts:
    def test_interrupt_trims_pending_audio_to_what_was_heard(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            # 50 frames = 1000ms of agent speech, of which 400ms was heard.
            enqueue_frames(sender, 50, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=400)

        run_call(url, scenario)

        heard_bytes = 400 * BYTES_PER_MS
        assert recorder.bytes_for(AGENT_SPAN_ID) == heard_bytes

        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1
        assert terminators[0].headers[HEADER_HEARD_MS] == "400"

    def test_frames_queued_after_an_interrupt_are_discarded(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 5, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=50)
            # Frames already in flight when the caller cut in; never played out.
            enqueue_frames(sender, 10, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)

        run_call(url, scenario)

        assert recorder.bytes_for(AGENT_SPAN_ID) == 50 * BYTES_PER_MS

    def test_interrupt_after_the_span_closed_sends_a_single_is_last_with_heard_ms(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 5, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            sender.mark_audio_end(role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            # LiveKit routinely reports the interrupt only after the span ended.
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=40)

        run_call(url, scenario)

        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1, "agent span must produce exactly one is_last chunk"
        assert terminators[0].headers[HEADER_HEARD_MS] == "40"
        assert terminators[0].body == b""

    def test_interrupt_after_the_heard_audio_was_already_sent_only_marks_the_cut(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 5, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            await asyncio.sleep(0.1)
            # Only the first frame's worth was heard, but 5 already went out.
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=FRAME_MS)

        # Flush every frame immediately, so all the audio is delivered up front.
        run_call(url, scenario, max_batch_frames=1)

        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1
        assert terminators[0].body == b"", "nothing more to send; the endpoint trims"
        assert terminators[0].headers[HEADER_HEARD_MS] == str(FRAME_MS)

    def test_interrupt_marks_the_cut_when_a_pending_batch_is_past_the_heard_point(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            # Two frames flush on the batch boundary, so they are already
            # delivered; a third stays pending.
            enqueue_frames(sender, 3, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            await asyncio.sleep(0.1)
            # Only the first frame was heard, which is behind what already went
            # out — so the pending batch trims to nothing but the endpoint still
            # has to be told where to cut.
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=FRAME_MS)

        run_call(url, scenario, max_batch_frames=2)

        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1, "the span must be terminated, not left open until session end"
        assert terminators[0].headers[HEADER_HEARD_MS] == str(FRAME_MS)
        assert terminators[0].headers[HEADER_SPAN_ID] == AGENT_SPAN_ID

    def test_a_trimmed_chunk_counts_only_the_frames_it_actually_sent(self, ingest_server) -> None:
        url, _ = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            # 5 frames pending, of which 2 frames' worth was heard.
            enqueue_frames(sender, 5, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=FRAME_MS * 2)

        sender = run_call(url, scenario)

        assert sender.stats.bytes_sent == FRAME_BYTES * 2
        assert sender.stats.frames_sent == 2, "the untrimmed frame count would report 5"


# ---------------------------------------------------------------------------
# Sender: failure handling
# ---------------------------------------------------------------------------


class TestAudioChunkSenderFailures:
    def test_a_full_queue_drops_frames_instead_of_blocking(self, ingest_server) -> None:
        url, _ = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            # Enqueued without awaiting, so the loop cannot drain any of them.
            enqueue_frames(sender, 50, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            assert sender.stats.frames_dropped == 45

        run_call(url, scenario, max_queue_frames=5)

    def test_a_rejected_credential_stops_the_call_from_sending_more(self, ingest_server) -> None:
        url, recorder = ingest_server
        recorder.status_code = 401

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 10, role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        sender = run_call(url, scenario, max_batch_frames=1)

        assert sender.stats.circuit_tripped is True
        assert sender.stats.chunks_sent == 0
        # One rejected attempt, then nothing further — not one per frame, and no
        # retry of a credential that cannot become valid mid-call.
        assert len(recorder.snapshot()) == 1

    def test_a_server_error_is_retried_then_given_up_on(self, ingest_server) -> None:
        url, recorder = ingest_server
        recorder.status_code = 500

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 1, role=SpeakerRole.USER, span_id=USER_SPAN_ID)

        sender = run_call(url, scenario)

        assert sender.stats.chunks_sent == 0
        assert sender.stats.errors > 1, "a 5xx is worth retrying"
        assert sender.stats.bytes_sent == 0, "nothing was accepted"

    def test_end_session_is_idempotent(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 2, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            # run_call closes it again once this returns.
            await sender.end_session()

        run_call(url, scenario)

        session_ends = [r for r in recorder.snapshot() if r.headers.get(HEADER_SESSION_LAST) == "true"]
        assert len(session_ends) == 1

    def test_a_given_up_chunk_does_not_leave_its_sequence_number_to_the_next_chunk(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            recorder.status_code = 500
            sender.enqueue(make_frame(value=1111), role=SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
            await asyncio.sleep(0.3)  # both attempts fail; the chunk is given up on
            recorder.status_code = 200
            sender.enqueue(make_frame(value=2222), role=SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
            await asyncio.sleep(0.3)

        run_call(url, scenario, max_batch_frames=1)

        # A number may repeat only across retries of the *same* bytes: that is what
        # makes it an idempotency key. Reusing it for different audio would have the
        # endpoint either drop the new chunk as a duplicate or overwrite the old.
        audio_by_sequence: Dict[str, set] = {}
        for request in recorder.chunks_for(AGENT_SPAN_ID):
            if request.body:
                audio_by_sequence.setdefault(request.headers[HEADER_SEQUENCE], set()).add(request.body)
        reused = {seq: len(bodies) for seq, bodies in audio_by_sequence.items() if len(bodies) > 1}
        assert not reused, f"sequence reused across distinct audio: {reused}"
        # The lost chunk still consumed its slot, so the gap is visible.
        assert sorted(audio_by_sequence) == ["0", "1"]

    def test_end_session_spends_one_total_budget_not_one_per_wait(self, ingest_server) -> None:
        url, recorder = ingest_server
        recorder.delay_seconds = 2.0

        async def drive() -> float:
            sender = build_sender(url, max_batch_frames=1)
            await sender.start()
            enqueue_frames(sender, 4, role=SpeakerRole.USER, span_id=USER_SPAN_ID)
            started_at = time.monotonic()
            await sender.end_session(drain_timeout_seconds=0.5)
            return time.monotonic() - started_at

        elapsed = asyncio.run(drive())

        # The two internal waits share the 0.5s deadline. Taking it each would put
        # this at 1s+, and the pre-fix 30s-per-wait default at a minute.
        assert elapsed < 1.5, f"teardown took {elapsed:.2f}s for a 0.5s budget"

    def test_a_tripped_circuit_does_not_warn_once_per_open_span(self, ingest_server, caplog) -> None:
        url, recorder = ingest_server
        recorder.status_code = 500

        async def scenario(sender: AudioChunkSender) -> None:
            for index in range(8):
                span_id = f"{index:016x}"
                enqueue_frames(sender, 1, role=SpeakerRole.USER, span_id=span_id)
                await asyncio.sleep(0.05)

        with caplog.at_level("WARNING"):
            sender = run_call(url, scenario, max_batch_frames=1)

        assert sender.stats.circuit_tripped is True
        left_open = [r for r in caplog.records if "finalizing span left open" in r.getMessage()]
        assert left_open == [], "the circuit breaker already said why once; per-span warnings bury it"

    def test_a_marker_from_another_thread_is_enqueued_on_the_loop_thread(self, ingest_server) -> None:
        url, recorder = ingest_server
        threads: Dict[str, Any] = {}

        async def scenario(sender: AudioChunkSender) -> None:
            # OTel invokes span callbacks on whichever thread ended the span, and
            # asyncio.Queue is not thread-safe. The marker must therefore reach the
            # queue from the loop's own thread, never from the foreign one.
            threads["loop"] = threading.get_ident()
            enqueued_from: List[int] = []
            original_put = sender._queue.put_nowait

            def recording_put(message: Any) -> None:
                enqueued_from.append(threading.get_ident())
                original_put(message)

            sender._queue.put_nowait = recording_put  # type: ignore[method-assign]
            worker = threading.Thread(
                target=lambda: sender.mark_audio_end(role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            )
            worker.start()
            await asyncio.to_thread(worker.join)
            threads["worker"] = worker.ident
            await asyncio.sleep(0.1)
            threads["enqueued_from"] = enqueued_from

        run_call(url, scenario)

        assert threads["enqueued_from"], "the marker never reached the queue"
        off_loop = [ident for ident in threads["enqueued_from"] if ident != threads["loop"]]
        assert off_loop == [], (
            f"queue touched from thread(s) {off_loop} instead of the loop thread "
            f"{threads['loop']} (the worker was {threads['worker']})"
        )
        terminators = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if r.is_last]
        assert len(terminators) == 1


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class TestSessionAudioCoordinator:
    def test_a_frame_inside_a_speaking_span_carries_that_span_id(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(
            SpeakerRole.USER,
            trace_id=TRACE_ID,
            span_id=USER_SPAN_ID,
            parent_span_id=PARENT_SPAN_ID,
        )

        coordinator.on_frame(SpeakerRole.USER, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["role"] is SpeakerRole.USER
        assert kwargs["span_id"] == USER_SPAN_ID
        assert kwargs["parent_span_id"] == PARENT_SPAN_ID
        assert kwargs["trace_id"] == TRACE_ID

    def test_a_frame_between_turns_is_sent_with_no_span_id(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator._session_trace_id = TRACE_ID

        coordinator.on_frame(SpeakerRole.USER, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["span_id"] == ""
        assert kwargs["parent_span_id"] == ""
        assert kwargs["trace_id"] == TRACE_ID

    def test_both_speakers_are_streamed(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)

        coordinator.on_frame(SpeakerRole.AGENT, make_frame())
        coordinator.on_frame(SpeakerRole.USER, make_frame())

        # Capture is all of the call's audio or none of it — there is no per-role
        # gate to leave one side out.
        streamed = [call.kwargs["role"] for call in sender.enqueue.call_args_list]
        assert streamed == [SpeakerRole.AGENT, SpeakerRole.USER]

    def test_closing_a_span_finalizes_its_recording(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)

        coordinator.on_speaking_end(SpeakerRole.USER)

        sender.mark_audio_end.assert_called_once_with(
            role=SpeakerRole.USER,
            span_id=USER_SPAN_ID,
            parent_span_id="",
            trace_id=TRACE_ID,
        )

    def test_close_finalizes_every_span_still_recording(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)

        coordinator.close()

        finalized = {call.kwargs["role"] for call in sender.mark_audio_end.call_args_list}
        assert finalized == {SpeakerRole.USER, SpeakerRole.AGENT}

    def test_agent_trailing_frames_are_attributed_after_span_end(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(
            SpeakerRole.AGENT,
            trace_id=TRACE_ID,
            span_id=AGENT_SPAN_ID,
            parent_span_id=PARENT_SPAN_ID,
        )
        coordinator.on_speaking_end(SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)

        coordinator.on_frame(SpeakerRole.AGENT, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["span_id"] == AGENT_SPAN_ID
        assert kwargs["parent_span_id"] == PARENT_SPAN_ID
        assert kwargs["trace_id"] == TRACE_ID

    def test_agent_active_speech_is_overridden_by_next_span(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        coordinator.on_speaking_end(SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)

        new_span_id = "5555666677778888"
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=new_span_id)
        coordinator.on_frame(SpeakerRole.AGENT, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["span_id"] == new_span_id

    def test_close_is_idempotent(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)

        coordinator.close()
        coordinator.close()

        assert sender.mark_audio_end.call_count == 1


class TestSessionAudioCoordinatorInterrupts:
    @staticmethod
    def _interrupted_coordinator(sender: MagicMock) -> SessionAudioCoordinator:
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        coordinator.on_output_buffer_cleared()
        return coordinator

    def test_agent_frames_stop_once_the_caller_cuts_in(self) -> None:
        sender = MagicMock()
        coordinator = self._interrupted_coordinator(sender)

        coordinator.on_frame(SpeakerRole.AGENT, make_frame())

        sender.enqueue.assert_not_called()

    def test_the_playback_position_is_reported_as_the_audio_heard(self) -> None:
        sender = MagicMock()
        coordinator = self._interrupted_coordinator(sender)

        event = MagicMock(interrupted=True, playback_position=0.75)
        coordinator.on_playback_finished(event)

        sender.interrupt_agent_span.assert_called_once_with(
            span_id=AGENT_SPAN_ID,
            playback_ms=750,
            parent_span_id="",
        )

    def test_an_interrupted_span_is_not_finalized_at_its_full_length(self) -> None:
        sender = MagicMock()
        coordinator = self._interrupted_coordinator(sender)

        coordinator.on_speaking_end(SpeakerRole.AGENT)

        sender.mark_audio_end.assert_not_called()

    def test_the_span_id_survives_the_span_closing_before_the_interrupt_is_reported(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        # LiveKit's ordering: the span ends, and only then does clear_buffer fire.
        coordinator.on_speaking_end(SpeakerRole.AGENT)
        coordinator.on_output_buffer_cleared()

        coordinator.on_playback_finished(MagicMock(interrupted=True, playback_position=0.2))

        sender.interrupt_agent_span.assert_called_once_with(
            span_id=AGENT_SPAN_ID,
            playback_ms=200,
            parent_span_id="",
        )

    def test_playback_that_was_not_interrupted_needs_no_correction(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)

        coordinator.on_playback_finished(MagicMock(interrupted=False, playback_position=2.0))

        sender.interrupt_agent_span.assert_not_called()

    def test_closing_while_agent_is_speaking_trims_to_what_was_heard(self) -> None:
        sender = MagicMock()
        sender.end_session = _async_noop
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        coordinator.on_frame(SpeakerRole.AGENT, make_frame())
        coordinator.on_playback_started(created_at=time.time() - 0.4)

        async def drive() -> None:
            await coordinator.prepare_close()
            # LiveKit reports heard position during its aclose, between prepare and finish.
            coordinator.on_playback_finished(MagicMock(interrupted=True, playback_position=0.4))
            await coordinator.finish_close()

        asyncio.run(drive())

        sender.interrupt_agent_span.assert_called_once_with(
            span_id=AGENT_SPAN_ID,
            playback_ms=400,
            parent_span_id="",
        )
        assert not any(call.kwargs.get("role") is SpeakerRole.AGENT for call in sender.mark_audio_end.call_args_list)

    def test_closing_after_agent_finished_does_not_trim_prior_turn(self) -> None:
        sender = MagicMock()
        sender.end_session = _async_noop
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        # Trailing-attribution keep-alive: active speech remains after OTel end.
        coordinator.on_speaking_end(SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
        coordinator.on_playback_finished(MagicMock(interrupted=False, playback_position=2.0))

        asyncio.run(coordinator.aclose())

        sender.interrupt_agent_span.assert_not_called()

    def test_closing_mid_speech_falls_back_to_wall_clock_when_playout_wait_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "netra.instrumentation.libraries.livekit.audio_capture._PLAYBACK_WAIT_ON_CLOSE_SECONDS",
            0.05,
        )
        sender = MagicMock()
        sender.end_session = _async_noop
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)
        started_at = time.time() - 0.25
        coordinator.on_playback_started(created_at=started_at)

        async def drive() -> None:
            await coordinator.prepare_close()
            # No playback_finished arrives; finish should estimate from the clock.
            await coordinator.finish_close()

        asyncio.run(drive())

        playback_ms = sender.interrupt_agent_span.call_args.kwargs["playback_ms"]
        assert 200 <= playback_ms <= 500


# ---------------------------------------------------------------------------
# Registry and span processor
# ---------------------------------------------------------------------------


def make_span(
    name: str,
    *,
    trace_id: int,
    span_id: int = 0xABCD,
    parent_span_id: Optional[int] = None,
) -> MagicMock:
    """Build a span whose context reports the given ids."""
    span = MagicMock()
    span.name = name
    span.get_span_context.return_value = MagicMock(is_valid=True, trace_id=trace_id, span_id=span_id)
    if parent_span_id is None:
        span.parent = None
    else:
        span.parent = MagicMock(is_valid=True, span_id=parent_span_id)
    return span


class TestAudioCoordinatorRegistry:
    def test_a_coordinator_is_handed_out_only_once(self) -> None:
        registry = AudioCoordinatorRegistry()
        coordinator = SessionAudioCoordinator()
        registry.register(1234, coordinator)

        assert registry.unregister(1234) is coordinator
        assert registry.unregister(1234) is None
        assert registry.get(1234) is None

    def test_pop_all_drains_the_registry(self) -> None:
        registry = AudioCoordinatorRegistry()
        registry.register(1, SessionAudioCoordinator())
        registry.register(2, SessionAudioCoordinator())

        assert len(registry.pop_all()) == 2
        assert registry.pop_all() == []


class TestAudioSpanProcessor:
    @pytest.fixture(autouse=True)
    def _clear_registry(self):
        audio_coordinators.pop_all()
        yield
        audio_coordinators.pop_all()

    @pytest.mark.parametrize(
        "span_name,role",
        [("user_speaking", SpeakerRole.USER), ("agent_speaking", SpeakerRole.AGENT)],
    )
    def test_a_speaking_span_opens_and_closes_a_recording(self, span_name: str, role: SpeakerRole) -> None:
        trace_id = 0xAAAABBBBCCCCDDDD
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        audio_coordinators.register(trace_id, coordinator)
        processor = AudioSpanProcessor()

        span = make_span(
            span_name,
            trace_id=trace_id,
            span_id=0x1234567890ABCDEF,
            parent_span_id=0xFEDCBA0987654321,
        )
        processor.on_start(span)

        assert sender.enqueue.call_count == 0
        coordinator.on_frame(role, make_frame())
        assert sender.enqueue.call_args.kwargs["span_id"] == format(0x1234567890ABCDEF, "016x")
        assert sender.enqueue.call_args.kwargs["parent_span_id"] == format(0xFEDCBA0987654321, "016x")

        processor.on_end(span)
        sender.mark_audio_end.assert_called_once_with(
            role=role,
            span_id=format(0x1234567890ABCDEF, "016x"),
            parent_span_id=format(0xFEDCBA0987654321, "016x"),
            trace_id=format(0xAAAABBBBCCCCDDDD, "032x"),
        )

    def test_a_span_from_another_call_is_ignored(self) -> None:
        sender = MagicMock()
        audio_coordinators.register(0x1111, SessionAudioCoordinator(sender=sender))
        processor = AudioSpanProcessor()

        processor.on_start(make_span("user_speaking", trace_id=0x2222))

        sender.enqueue.assert_not_called()

    def test_a_span_that_is_not_speech_is_ignored(self) -> None:
        processor = AudioSpanProcessor()
        span = make_span("llm_request", trace_id=0x1111)

        processor.on_start(span)

        span.get_span_context.assert_not_called()


# ---------------------------------------------------------------------------
# Session wiring
# ---------------------------------------------------------------------------


class TestSessionWiring:
    @pytest.fixture(autouse=True)
    def _clear_registry(self):
        audio_coordinators.pop_all()
        yield
        audio_coordinators.pop_all()

    def test_the_sender_is_built_from_the_configured_limits(self) -> None:
        config = MagicMock(
            api_key="key",
            headers={"x-api-key": "key", "x-tenant": "acme"},
            audio_batch_interval_ms=250,
            audio_batch_bytes=4096,
            audio_max_request_bytes=65536,
            audio_buffer_bytes=960_000,
        )
        config.audio_endpoint.return_value = "https://ingest.example/v1/audio/chunk"

        sender = build_audio_sender(config, "session-1")

        assert sender is not None
        assert sender._flush_at_bytes == 4096
        assert sender._max_request_bytes == 65536
        assert sender._queue.maxsize == 1000
        # Only credential headers are forwarded, never arbitrary config headers.
        assert sender._auth_headers == {"x-api-key": "key"}

    def test_no_endpoint_means_no_sender(self) -> None:
        config = MagicMock()
        config.audio_endpoint.return_value = None

        assert build_audio_sender(config, "session-1") is None

    def test_stopping_a_call_unregisters_it_and_records_what_was_sent(self) -> None:
        sender = MagicMock()
        sender.stats = MagicMock(bytes_sent=4096, chunks_sent=3, frames_dropped=1, errors=0, circuit_tripped=False)
        sender.end_session = _async_noop
        coordinator = SessionAudioCoordinator(sender=sender)
        audio_coordinators.register(0x99, coordinator)
        session_span = MagicMock()

        asyncio.run(stop_audio_capture(0x99, session_span=session_span))

        assert audio_coordinators.get(0x99) is None
        stamped = session_span.set_attributes.call_args.args[0]
        assert stamped[NETRA_AUDIO_SENT_BYTES] == 4096
        assert stamped[NETRA_AUDIO_SENT_CHUNKS] == 3
        assert stamped[NETRA_AUDIO_DROPPED_FRAMES] == 1

    def test_stopping_a_call_twice_is_harmless(self) -> None:
        sender = MagicMock()
        sender.end_session = _async_noop
        audio_coordinators.register(0x99, SessionAudioCoordinator(sender=sender))
        session_span = MagicMock()

        asyncio.run(stop_audio_capture(0x99, session_span=session_span))
        asyncio.run(stop_audio_capture(0x99, session_span=session_span))

        assert session_span.set_attributes.call_count == 1

    def test_a_failed_attach_leaves_no_sender_running(self, ingest_server) -> None:
        url, _ = ingest_server
        config = MagicMock(
            api_key="key",
            headers={"x-api-key": "key"},
            audio_batch_interval_ms=1000,
            audio_batch_bytes=32768,
            audio_max_request_bytes=262144,
            audio_buffer_bytes=2097152,
        )
        config.audio_endpoint.return_value = url

        # A custom AudioOutput whose capture_frame cannot be reassigned, which is
        # what attach() trips over.
        session = MagicMock()
        session.input.audio = None
        unpatchable = MagicMock()
        type(unpatchable).capture_frame = property(lambda self: _async_noop)
        session.output.audio = unpatchable

        async def drive() -> List[asyncio.Task]:
            await start_audio_capture(session, config=config, session_id="s", trace_id=0xABC)
            await asyncio.sleep(0.05)
            return [task for task in asyncio.all_tasks() if task.get_name() == "netra-audio-chunk-sender"]

        leaked = asyncio.run(drive())

        # The sender owns a background task and an HTTP client from start()
        # onwards; if attach() fails after that, nothing else can ever close them.
        assert leaked == [], "a started sender was stranded with no coordinator registered to close it"
        assert audio_coordinators.get(0xABC) is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestAudioConfigResolution:
    def test_a_missing_credential_is_reported_once_not_once_per_session(self, monkeypatch, caplog) -> None:
        for name in list(os.environ):
            if name.startswith(("NETRA_", "OTEL_")):
                monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("NETRA_AUDIO_ENDPOINT", "https://ingest.example/v1/audio/chunk")

        with caplog.at_level("WARNING"):
            config = Config()
            # Every per-session hook asks; the answer is fixed at init time.
            for _ in range(5):
                assert config.audio_endpoint() is None
                assert config.audio_capture_enabled is False

        missing_credential = [r for r in caplog.records if "no credential is configured" in r.getMessage()]
        assert len(missing_credential) == 1

    def test_a_resolved_endpoint_is_the_whole_gate(self, monkeypatch) -> None:
        for name in list(os.environ):
            if name.startswith(("NETRA_", "OTEL_")):
                monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.getnetra.com")
        monkeypatch.setenv("NETRA_AUDIO_ENDPOINT", "https://ingest.example/v1/audio/chunk")
        monkeypatch.setenv("NETRA_API_KEY", "key")
        # A leftover role list from before capture became all-or-nothing must not
        # still gate anything.
        monkeypatch.setenv("NETRA_AUDIO_ROLES", "")

        config = Config()

        assert config.audio_capture_enabled is True
        assert not hasattr(config, "audio_roles")
