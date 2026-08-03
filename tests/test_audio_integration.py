"""Tests for the LiveKit call-audio capture pipeline.

The sender is exercised end to end against a recording HTTP server defined in
this module, so the ``x-audio-*`` wire contract the Netra backend depends on is
asserted on real requests rather than on a mock's call args. The coordinator and
the span processor are tested directly, with a stub sender.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Awaitable, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from netra.instrumentation.livekit.audio_capture import (
    AudioCoordinatorRegistry,
    SessionAudioCoordinator,
    audio_coordinators,
    build_audio_sender,
    stop_audio_capture,
)
from netra.instrumentation.livekit.audio_processor import AudioSpanProcessor
from netra.instrumentation.livekit.audio_sender import AudioChunkSender
from netra.instrumentation.livekit.audio_types import (
    HEADER_HEARD_MS,
    HEADER_LAST_CHUNK,
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
    speaker_roles_from,
)

# 24kHz mono 16-bit — what livekit-agents delivers by default.
SAMPLE_RATE_HZ = 24000
BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1000
SAMPLES_PER_FRAME = 480
FRAME_BYTES = SAMPLES_PER_FRAME * 2
FRAME_MS = FRAME_BYTES // BYTES_PER_MS

USER_SPAN_ID = "aaaabbbbccccdddd"
AGENT_SPAN_ID = "1111222233334444"
TRACE_ID = "0123456789abcdef0123456789abcdef"

# Large enough that no test hits a batch boundary it did not ask for.
UNBOUNDED_BYTES = 10_000_000
UNBOUNDED_FRAMES = 10_000
LONG_INTERVAL_SECONDS = 30.0


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
        "batch_interval_seconds": LONG_INTERVAL_SECONDS,
        "max_batch_frames": UNBOUNDED_FRAMES,
        "flush_at_bytes": UNBOUNDED_BYTES,
        "max_request_bytes": UNBOUNDED_BYTES,
    }
    settings.update(overrides)
    return AudioChunkSender(**settings)


def enqueue_frames(sender: AudioChunkSender, count: int, *, role: SpeakerRole, span_id: str) -> None:
    """Enqueue *count* identical frames for one span."""
    for _ in range(count):
        sender.enqueue(make_frame(), role=role, trace_id=TRACE_ID, span_id=span_id)


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

    def test_speaker_roles_from_keeps_known_roles_and_drops_the_rest(self) -> None:
        assert speaker_roles_from(["user", "agent"]) == frozenset(SpeakerRole)
        assert speaker_roles_from(["user"]) == frozenset({SpeakerRole.USER})
        assert speaker_roles_from(["hold_music"]) == frozenset()


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
            assert HEADER_SEQUENCE not in chunk.headers
            assert HEADER_LAST_CHUNK not in chunk.headers
            assert chunk.headers[HEADER_TRACE_ID] == TRACE_ID

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

    def test_interrupt_after_the_span_closed_sends_a_correction(self, ingest_server) -> None:
        url, recorder = ingest_server

        async def scenario(sender: AudioChunkSender) -> None:
            enqueue_frames(sender, 5, role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            sender.mark_audio_end(role=SpeakerRole.AGENT, span_id=AGENT_SPAN_ID)
            # LiveKit routinely reports the interrupt only after the span ended.
            sender.interrupt_agent_span(span_id=AGENT_SPAN_ID, playback_ms=40)

        run_call(url, scenario)

        corrections = [r for r in recorder.chunks_for(AGENT_SPAN_ID) if HEADER_HEARD_MS in r.headers]
        assert len(corrections) == 1
        assert corrections[0].headers[HEADER_HEARD_MS] == "40"
        assert corrections[0].headers[HEADER_LAST_CHUNK] == "true"

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


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class TestSessionAudioCoordinator:
    def test_a_frame_inside_a_speaking_span_carries_that_span_id(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)

        coordinator.on_frame(SpeakerRole.USER, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["role"] is SpeakerRole.USER
        assert kwargs["span_id"] == USER_SPAN_ID
        assert kwargs["trace_id"] == TRACE_ID

    def test_a_frame_between_turns_is_sent_with_no_span_id(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator._session_trace_id = TRACE_ID

        coordinator.on_frame(SpeakerRole.USER, make_frame())

        kwargs = sender.enqueue.call_args.kwargs
        assert kwargs["span_id"] == ""
        assert kwargs["trace_id"] == TRACE_ID

    def test_a_disabled_role_is_never_streamed(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender, enabled_roles=frozenset({SpeakerRole.USER}))
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)

        coordinator.on_frame(SpeakerRole.AGENT, make_frame())
        coordinator.on_frame(SpeakerRole.USER, make_frame())

        assert sender.enqueue.call_count == 1
        assert sender.enqueue.call_args.kwargs["role"] is SpeakerRole.USER

    def test_closing_a_span_finalizes_its_recording(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)

        coordinator.on_speaking_end(SpeakerRole.USER)

        sender.mark_audio_end.assert_called_once_with(role=SpeakerRole.USER, span_id=USER_SPAN_ID)

    def test_close_finalizes_every_span_still_recording(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.USER, trace_id=TRACE_ID, span_id=USER_SPAN_ID)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)

        coordinator.close()

        finalized = {call.kwargs["role"] for call in sender.mark_audio_end.call_args_list}
        assert finalized == {SpeakerRole.USER, SpeakerRole.AGENT}

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

        sender.interrupt_agent_span.assert_called_once_with(span_id=AGENT_SPAN_ID, playback_ms=750)

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

        sender.interrupt_agent_span.assert_called_once_with(span_id=AGENT_SPAN_ID, playback_ms=200)

    def test_playback_that_was_not_interrupted_needs_no_correction(self) -> None:
        sender = MagicMock()
        coordinator = SessionAudioCoordinator(sender=sender)
        coordinator.on_speaking_start(SpeakerRole.AGENT, trace_id=TRACE_ID, span_id=AGENT_SPAN_ID)

        coordinator.on_playback_finished(MagicMock(interrupted=False, playback_position=2.0))

        sender.interrupt_agent_span.assert_not_called()


# ---------------------------------------------------------------------------
# Registry and span processor
# ---------------------------------------------------------------------------


def make_span(name: str, *, trace_id: int, span_id: int = 0xABCD) -> MagicMock:
    """Build a span whose context reports the given ids."""
    span = MagicMock()
    span.name = name
    span.get_span_context.return_value = MagicMock(is_valid=True, trace_id=trace_id, span_id=span_id)
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

        span = make_span(span_name, trace_id=trace_id, span_id=0x1234567890ABCDEF)
        processor.on_start(span)

        assert sender.enqueue.call_count == 0
        coordinator.on_frame(role, make_frame())
        assert sender.enqueue.call_args.kwargs["span_id"] == format(0x1234567890ABCDEF, "016x")

        processor.on_end(span)
        sender.mark_audio_end.assert_called_once_with(role=role, span_id=format(0x1234567890ABCDEF, "016x"))

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
        assert sender._batch_interval_seconds == 0.25
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
