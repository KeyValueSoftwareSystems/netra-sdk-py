"""Integration tests for the LiveKit audio capture pipeline.

Tests the ``AudioChunkSender`` against the real ``audio_server`` mock,
plus unit tests for ``AudioSpanProcessor`` and ``AudioHookManager``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from dataclasses import dataclass
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest
from audio_server.server import AudioHandler, AudioSessionStore, set_store

from netra.instrumentation.livekit.audio_sender import AudioChunkSender

# ---------------------------------------------------------------------------
# Fake AudioFrame (avoids livekit dependency)
# ---------------------------------------------------------------------------


@dataclass
class FakeAudioFrame:
    """Minimal stand-in for ``livekit.rtc.AudioFrame``."""

    _pcm: bytes
    sample_rate: int = 24000
    num_channels: int = 1

    @property
    def data(self) -> memoryview:
        return memoryview(self._pcm)


def _make_frame(n_samples: int = 480, value: int = 1000) -> FakeAudioFrame:
    pcm = (value).to_bytes(2, "little", signed=True) * n_samples
    return FakeAudioFrame(_pcm=pcm)


# ---------------------------------------------------------------------------
# Server fixture — runs audio_server on a random port in a background thread
# ---------------------------------------------------------------------------


@pytest.fixture()
def audio_server():
    """Start the mock audio server and yield ``(base_url, store)``."""
    tmpdir = tempfile.mkdtemp(prefix="netra_audio_test_")
    store = AudioSessionStore(output_dir=tmpdir)
    set_store(store)

    server = HTTPServer(("127.0.0.1", 0), AudioHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}/telemetry/v1/audio/chunk", store

    server.shutdown()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sender integration tests
# ---------------------------------------------------------------------------


class TestAudioChunkSender:
    """End-to-end: sender -> HTTP -> audio_server."""

    @pytest.mark.asyncio
    async def test_send_and_end_session(self, audio_server: tuple[str, AudioSessionStore]) -> None:
        chunk_url, store = audio_server

        sender = AudioChunkSender(
            url=chunk_url,
            session_id="test-session-1",
            api_key="test-key",
            batch_interval=0.1,
            max_batch_frames=5,
        )
        await sender.start()

        for _ in range(10):
            sender.enqueue(
                _make_frame(),
                kind="user",
                trace_id="aaaa" * 8,
                span_id="bbbb" * 4,
            )

        await asyncio.sleep(0.5)

        sender.enqueue(
            _make_frame(),
            kind="agent",
            trace_id="aaaa" * 8,
            span_id="cccc" * 4,
        )

        await sender.end_session()

        assert sender.stats.chunks_sent > 0
        assert sender.stats.frames_sent == 11

        session = store.get_session("test-session-1")
        assert session is not None
        assert session["finalized"] is True

    @pytest.mark.asyncio
    async def test_bounded_queue_drops_frames(self, audio_server: tuple[str, AudioSessionStore]) -> None:
        chunk_url, store = audio_server

        sender = AudioChunkSender(
            url=chunk_url,
            session_id="test-drop",
            api_key="test-key",
            batch_interval=10.0,
            max_batch_frames=1000,
            max_queue_size=5,
        )
        await sender.start()

        for _ in range(50):
            sender.enqueue(
                _make_frame(),
                kind="user",
                trace_id="dddd" * 8,
            )

        assert sender.stats.frames_dropped > 0
        await sender.end_session()

    @pytest.mark.asyncio
    async def test_mark_audio_end_sends_terminal_chunk(self, audio_server: tuple[str, AudioSessionStore]) -> None:
        chunk_url, store = audio_server

        sender = AudioChunkSender(
            url=chunk_url,
            session_id="test-end-marker",
            api_key="test-key",
            batch_interval=0.1,
            max_batch_frames=100,
        )
        await sender.start()

        span_id = "eeee" * 4
        for _ in range(3):
            sender.enqueue(
                _make_frame(),
                kind="user",
                trace_id="aaaa" * 8,
                span_id=span_id,
            )

        sender.mark_audio_end(kind="user", span_id=span_id)
        await asyncio.sleep(0.5)
        await sender.end_session()

        assert sender.stats.errors == 0
        session = store.get_session("test-end-marker")
        assert session is not None
        completed_turns = session.get("completed_turns", 0)
        assert completed_turns >= 1


# ---------------------------------------------------------------------------
# AudioSpanProcessor unit tests
# ---------------------------------------------------------------------------


class TestAudioSpanProcessor:

    def test_dispatches_user_speaking(self) -> None:
        from netra.instrumentation.livekit.processors import (
            AudioSpanProcessor,
            register_audio_hooks,
            unregister_audio_hooks,
        )
        from netra.instrumentation.livekit.wrappers import AudioHookManager

        hooks = AudioHookManager()
        trace_id_int = 0xAAAABBBBCCCCDDDD
        register_audio_hooks(trace_id_int, hooks)

        processor = AudioSpanProcessor()

        span = MagicMock()
        span.name = "user_speaking"
        span_ctx = MagicMock()
        span_ctx.is_valid = True
        span_ctx.trace_id = trace_id_int
        span_ctx.span_id = 0x1234567890ABCDEF
        span.get_span_context.return_value = span_ctx

        processor.on_start(span)
        assert hooks._current_user_span_id == format(0x1234567890ABCDEF, "016x")
        assert hooks._current_user_trace_id == format(trace_id_int, "032x")

        readable = MagicMock()
        readable.name = "user_speaking"
        readable.get_span_context.return_value = span_ctx
        processor.on_end(readable)
        assert hooks._current_user_span_id is None

        unregister_audio_hooks(trace_id_int)

    def test_dispatches_agent_speaking(self) -> None:
        from netra.instrumentation.livekit.processors import (
            AudioSpanProcessor,
            register_audio_hooks,
            unregister_audio_hooks,
        )
        from netra.instrumentation.livekit.wrappers import AudioHookManager

        hooks = AudioHookManager()
        trace_id_int = 0x1111222233334444
        register_audio_hooks(trace_id_int, hooks)

        processor = AudioSpanProcessor()

        span = MagicMock()
        span.name = "agent_speaking"
        span_ctx = MagicMock()
        span_ctx.is_valid = True
        span_ctx.trace_id = trace_id_int
        span_ctx.span_id = 0xABCD
        span.get_span_context.return_value = span_ctx

        processor.on_start(span)
        assert hooks._current_agent_span_id == format(0xABCD, "016x")

        readable = MagicMock()
        readable.name = "agent_speaking"
        readable.get_span_context.return_value = span_ctx
        processor.on_end(readable)
        assert hooks._current_agent_span_id is None

        unregister_audio_hooks(trace_id_int)

    def test_ignores_unrelated_spans(self) -> None:
        from netra.instrumentation.livekit.processors import AudioSpanProcessor

        processor = AudioSpanProcessor()

        span = MagicMock()
        span.name = "llm_request"
        processor.on_start(span)
        span.get_span_context.assert_not_called()


# ---------------------------------------------------------------------------
# AudioHookManager unit tests
# ---------------------------------------------------------------------------


class TestAudioHookManager:

    def test_on_user_frame_enqueues(self) -> None:
        from netra.instrumentation.livekit.wrappers import AudioHookManager

        sender = MagicMock()
        hooks = AudioHookManager(sender=sender)
        hooks._session_trace_id = "aabb" * 8

        frame = _make_frame()
        hooks.on_user_frame(frame)

        sender.enqueue.assert_called_once()
        call_kwargs = sender.enqueue.call_args
        assert call_kwargs.kwargs["kind"] == "user"
        assert call_kwargs.kwargs["trace_id"] == "aabb" * 8

    def test_on_agent_frame_uses_span_id(self) -> None:
        from netra.instrumentation.livekit.wrappers import AudioHookManager

        sender = MagicMock()
        hooks = AudioHookManager(sender=sender)
        hooks._session_trace_id = "ccdd" * 8
        hooks.on_agent_speaking_start("ccdd" * 8, "span1234" * 2)

        frame = _make_frame()
        hooks.on_agent_frame(frame)

        call_kwargs = sender.enqueue.call_args
        assert call_kwargs.kwargs["span_id"] == "span1234" * 2
        assert call_kwargs.kwargs["kind"] == "agent"

    def test_end_all_marks_open_spans(self) -> None:
        from netra.instrumentation.livekit.wrappers import AudioHookManager

        sender = MagicMock()
        hooks = AudioHookManager(sender=sender)
        hooks.on_user_speaking_start("trace1", "span_u")
        hooks.on_agent_speaking_start("trace1", "span_a")

        hooks.end_all()

        assert sender.mark_audio_end.call_count == 2
        calls = [c.kwargs for c in sender.mark_audio_end.call_args_list]
        kinds = {c["kind"] for c in calls}
        assert kinds == {"user", "agent"}
        assert hooks._current_user_span_id is None
        assert hooks._current_agent_span_id is None
