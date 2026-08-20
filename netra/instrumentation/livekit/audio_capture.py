"""Captures a LiveKit session's audio and attributes it to speaking spans.

:class:`SessionAudioCoordinator` sits between livekit-agents' audio I/O and
:class:`~netra.instrumentation.livekit.audio_sender.AudioChunkSender`. It owns
two things:

* **where a frame belongs** — the ``user_speaking``/``agent_speaking`` span open
  at the moment of capture, pushed in by
  :class:`~netra.instrumentation.livekit.audio_processor.AudioSpanProcessor`.
  Frames captured between turns are still sent, attributed to the call but to no
  span;
* **what the caller actually heard** — when a caller interrupts the agent,
  LiveKit discards the un-played tail of the utterance, so the coordinator stops
  forwarding agent frames and reports the playback position so the recorded
  audio can be trimmed to match.

Nothing here may change the behaviour of the user's agent: every patched method
forwards to the original whether or not our own work succeeded.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from netra.instrumentation.livekit.audio_sender import (
    _MAX_DRAIN_TIMEOUT_SECONDS,
    AudioChunkSender,
)
from netra.instrumentation.livekit.audio_types import (
    CREDENTIAL_HEADER_NAMES,
    NETRA_AUDIO_CIRCUIT_TRIPPED,
    NETRA_AUDIO_DROPPED_FRAMES,
    NETRA_AUDIO_ERRORS,
    NETRA_AUDIO_SENT_BYTES,
    NETRA_AUDIO_SENT_CHUNKS,
    SpeakerRole,
)

if TYPE_CHECKING:
    from livekit.agents import AgentSession
    from livekit.agents.voice.io import AgentInput, AudioInput, AudioOutput, PlaybackFinishedEvent
    from livekit.rtc import AudioFrame
    from opentelemetry.trace import Span

    from netra.config import Config

logger = logging.getLogger(__name__)

# Nominal PCM bytes in one captured frame: 20ms of 24kHz mono 16-bit audio, what
# livekit-agents delivers by default. Used only to turn the byte budget
# ``NETRA_AUDIO_BUFFER_BYTES`` into the frame count the queue is actually bounded
# by — a different frame size simply makes the queue hold proportionally more or
# less audio than the budget names.
_NOMINAL_FRAME_BYTES = 960

_MILLISECONDS_PER_SECOND = 1000

# Slack added to the wait in ``_close_from_outside`` on top of the drain budget it
# hands the coordinator, so the coordinator's own deadline is the one that fires.
_TEARDOWN_GRACE_SECONDS = 1.0

# How long finish-close will wait for LiveKit's natural ``playback_finished``
# (emitted during ``_aclose_impl`` after we stop forwarding frames) before falling
# back to a wall-clock estimate of how much was heard.
_PLAYBACK_WAIT_ON_CLOSE_SECONDS = 1.5

# Attribute on AgentSession stashing (trace_id, session_span) between the prepare
# and finish halves of audio teardown around LiveKit's ``_aclose_impl``.
_PENDING_AUDIO_CLOSE_ATTR = "_netra_pending_audio_close"


@dataclass(frozen=True)
class _ActiveSpeech:
    """The speaking span currently open for one speaker.

    Attributes:
        span_id: Hex id of the open ``*_speaking`` span.
        trace_id: Hex trace id of the call the span belongs to.
        parent_span_id: Hex id of the speaking span's parent, or ``""`` if none.
    """

    span_id: str
    trace_id: str
    parent_span_id: str = ""


class SessionAudioCoordinator:
    """Routes one AgentSession's audio frames to the sender, tagged by span.

    Lifecycle:

    1. :meth:`attach` patches the session's audio I/O, once ``start()`` has
       returned and ``session.input``/``session.output`` exist;
    2. :class:`AudioSpanProcessor` calls :meth:`on_speaking_start` /
       :meth:`on_speaking_end` as LiveKit opens and closes speaking spans;
    3. the patched I/O calls :meth:`on_frame` for every frame in either
       direction;
    4. :meth:`aclose` closes any span still recording and shuts the sender down.

    Confined to the agent's event loop, like the sender it feeds.
    """

    def __init__(self, *, sender: Optional[AudioChunkSender] = None) -> None:
        """Bind the coordinator to a sender.

        Args:
            sender: Where frames are handed off. ``None`` makes the coordinator
                inert, which is what the span processor's callbacks expect when
                audio capture is off.
        """
        self._sender = sender

        self._active_speech: Dict[SpeakerRole, Optional[_ActiveSpeech]] = {role: None for role in SpeakerRole}
        self._session_trace_id = ""
        self._agent_speech_ended = False

        # The agent span most recently opened, kept after it closes: LiveKit
        # routinely ends the ``agent_speaking`` span *before* it reports the
        # interrupt that cut it short, so the id would otherwise be gone by the
        # time there is something to report about it.
        self._last_agent_span_id = ""
        self._last_agent_parent_span_id = ""
        self._is_agent_interrupted = False
        self._interrupted_agent_span_id = ""
        self._interrupted_agent_parent_span_id = ""

        # Patched audio output — kept so we can observe playout lifecycle.
        self._audio_output: Optional["AudioOutput"] = None
        # Wall-clock start of playout from LiveKit ``playback_started``.
        self._agent_playback_started_at: Optional[float] = None
        # Wall-clock of the first agent frame we captured for the current
        # utterance. Used when ``playback_started`` never reaches us (wrapper
        # chain) so mid-speech session close is still detected and estimated.
        self._agent_capture_started_at: Optional[float] = None
        # Set once interrupt_agent_span has been asked to trim the current
        # utterance, so prepare/finish close do not double-report.
        self._agent_playback_trim_reported = False
        # Completed when an interrupted ``playback_finished`` reports heard_ms,
        # so finish-close can wait for LiveKit's natural interrupt during aclose.
        self._playback_trim_event: Optional[asyncio.Event] = None

    # -- attachment ---------------------------------------------------------

    def attach(self, session: "AgentSession") -> None:
        """Patch *session*'s audio input and output to feed this coordinator.

        Must run after ``session.start()``: before that, ``session.input`` and
        ``session.output`` are not yet populated.

        Args:
            session: The started LiveKit ``AgentSession``.
        """
        self._session_trace_id = _current_trace_id()
        self._patch_audio_input(session)
        self._patch_audio_output(session)

    # -- span callbacks -----------------------------------------------------

    def on_speaking_start(
        self,
        role: SpeakerRole,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str = "",
    ) -> None:
        """Attribute subsequent frames from *role* to a newly opened span.

        When *role* is :attr:`SpeakerRole.USER` and a previous agent speaking
        span has already ended, the stale ``_active_speech[AGENT]`` is cleared.
        This prevents preemptive TTS audio (synthesized during the user's turn
        but never played) from being misattributed to the previous agent span.

        Args:
            role: The speaker whose span opened.
            trace_id: Hex trace id of the span.
            span_id: Hex id of the span.
            parent_span_id: Hex id of the speaking span's parent, or ``""``.
        """
        self._active_speech[role] = _ActiveSpeech(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        if role is SpeakerRole.AGENT:
            self._last_agent_span_id = span_id
            self._last_agent_parent_span_id = parent_span_id
            self._is_agent_interrupted = False
            self._agent_speech_ended = False
            self._interrupted_agent_span_id = ""
            self._interrupted_agent_parent_span_id = ""
            self._agent_playback_started_at = None
            self._agent_capture_started_at = None
            self._agent_playback_trim_reported = False
            self._playback_trim_event = None
        if role is SpeakerRole.USER and self._agent_speech_ended:
            self._active_speech[SpeakerRole.AGENT] = None
        logger.debug(
            "netra.audio: %s speaking started — span_id=%s parent_span_id=%s",
            role.value,
            span_id,
            parent_span_id or "(none)",
        )

    def on_speaking_end(self, role: SpeakerRole, *, span_id: str = "") -> None:
        """Close the recording for *role*'s open span.

        An interrupted agent span is left for :meth:`on_playback_finished` to
        finalize: only the playback report says how much of the utterance was
        heard, and finalizing here would fix the recording at its full length.

        For the agent role, ``_active_speech`` is intentionally **not** cleared
        immediately: LiveKit routinely ends the ``agent_speaking`` span before
        the TTS has finished outputting all frames — or starts and ends it
        within a single event-loop tick for tool-call exit messages. Keeping
        the active speech ensures trailing frames are still attributed to the
        correct span. The next :meth:`on_speaking_start` naturally overwrites
        it, :meth:`on_speaking_start` for the USER role clears it once a new
        user turn begins (so preemptive TTS audio that was never played is
        dropped rather than misattributed), and :meth:`close` forcibly clears
        it at teardown.

        Args:
            role: The speaker whose span closed.
            span_id: Hex id of the specific span that ended. When given, only
                that span's end is signalled to the sender; when omitted the
                currently active span (if any) is used.
        """
        active = self._active_speech[role]

        # Determine which span actually ended.
        ended_span_id = span_id or (active.span_id if active else "")
        ended_parent_span_id = active.parent_span_id if active and active.span_id == ended_span_id else ""
        ended_trace_id = active.trace_id if active and active.span_id == ended_span_id else ""

        # For the agent role, keep _active_speech populated so that trailing
        # TTS frames are still attributed.  For user role, clear immediately.
        if role is not SpeakerRole.AGENT:
            self._active_speech[role] = None
        else:
            self._agent_speech_ended = True

        if not ended_span_id:
            return
        if role is SpeakerRole.AGENT and self._is_agent_interrupted:
            return
        if self._sender is not None:
            self._sender.mark_audio_end(
                role=role,
                span_id=ended_span_id,
                parent_span_id=ended_parent_span_id,
                trace_id=ended_trace_id,
            )

    # -- frame callbacks ----------------------------------------------------

    def on_frame(self, role: SpeakerRole, frame: "AudioFrame") -> None:
        """Hand one captured frame to the sender.

        Stamps the capture time here, the earliest point the frame is seen, so
        the timeline is not skewed by time spent queued.

        Args:
            role: The speaker the frame came from.
            frame: The frame LiveKit just captured.
        """
        if self._sender is None:
            return
        if role is SpeakerRole.AGENT and self._is_agent_interrupted:
            # Produced after the caller cut in, so never played out.
            return
        if role is SpeakerRole.AGENT and self._active_speech[role] is None:
            # Preemptive TTS: synthesized during the user's turn with no
            # agent_speaking span, so never played out — drop silently.
            return

        if role is SpeakerRole.AGENT and self._agent_capture_started_at is None:
            self._agent_capture_started_at = time.time()

        active = self._active_speech[role]
        self._sender.enqueue(
            frame,
            role=role,
            span_id=active.span_id if active is not None else "",
            parent_span_id=active.parent_span_id if active is not None else "",
            trace_id=(active.trace_id if active is not None else "") or self._session_trace_id,
            timestamp_ns=time.time_ns(),
        )

    # -- interrupt callbacks ------------------------------------------------

    def on_output_buffer_cleared(self) -> None:
        """Note that LiveKit dropped the agent's queued audio — a caller interrupt.

        Stops further agent frames from being forwarded and remembers which span
        was cut, for :meth:`on_playback_finished` to trim.
        """
        active = self._active_speech[SpeakerRole.AGENT]
        self._is_agent_interrupted = True
        if active is not None:
            self._interrupted_agent_span_id = active.span_id
            self._interrupted_agent_parent_span_id = active.parent_span_id
        else:
            self._interrupted_agent_span_id = self._last_agent_span_id
            self._interrupted_agent_parent_span_id = self._last_agent_parent_span_id
        logger.debug(
            "netra.audio: agent audio buffer cleared — utterance interrupted (span_id=%s)",
            self._interrupted_agent_span_id,
        )

    def on_playback_started(self, event: Any = None, **kwargs: Any) -> None:
        """Remember when the current agent utterance began playing out.

        Args:
            event: LiveKit ``playback_started`` payload, if emitted as an object.
            **kwargs: Alternate form with ``created_at`` (some LiveKit paths).
        """
        created_at = kwargs.get("created_at")
        if created_at is None and event is not None:
            created_at = getattr(event, "created_at", None)
        self._agent_playback_started_at = float(created_at) if created_at is not None else time.time()

    def on_playback_finished(self, event: "PlaybackFinishedEvent") -> None:
        """Trim an interrupted utterance to the audio that was played out.

        Args:
            event: LiveKit's ``playback_finished`` event. Only an event flagged
                ``interrupted`` is acted on; a normal end of playback needs no
                correction.
        """
        if not getattr(event, "interrupted", False):
            self._agent_playback_started_at = None
            self._agent_capture_started_at = None
            return
        span_id = self._interrupted_agent_span_id or self._last_agent_span_id
        if not span_id or self._sender is None:
            return

        playback_ms = int(getattr(event, "playback_position", 0.0) * _MILLISECONDS_PER_SECOND)
        self._report_agent_playback_trim(
            span_id=span_id,
            playback_ms=playback_ms,
            parent_span_id=self._interrupted_agent_parent_span_id or self._last_agent_parent_span_id,
        )
        self._agent_playback_started_at = None
        self._agent_capture_started_at = None
        logger.debug(
            "netra.audio: interrupted playback finished — span_id=%s heard=%dms",
            span_id,
            playback_ms,
        )

    def _report_agent_playback_trim(self, *, span_id: str, playback_ms: int, parent_span_id: str = "") -> None:
        """Tell the sender how much of an agent utterance was heard. Idempotent."""
        if self._sender is None or not span_id or self._agent_playback_trim_reported:
            return
        self._agent_playback_trim_reported = True
        self._sender.interrupt_agent_span(
            span_id=span_id,
            playback_ms=max(0, playback_ms),
            parent_span_id=parent_span_id,
        )
        if self._playback_trim_event is not None:
            self._playback_trim_event.set()

    def _agent_is_mid_utterance(self) -> bool:
        """True when agent audio may still be playing or buffered unheard."""
        pending_playback = 0
        if self._audio_output is not None:
            pending_playback = int(getattr(self._audio_output, "_pending_playback_count", 0) or 0)
        return (
            pending_playback > 0
            or self._agent_playback_started_at is not None
            or self._agent_capture_started_at is not None
        )

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        """Close every span still recording, without touching the sender.

        Separate from :meth:`aclose` because the session span has to be stamped
        with the sender's final statistics, which means the two teardown halves
        run at different points.

        Unlike the per-event :meth:`on_speaking_end` (which deliberately leaves
        agent active speech in place for trailing frames), this teardown path
        forcibly clears both roles and signals their end to the sender.
        """
        for role in SpeakerRole:
            active = self._active_speech[role]
            self._active_speech[role] = None
            if active is None:
                continue
            if role is SpeakerRole.AGENT and self._is_agent_interrupted:
                continue
            if self._sender is not None:
                self._sender.mark_audio_end(
                    role=role,
                    span_id=active.span_id,
                    parent_span_id=active.parent_span_id,
                    trace_id=active.trace_id,
                )

    async def prepare_close(self) -> None:
        """Stop capturing agent frames before LiveKit tears the session down.

        Does **not** drain the sender. LiveKit emits ``clear_buffer`` /
        ``playback_finished`` (with the real ``playback_position``) only inside
        its own ``_aclose_impl``, which runs *after* this prepare step. Draining
        here would finalize the span before that report arrives.
        """
        if not self._agent_is_mid_utterance() and not (
            self._is_agent_interrupted and not self._agent_playback_trim_reported
        ):
            return

        active = self._active_speech[SpeakerRole.AGENT]
        span_id = (
            active.span_id if active is not None else (self._interrupted_agent_span_id or self._last_agent_span_id)
        )
        parent_span_id = (
            active.parent_span_id
            if active is not None
            else (self._interrupted_agent_parent_span_id or self._last_agent_parent_span_id)
        )
        if not span_id:
            return

        self._is_agent_interrupted = True
        self._interrupted_agent_span_id = span_id
        self._interrupted_agent_parent_span_id = parent_span_id
        if self._playback_trim_event is None:
            self._playback_trim_event = asyncio.Event()
        logger.debug(
            "netra.audio: prepare close mid-agent-speech — span_id=%s (waiting for playback_finished)",
            span_id,
        )

    async def finish_close(self, *, drain_timeout_seconds: Optional[float] = None) -> None:
        """Trim unheard agent audio if needed, then drain the sender.

        Call after LiveKit's ``_aclose_impl`` so ``playback_finished`` has had a
        chance to deliver ``heard_ms``. Falls back to wall-clock if it does not.
        """
        await self._finalize_mid_speech_trim_after_livekit_close()
        self.close()
        if self._sender is None:
            return
        await self._sender.end_session(drain_timeout_seconds=drain_timeout_seconds)

    async def aclose(self, *, drain_timeout_seconds: Optional[float] = None) -> None:
        """Close the open recordings and shut the sender down.

        Prefer :meth:`prepare_close` then :meth:`finish_close` around LiveKit
        session teardown when possible. This combined path is the backstop used
        by ``Netra.shutdown()`` and tests.

        Args:
            drain_timeout_seconds: Explicit total budget for the sender's drain.
                When set, that value is used as the hard deadline. When ``None``
                (the default), the sender computes a dynamic budget from the
                remaining queue depth and open spans.
        """
        await self.prepare_close()
        await self.finish_close(drain_timeout_seconds=drain_timeout_seconds)

    async def _finalize_mid_speech_trim_after_livekit_close(self) -> None:
        """Apply heard_ms for a mid-speech disconnect, waiting briefly if needed."""
        if self._sender is None or self._agent_playback_trim_reported:
            return
        if not self._is_agent_interrupted and not self._agent_is_mid_utterance():
            return

        span_id = self._interrupted_agent_span_id or self._last_agent_span_id
        parent_span_id = self._interrupted_agent_parent_span_id or self._last_agent_parent_span_id
        if not span_id:
            return

        self._is_agent_interrupted = True
        self._interrupted_agent_span_id = span_id
        self._interrupted_agent_parent_span_id = parent_span_id

        if self._playback_trim_event is not None and not self._agent_playback_trim_reported:
            try:
                await asyncio.wait_for(
                    self._playback_trim_event.wait(),
                    timeout=_PLAYBACK_WAIT_ON_CLOSE_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "netra.audio: timed out waiting for playback_finished on close; "
                    "falling back to wall-clock estimate"
                )

        if self._agent_playback_trim_reported:
            return

        playback_ms = self._estimate_playback_ms_from_clock()
        if playback_ms is None:
            playback_ms = 0
        self._report_agent_playback_trim(
            span_id=span_id,
            playback_ms=playback_ms,
            parent_span_id=parent_span_id,
        )
        logger.debug(
            "netra.audio: session closing mid-agent-speech — span_id=%s heard=%dms (estimated)",
            span_id,
            playback_ms,
        )

    def _estimate_playback_ms_from_clock(self) -> Optional[int]:
        """Estimate heard ms from playback_started or first captured agent frame."""
        started_at = self._agent_playback_started_at or self._agent_capture_started_at
        if started_at is None:
            return None
        return max(0, int((time.time() - started_at) * _MILLISECONDS_PER_SECOND))

    @property
    def sender(self) -> Optional[AudioChunkSender]:
        """The sender this coordinator feeds, if audio capture is on."""
        return self._sender

    # -- audio input --------------------------------------------------------

    def _patch_audio_input(self, session: "AgentSession") -> None:
        """Intercept the caller's audio by proxying the session's input stream.

        Args:
            session: The started LiveKit ``AgentSession``.
        """
        session_input = getattr(session, "input", None)
        audio_input = getattr(session_input, "audio", None)
        if session_input is None or audio_input is None:
            logger.warning("netra.audio: session.input.audio is unavailable — caller audio is not captured")
            return

        leaf = _leaf_audio_source(audio_input)
        proxy = _AudioInputProxy(leaf, self)

        for holder, attribute in _proxy_mount_points(session_input, audio_input, leaf):
            if _try_set(holder, attribute, proxy):
                logger.debug("netra.audio: caller audio proxied at %s.%s", type(holder).__name__, attribute)
                return

        # Every mount point is read-only, so there is nowhere to insert a proxy;
        # patch the iteration protocol on the leaf itself instead.
        _patch_anext(leaf, self)

    # -- audio output -------------------------------------------------------

    def _patch_audio_output(self, session: "AgentSession") -> None:
        """Intercept the agent's audio and the events describing its playback.

        Args:
            session: The started LiveKit ``AgentSession``.
        """
        audio_output = getattr(getattr(session, "output", None), "audio", None)
        if audio_output is None:
            logger.warning("netra.audio: session.output.audio is unavailable — agent audio is not captured")
            return

        self._audio_output = audio_output
        self._patch_capture_frame(audio_output)
        self._patch_clear_buffer(audio_output)
        self._subscribe_to_playback_events(audio_output)

    def _patch_capture_frame(self, audio_output: "AudioOutput") -> None:
        """Wrap ``capture_frame`` so every outgoing frame is seen.

        Args:
            audio_output: LiveKit's agent audio output.
        """
        original = audio_output.capture_frame

        @functools.wraps(original)
        async def capture_frame(frame: "AudioFrame") -> Any:
            _run_hook_safely(lambda: self.on_frame(SpeakerRole.AGENT, frame), "agent frame")
            return await original(frame)

        audio_output.capture_frame = capture_frame
        logger.debug("netra.audio: wrapped agent capture_frame")

    def _patch_clear_buffer(self, audio_output: "AudioOutput") -> None:
        """Wrap ``clear_buffer``, LiveKit's signal that the caller interrupted.

        Args:
            audio_output: LiveKit's agent audio output.
        """
        original = getattr(audio_output, "clear_buffer", None)
        if not callable(original):
            logger.debug("netra.audio: no clear_buffer on the audio output — interrupts are not detected")
            return

        @functools.wraps(original)
        def clear_buffer() -> Any:
            _run_hook_safely(self.on_output_buffer_cleared, "clear_buffer")
            return original()

        audio_output.clear_buffer = clear_buffer
        logger.debug("netra.audio: wrapped clear_buffer for interrupt detection")

    def _subscribe_to_playback_events(self, audio_output: "AudioOutput") -> None:
        """Listen for playback start/finish, which say when and how much was heard.

        Args:
            audio_output: LiveKit's agent audio output.
        """
        subscribe = getattr(audio_output, "on", None)
        if not callable(subscribe):
            logger.debug("netra.audio: audio output is not an event emitter — interrupts are not trimmed")
            return
        try:
            subscribe("playback_finished", self.on_playback_finished)
            subscribe("playback_started", self.on_playback_started)
        except (TypeError, ValueError):
            logger.debug("netra.audio: could not subscribe to playback events", exc_info=True)
            return
        logger.debug("netra.audio: subscribed to playback_started and playback_finished")


# ---------------------------------------------------------------------------
# Audio input plumbing
# ---------------------------------------------------------------------------


def _run_hook_safely(action: Callable[[], None], description: str) -> None:
    """Run one of our own hooks without letting it reach the user's agent.

    The one place this package swallows an exception, and deliberately: these
    hooks run inline in the agent's audio path, where a raise would drop the
    caller's audio or kill the playout task. The failure is logged, and losing
    observability is always preferable to breaking the call.

    Args:
        action: The hook to run.
        description: What it was doing, for the log line.
    """
    try:
        action()
    except Exception:
        logger.debug("netra.audio: %s hook failed", description, exc_info=True)


class _AudioInputProxy:
    """Transparent proxy over an async audio iterator, tapping each frame.

    ``__aiter__``/``__anext__`` are defined on the class rather than the
    instance because ``async for`` resolves them on the *type*: an instance
    attribute would simply be ignored.
    """

    def __init__(self, source: "AudioInput", coordinator: SessionAudioCoordinator) -> None:
        """Wrap *source*, reporting each frame it yields to *coordinator*.

        Args:
            source: The audio iterator being proxied.
            coordinator: Where captured frames are reported.
        """
        self._source = source
        self._coordinator = coordinator

    def __aiter__(self) -> "_AudioInputProxy":
        """Return self; the proxy is its own iterator."""
        return self

    async def __anext__(self) -> "AudioFrame":
        """Yield the next frame from the wrapped source, tapping it on the way.

        Returns:
            The frame, untouched.
        """
        frame: "AudioFrame" = await self._source.__anext__()
        _run_hook_safely(lambda: self._coordinator.on_frame(SpeakerRole.USER, frame), "caller frame")
        return frame

    def __getattr__(self, name: str) -> Any:
        """Forward every other attribute to the wrapped source.

        Args:
            name: The attribute being looked up.

        Returns:
            The wrapped source's attribute.
        """
        return getattr(self._source, name)


def _leaf_audio_source(audio_input: "AudioInput") -> "AudioInput":
    """Follow the ``.source`` chain to the object actually producing frames.

    LiveKit stacks audio streams (resamplers, buffers) each holding the next in
    ``.source``. Tapping the innermost one captures the caller's audio before
    any of that processing.

    Args:
        audio_input: The outermost audio input.

    Returns:
        The innermost source, which may be *audio_input* itself.
    """
    current = audio_input
    while getattr(current, "source", None) is not None:
        current = current.source
    return current


def _proxy_mount_points(
    session_input: "AgentInput", audio_input: "AudioInput", leaf: "AudioInput"
) -> List[Tuple[Any, str]]:
    """Return the places a proxy over *leaf* could be installed, best first.

    Args:
        session_input: The session's input container.
        audio_input: The outermost audio input.
        leaf: The innermost audio source.

    Returns:
        ``(holder, attribute)`` pairs to try assigning the proxy to.
    """
    if leaf is audio_input:
        return [(session_input, "audio")]

    parent = _parent_of(audio_input, leaf)
    return [(parent, "source")] if parent is not None else []


def _parent_of(audio_input: "AudioInput", leaf: "AudioInput") -> Optional["AudioInput"]:
    """Return the object whose ``.source`` is *leaf*.

    Args:
        audio_input: The outermost audio input to search from.
        leaf: The innermost audio source.

    Returns:
        The holder of *leaf*, or ``None`` when *leaf* is not in the chain.
    """
    current = audio_input
    while current is not None:
        if getattr(current, "source", None) is leaf:
            return current
        current = getattr(current, "source", None)
    return None


def _try_set(holder: Any, attribute: str, value: Any) -> bool:
    """Assign *attribute* on *holder*, reporting whether it took.

    Args:
        holder: The object to assign on.
        attribute: The attribute name.
        value: The value to assign.

    Returns:
        True on success; False when the attribute is read-only or slotted.
    """
    try:
        setattr(holder, attribute, value)
    except (AttributeError, TypeError):
        return False
    return True


def _patch_anext(leaf: "AudioInput", coordinator: SessionAudioCoordinator) -> None:
    """Tap frames by replacing ``__anext__`` on the leaf instance itself.

    Last resort: it only works for code that calls ``leaf.__anext__()``
    explicitly, since ``async for`` looks the method up on the type.

    Args:
        leaf: The innermost audio source.
        coordinator: Where captured frames are reported.
    """
    original = leaf.__anext__

    @functools.wraps(original)
    async def traced_anext() -> "AudioFrame":
        frame = await original()
        _run_hook_safely(lambda: coordinator.on_frame(SpeakerRole.USER, frame), "caller frame")
        return frame

    if not _try_set(leaf, "__anext__", traced_anext):
        logger.warning("netra.audio: could not intercept the caller audio stream — caller audio is not captured")
        return
    logger.debug("netra.audio: fell back to patching __anext__ on the audio source")


def _current_trace_id() -> str:
    """Return the active span's trace id as hex, or ``""`` when there is none.

    Frames captured between speaking spans still belong to the call, so they are
    attributed to this trace rather than dropped.
    """
    from opentelemetry import context, trace

    span_context = trace.get_current_span(context.get_current()).get_span_context()
    if span_context is None or not span_context.is_valid:
        return ""
    return format(span_context.trace_id, "032x")


# ---------------------------------------------------------------------------
# Per-session registry
# ---------------------------------------------------------------------------


class AudioCoordinatorRegistry:
    """Finds the coordinator for a call, given the trace its spans belong to.

    :class:`AudioSpanProcessor` is registered once for the process but speaking
    spans arrive for every concurrent call, so the span's trace id is what says
    which call's audio a span delimits.

    Locked rather than loop-confined. Most traffic is on the agent's event loop —
    registration from the session wrapper, lookups from span callbacks — but
    ``Netra.shutdown()`` reaches :meth:`pop_all` from whichever thread called it,
    and that has to be atomic against a concurrent :meth:`register` or a call's
    coordinator is dropped on the floor with its audio still queued. Contention is
    a handful of operations per call, so a plain lock costs nothing measurable.
    """

    def __init__(self) -> None:
        """Start with no calls registered."""
        self._by_trace_id: Dict[int, SessionAudioCoordinator] = {}
        self._lock = threading.Lock()

    def register(self, trace_id: int, coordinator: SessionAudioCoordinator) -> None:
        """Record the coordinator capturing audio for a call.

        Args:
            trace_id: The ``agent_session`` span's trace id.
            coordinator: The call's coordinator.
        """
        with self._lock:
            self._by_trace_id[trace_id] = coordinator

    def get(self, trace_id: int) -> Optional[SessionAudioCoordinator]:
        """Return the coordinator for a call, or ``None`` if it is not capturing.

        Args:
            trace_id: The trace id off a speaking span.

        Returns:
            The call's coordinator, if one is registered.
        """
        with self._lock:
            return self._by_trace_id.get(trace_id)

    def unregister(self, trace_id: int) -> Optional[SessionAudioCoordinator]:
        """Remove and return a call's coordinator. Idempotent.

        Args:
            trace_id: The ``agent_session`` span's trace id.

        Returns:
            The coordinator that was registered, if any.
        """
        with self._lock:
            return self._by_trace_id.pop(trace_id, None)

    def pop_all(self) -> List[SessionAudioCoordinator]:
        """Remove and return every registered coordinator.

        Used by ``Netra.shutdown()`` as a backstop for calls whose session never
        closed cleanly. Atomic, so a call registering concurrently is either
        returned here or left registered — never lost between the read and the
        clear.

        Returns:
            The coordinators that were registered.
        """
        with self._lock:
            coordinators = list(self._by_trace_id.values())
            self._by_trace_id.clear()
        return coordinators


audio_coordinators = AudioCoordinatorRegistry()


# ---------------------------------------------------------------------------
# Session wiring
# ---------------------------------------------------------------------------


def build_audio_sender(config: "Config", session_id: str) -> Optional[AudioChunkSender]:
    """Construct the sender for one call from the active Netra config.

    Args:
        config: The active Netra config.
        session_id: The Netra session id for this call.

    Returns:
        A configured, unstarted sender, or ``None`` when no audio endpoint
        resolves — which is the single gate on audio capture.
    """
    url = config.audio_endpoint()
    if not url:
        return None

    credential_headers = {
        name: value for name, value in (config.headers or {}).items() if name.lower() in CREDENTIAL_HEADER_NAMES
    }
    return AudioChunkSender(
        url=url,
        session_id=session_id,
        api_key=config.api_key or "",
        auth_headers=credential_headers,
        flush_at_bytes=config.audio_batch_bytes,
        max_request_bytes=config.audio_max_request_bytes,
        max_queue_frames=max(1, config.audio_buffer_bytes // _NOMINAL_FRAME_BYTES),
    )


async def start_audio_capture(session: "AgentSession", *, config: "Config", session_id: str, trace_id: int) -> None:
    """Begin capturing a started session's call audio.

    Isolated from the caller by design: audio capture failing must never make
    ``AgentSession.start()`` fail, and traces are unaffected either way.

    Args:
        session: The started LiveKit ``AgentSession``.
        config: The active Netra config.
        session_id: The Netra session id for this call.
        trace_id: The ``agent_session`` span's trace id, under which the
            coordinator is registered for the span processor to find.
    """
    try:
        sender = build_audio_sender(config, session_id)
        if sender is None:
            return

        coordinator = SessionAudioCoordinator(sender=sender)
        await sender.start()

        # Registered before attaching, not after: from here on the sender owns a
        # background task and an HTTP client, and the registry is the only handle
        # anything has for closing them. ``attach`` patches third-party objects
        # that may refuse assignment, so it is exactly the step that can raise —
        # and a raise between start() and register() would strand both resources
        # for the life of the process. ``attach`` does not need the registry.
        audio_coordinators.register(trace_id, coordinator)
        try:
            coordinator.attach(session)
        except Exception:
            await stop_audio_capture(trace_id)
            raise
        logger.debug("netra.audio: capture attached for trace_id=%032x", trace_id)
    except Exception:
        logger.warning("netra.livekit: audio capture setup failed; the call is traced without audio", exc_info=True)


async def prepare_audio_capture_close(trace_id: int) -> None:
    """Stop forwarding agent frames before LiveKit closes the session.

    Leaves the coordinator registered so ``playback_finished`` during LiveKit's
    ``_aclose_impl`` can still trim with the real ``heard_ms``.
    """
    coordinator = audio_coordinators.get(trace_id)
    if coordinator is None:
        return
    try:
        await coordinator.prepare_close()
    except Exception:
        logger.warning("netra.audio: audio capture prepare-close failed", exc_info=True)


async def finish_audio_capture_close(trace_id: int, session_span: Optional["Span"] = None) -> None:
    """Drain the sender after LiveKit has had a chance to report playback position."""
    coordinator = audio_coordinators.unregister(trace_id)
    if coordinator is None:
        return

    try:
        await coordinator.finish_close()
    except Exception:
        logger.warning("netra.audio: audio capture finish-close failed", exc_info=True)

    sender = coordinator.sender
    if session_span is not None and sender is not None:
        _stamp_audio_stats(session_span, sender)


async def stop_audio_capture(trace_id: int, session_span: Optional["Span"] = None) -> None:
    """Stop capturing a call's audio and record what was delivered.

    Idempotent: a call whose coordinator has already been removed does nothing.
    Combined prepare+finish; session wrappers prefer the split helpers so
    LiveKit can emit ``playback_finished`` between them.

    Args:
        trace_id: The ``agent_session`` span's trace id.
        session_span: The still-recording ``agent_session`` span, stamped with
            the delivery statistics when given.
    """
    coordinator = audio_coordinators.unregister(trace_id)
    if coordinator is None:
        return

    try:
        await coordinator.aclose()
    except Exception:
        logger.warning("netra.audio: audio capture teardown failed", exc_info=True)

    sender = coordinator.sender
    if session_span is not None and sender is not None:
        _stamp_audio_stats(session_span, sender)


def close_all_audio_capture(timeout_seconds: Optional[float] = None) -> None:
    """Shut down every call still capturing audio. Backstop for ``Netra.shutdown()``.

    A sender's queue and task belong to the event loop its call was running on,
    so it cannot simply be awaited from wherever shutdown happens to be called.
    Each one is driven through its own loop instead — and a call whose loop is
    already gone is reported rather than silently skipped, because its unsent
    audio is genuinely lost.

    Args:
        timeout_seconds: Explicit drain budget for each call, forwarded as the
            sender's ``drain_timeout_seconds``. When set, that hard deadline is
            used. When ``None`` (the default), each sender computes a dynamic
            budget from its remaining queue depth and open spans. The outer wait
            when driving another loop is then capped at the sender's maximum
            dynamic budget plus a small grace.
    """
    coordinators = audio_coordinators.pop_all()
    if not coordinators:
        return

    logger.info("netra.audio: shutting down %d call(s) still capturing audio", len(coordinators))
    try:
        current_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    for coordinator in coordinators:
        _close_from_outside(coordinator, current_loop, timeout_seconds)


def _close_from_outside(
    coordinator: SessionAudioCoordinator,
    current_loop: Optional["asyncio.AbstractEventLoop"],
    timeout_seconds: Optional[float],
) -> None:
    """Drive one coordinator's teardown from whichever loop is available.

    Args:
        coordinator: The coordinator to shut down.
        current_loop: The loop the caller is running on, if any.
        timeout_seconds: Explicit drain budget handed to the coordinator, or
            ``None`` to let the sender pick a dynamic budget. Also bounds how
            long this function waits when driving another loop.
    """
    sender = coordinator.sender
    target_loop = sender.loop if sender is not None else None

    if target_loop is None or target_loop.is_closed():
        logger.warning("netra.audio: a call's event loop is gone; its unsent audio is lost")
        return

    if target_loop is current_loop:
        # Cannot block the loop we are on, so this is scheduled and not awaited:
        # whether it finishes depends on the caller keeping the loop alive, which
        # a synchronous shutdown() cannot promise. Said plainly rather than left
        # looking like a completed teardown.
        target_loop.create_task(coordinator.aclose(drain_timeout_seconds=timeout_seconds))
        logger.warning(
            "netra.audio: shutdown was called from a call's own event loop; its drain is scheduled "
            "but cannot be awaited. Await AgentSession.aclose() before Netra.shutdown() to be sure "
            "the audio is delivered"
        )
        return

    future = asyncio.run_coroutine_threadsafe(coordinator.aclose(drain_timeout_seconds=timeout_seconds), target_loop)
    # A shade past the inner budget, so the coordinator's own deadline is what
    # gives up and it still gets to log its statistics. When the budget is
    # dynamic, bound the outer wait by the sender's maximum dynamic timeout.
    outer_timeout = (
        timeout_seconds if timeout_seconds is not None else _MAX_DRAIN_TIMEOUT_SECONDS
    ) + _TEARDOWN_GRACE_SECONDS
    try:
        future.result(timeout=outer_timeout)
    except FuturesTimeoutError:
        logger.warning(
            "netra.audio: a call did not finish sending within %.0fs",
            outer_timeout - _TEARDOWN_GRACE_SECONDS,
        )
    except Exception:
        logger.warning("netra.audio: a call failed to shut down cleanly", exc_info=True)


def _stamp_audio_stats(session_span: "Span", sender: AudioChunkSender) -> None:
    """Record the call's audio delivery counters on its session span.

    Args:
        session_span: The still-recording ``agent_session`` span.
        sender: The sender whose statistics to record.
    """
    stats = sender.stats
    try:
        session_span.set_attributes(
            {
                NETRA_AUDIO_SENT_BYTES: stats.bytes_sent,
                NETRA_AUDIO_SENT_CHUNKS: stats.chunks_sent,
                NETRA_AUDIO_DROPPED_FRAMES: stats.frames_dropped,
                NETRA_AUDIO_ERRORS: stats.errors,
                NETRA_AUDIO_CIRCUIT_TRIPPED: stats.circuit_tripped,
            }
        )
    except Exception:
        logger.debug("netra.audio: could not stamp audio stats on the session span", exc_info=True)
