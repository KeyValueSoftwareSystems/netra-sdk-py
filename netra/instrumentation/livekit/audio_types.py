"""Vocabulary shared by every part of the LiveKit call-audio pipeline.

Three kinds of thing live here, and nothing else:

* :class:`SpeakerRole` — the two speakers a frame can belong to, as an enum
  rather than the string ``"user"``/``"agent"`` that used to be threaded through
  the sender, the coordinator and the span processor independently;
* the PCM format constants and the one arithmetic helper that converts a
  playback duration to a byte offset;
* the wire contract — the ``x-audio-*`` request headers the ingest endpoint
  reads, and the ``netra.audio.*`` span attributes the session root is stamped
  with.

Free of OTel and LiveKit imports, so the wire contract can be asserted against
in tests without a tracer or a livekit-agents install.
"""

from enum import Enum
from typing import Dict

# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------


class SpeakerRole(str, Enum):
    """Which side of the call a run of audio came from.

    A ``str`` enum because the value is also the wire value of the
    ``x-audio-role`` header, so the two cannot drift apart.
    """

    USER = "user"
    AGENT = "agent"


# LiveKit span name -> the speaker whose audio that span delimits. The audio for
# a call is addressed by these spans' ids, so a frame arriving while one is open
# is attributed to it and a frame arriving between them is attributed to nobody
# (see ``SessionAudioCoordinator``).
SPEAKING_SPAN_ROLES: Dict[str, SpeakerRole] = {
    "user_speaking": SpeakerRole.USER,
    "agent_speaking": SpeakerRole.AGENT,
}


# ---------------------------------------------------------------------------
# PCM format
# ---------------------------------------------------------------------------

# What the ingest endpoint is told when a frame reported no format of its own —
# the terminal empty chunk of a span, which carries no frame to read it from.
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CHANNEL_COUNT = 1

# The body is always signed 16-bit little-endian PCM. Not negotiable per chunk:
# the ingest endpoint reads it off the header only so a future format change can
# be rolled out without breaking stored audio.
PCM_BIT_DEPTH = 16
PCM_BYTES_PER_SAMPLE = PCM_BIT_DEPTH // 8

_MILLISECONDS_PER_SECOND = 1000


def pcm_byte_offset_at(*, playback_ms: int, sample_rate_hz: int, channel_count: int) -> int:
    """Return the PCM byte offset *playback_ms* into a stream, on a frame boundary.

    Used to trim an interrupted agent utterance down to the audio the caller
    actually heard. The result is rounded *down* to a whole sample frame:
    cutting mid-sample would leave the stored audio one byte out of phase for
    its whole remaining length.

    Args:
        playback_ms: Milliseconds of audio played out. A non-positive value means
            nothing was heard and yields 0.
        sample_rate_hz: Samples per second, per channel. Must be positive.
        channel_count: Number of interleaved channels. Must be positive.

    Returns:
        The byte offset, never negative and always a multiple of the frame size.

    Raises:
        ValueError: If the PCM format is not playable. Callers substitute
            :data:`DEFAULT_SAMPLE_RATE_HZ` / :data:`DEFAULT_CHANNEL_COUNT` for a
            frame that reported neither, so reaching this is a programming error
            rather than bad input.
    """
    if sample_rate_hz <= 0 or channel_count <= 0:
        raise ValueError(f"unplayable PCM format: sample_rate_hz={sample_rate_hz} channel_count={channel_count}")
    if playback_ms <= 0:
        return 0

    frame_size = channel_count * PCM_BYTES_PER_SAMPLE
    bytes_per_ms = sample_rate_hz * frame_size / _MILLISECONDS_PER_SECOND
    return int(playback_ms * bytes_per_ms) // frame_size * frame_size


# ---------------------------------------------------------------------------
# Wire contract: request headers
# ---------------------------------------------------------------------------

HEADER_CONTENT_TYPE = "Content-Type"
CONTENT_TYPE_PCM = "application/octet-stream"

HEADER_API_KEY = "x-api-key"

HEADER_SESSION_ID = "x-audio-session-id"
HEADER_TRACE_ID = "x-audio-trace-id"
HEADER_SPAN_ID = "x-audio-span-id"
HEADER_ROLE = "x-audio-role"
HEADER_SAMPLE_RATE = "x-audio-sample-rate"
HEADER_CHANNELS = "x-audio-channels"
HEADER_BIT_DEPTH = "x-audio-bit-depth"

# Epoch milliseconds at which the first frame of this chunk was captured.
HEADER_START_MS = "x-audio-start-ms"

# 0-based and monotonic *per span* — a chunk's position in that span's stream,
# not a count of what arrived. Two properties follow, and the endpoint depends on
# both:
#
# * the retries of a single chunk all carry the same number and the same bytes,
#   so the endpoint can treat them as idempotent;
# * a chunk the sender gave up on still consumes its number, so a gap in the
#   sequence is the endpoint's signal that audio was lost — never a number
#   reused for different bytes.
HEADER_SEQUENCE = "x-audio-seq"

# Present on the final chunk of a span, and on that chunk only.
HEADER_LAST_CHUNK = "x-audio-last"

# Only on the final chunk of an *interrupted* agent span: how many milliseconds
# of the utterance the caller heard before cutting in.
HEADER_HEARD_MS = "x-audio-heard-ms"

# Present on the bodyless request that closes the session.
HEADER_SESSION_LAST = "x-audio-session-last"

HEADER_VALUE_TRUE = "true"

# The request headers a Netra config may contribute as an audio-ingest
# credential. Lower-cased for comparison against user-supplied header names.
CREDENTIAL_HEADER_NAMES = frozenset({"x-api-key", "authorization"})


# ---------------------------------------------------------------------------
# Wire contract: span attributes
# ---------------------------------------------------------------------------

# Stamped on the ``agent_session`` span as it closes, so a trace shows what the
# audio pipeline actually managed to deliver for that call.
NETRA_AUDIO_SENT_BYTES = "netra.audio.sent_bytes"
NETRA_AUDIO_SENT_CHUNKS = "netra.audio.sent_chunks"
NETRA_AUDIO_DROPPED_FRAMES = "netra.audio.dropped_frames"
NETRA_AUDIO_ERRORS = "netra.audio.errors"
NETRA_AUDIO_CIRCUIT_TRIPPED = "netra.audio.circuit_tripped"
