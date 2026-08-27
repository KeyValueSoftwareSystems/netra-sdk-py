"""Tells the audio coordinator which turn is being spoken, as spans open and close.

LiveKit brackets each run of speech in a ``user_speaking`` or ``agent_speaking``
span. This processor is the only thing that sees those spans start and end, so
it is what lets a frame captured milliseconds later be filed under the turn it
belongs to.

Registered once for the process, while coordinators are per call — hence the
lookup by the span's trace id in
:data:`~netra.instrumentation.libraries.livekit.audio_capture.audio_coordinators`.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Union

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import SpanContext

from netra.instrumentation.libraries.livekit.audio_capture import SessionAudioCoordinator, audio_coordinators
from netra.instrumentation.libraries.livekit.audio_types import SPEAKING_SPAN_ROLES, SpeakerRole

logger = logging.getLogger(__name__)

_TRACE_ID_HEX_DIGITS = "032x"
_SPAN_ID_HEX_DIGITS = "016x"


class _SpeakingSpan(NamedTuple):
    """A span that delimits speech, resolved to the call it belongs to.

    Attributes:
        role: The speaker the span delimits.
        coordinator: The coordinator capturing that call's audio.
        span_context: The span's own context, for its trace and span ids.
        parent_span_id: Hex id of the speaking span's parent, or ``""`` if none.
    """

    role: SpeakerRole
    coordinator: SessionAudioCoordinator
    span_context: SpanContext
    parent_span_id: str


class AudioSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Opens and closes an audio recording alongside each speaking span."""

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Start attributing this speaker's audio to the span that just opened.

        Args:
            span: The span that was started.
            parent_context: The parent context (unused).
        """
        speaking = _resolve_speaking_span(span)
        if speaking is None:
            return

        speaking.coordinator.on_speaking_start(
            speaking.role,
            trace_id=format(speaking.span_context.trace_id, _TRACE_ID_HEX_DIGITS),
            span_id=format(speaking.span_context.span_id, _SPAN_ID_HEX_DIGITS),
            parent_span_id=speaking.parent_span_id,
        )

    def on_end(self, span: ReadableSpan) -> None:
        """Close the recording for the speaking span that just ended.

        Args:
            span: The span that has ended.
        """
        speaking = _resolve_speaking_span(span)
        if speaking is None:
            return

        speaking.coordinator.on_speaking_end(
            speaking.role,
            span_id=format(speaking.span_context.span_id, _SPAN_ID_HEX_DIGITS),
        )

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush; this processor holds nothing pending.

        Args:
            timeout_millis: Maximum time to wait (unused).

        Returns:
            Always True.
        """
        return True

    def shutdown(self) -> None:
        """No-op shutdown; coordinator teardown belongs to the session wrapper."""


def _resolve_speaking_span(span: Union[Span, ReadableSpan]) -> Optional[_SpeakingSpan]:
    """Identify a speaking span and the call whose audio it delimits.

    Never raises: this runs on every span the process produces, so a failure
    here would be a failure of the user's tracing, not just of audio capture.

    Args:
        span: The span that started or ended.

    Returns:
        The resolved speaking span, or ``None`` when *span* does not delimit
        speech or its call is not capturing audio — the common case by far.
    """
    try:
        role = SPEAKING_SPAN_ROLES.get(span.name or "")
        if role is None:
            return None

        span_context = span.get_span_context()
        if span_context is None or not span_context.is_valid:
            return None

        coordinator = audio_coordinators.get(span_context.trace_id)
        if coordinator is None:
            return None
        return _SpeakingSpan(
            role=role,
            coordinator=coordinator,
            span_context=span_context,
            parent_span_id=_parent_span_id_hex(span),
        )
    except Exception:
        logger.debug("netra.audio: could not resolve a speaking span", exc_info=True)
        return None


def _parent_span_id_hex(span: Union[Span, ReadableSpan]) -> str:
    """Return the hex id of *span*'s parent, or ``""`` when there is none.

    Args:
        span: The speaking span whose parent to read.

    Returns:
        A 16-digit lowercase hex span id, or an empty string for a root span
        or an invalid/missing parent context.
    """
    parent = getattr(span, "parent", None)
    if parent is None:
        return ""
    if hasattr(parent, "is_valid") and not parent.is_valid:
        return ""
    parent_span_id = getattr(parent, "span_id", None)
    if not isinstance(parent_span_id, int) or not parent_span_id:
        return ""
    return format(parent_span_id, _SPAN_ID_HEX_DIGITS)
