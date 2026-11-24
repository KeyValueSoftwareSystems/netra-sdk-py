"""Strip recorded content from spans when ``trace_content`` is off.

Turning ``trace_content`` off is a promise that prompts, completions, function
arguments and return values, and HTTP bodies never leave the process. Each
instrumentor is expected to skip writing content itself, but that check has to
be remembered in every code path of every instrumentor. This processor is the
backstop: it removes content attributes from every span before export, whichever
code wrote them.

It is only installed when ``trace_content`` is off (see ``tracer.py``), and must
be registered before the exporting span processor, because every processor's
``on_end`` receives the same ``ReadableSpan`` and the exporter sees whatever is
left on it.

A span started while content capture was force-enabled for its context
(``override_enable_content_tracing``) is left untouched, matching how the
Traceloop instrumentors treat that override.
"""

import threading
from typing import Any, List, Mapping, Optional, Set, Tuple

from opentelemetry import context as otel_context
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.util import BoundedList

from netra.content_capture import is_content_override_active
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# Attribute keys whose value is content. Covers the generic keys every Netra
# instrumentor and the decorators use, the OpenTelemetry GenAI semantic
# conventions, Traceloop's entity keys, and the instrumentor-specific keys that
# hold user or model text.
CONTENT_ATTRIBUTE_KEYS = frozenset(
    {
        # Netra generic (decorators, HTTP, most instrumentors)
        "input",
        "output",
        # Netra public APIs: Netra.set_input/set_output (normally promoted to
        # input/output by SpanIOProcessor, listed in case promotion is skipped),
        # Netra.add_conversation, and SpanWrapper.set_prompt/set_negative_prompt
        "netra.user.input",
        "netra.user.output",
        "conversation",
        "netra.prompt",
        "netra.negative_prompt",
        # Traceloop decorators
        "traceloop.entity.input",
        "traceloop.entity.output",
        # OpenTelemetry GenAI semantic conventions
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        # Instrumentor-specific
        "gen_ai.request.input",  # dspy
        "gen_ai.response.output",  # dspy
        "gen_ai.request.previous_text",  # elevenlabs
        "gen_ai.request.next_text",  # elevenlabs
        "gen_ai.response.text",  # cartesia
        "gen_ai.response.summary.text",  # deepgram
        "gen_ai.response.summary.short",  # deepgram
        "gen_ai.response.summary.result",  # deepgram
        "gen_ai.response.topics",  # deepgram
        "gen_ai.response.intents",  # deepgram
        "deepgram.agent.history.content",
        "deepgram.agent.conversation.content",
        "deepgram.agent.thinking.content",
        "deepgram.agent.injection_refused.message",
        "gen_ai.agent.instructions",  # agno
        "gen_ai.memory.query.text",  # agno
        "gen_ai.agno.memory.input",  # agno
        "gen_ai.retrieval.query.text",  # agno
        "gen_ai.claude_code.structured_output",  # claude_agent_sdk
        "pydantic_ai.user_prompt",
        "pydantic_ai.instructions",
        "pydantic_ai.final_output",
        # LiveKit (written by livekit-agents itself; lk.pii.* is covered by prefix)
        "lk.chat_ctx",
        "lk.instructions",
        "lk.user_input",
        "lk.input_text",
        "lk.user_transcript",
        "lk.response.text",
        "lk.function_tool.arguments",
        "lk.function_tool.output",
    }
)

# Key prefixes whose every attribute is content (indexed prompt/completion keys,
# and LiveKit's attributes it marks as PII itself).
CONTENT_ATTRIBUTE_PREFIXES: Tuple[str, ...] = (
    "gen_ai.prompt.",
    "gen_ai.prompts.",
    "gen_ai.completion.",
    "gen_ai.completions.",
    "llm.prompts.",
    "llm.completions.",
    "lk.pii.",
    "retrieval.documents.",  # dspy retrieved passages
    "gen_ai.embeddings.",  # dspy embedded texts
)

# Under the indexed prefixes, these trailing fields describe a message without
# carrying its text, so they are kept.
_METADATA_SUFFIXES: Tuple[str, ...] = (".role", ".finish_reason")


def is_content_attribute(key: str) -> bool:
    """Return True if the attribute *key* holds content that ``trace_content=False`` must not export."""
    if key in CONTENT_ATTRIBUTE_KEYS:
        return True
    if key.startswith(CONTENT_ATTRIBUTE_PREFIXES):
        return key.startswith("lk.pii.") or not key.endswith(_METADATA_SUFFIXES)
    return False


def _content_keys(attributes: Optional[Mapping[str, Any]]) -> List[str]:
    """Return the keys of *attributes* that hold content."""
    return [key for key in attributes or () if is_content_attribute(key)]


def _strip_span_attributes(span: ReadableSpan) -> None:
    """Delete content keys from the span's attributes in place.

    Deleting keeps the span's ``BoundedAttributes`` (and so its limits and
    ``dropped_attributes`` count), which swapping in a new dict would lose. A
    span whose attributes are immutable falls back to a filtered copy.
    """
    attributes = getattr(span, "_attributes", None)
    content_keys = _content_keys(attributes)
    if not content_keys:
        return
    try:
        for key in content_keys:
            del attributes[key]  # type: ignore[union-attr]
    except TypeError:
        span._attributes = {key: value for key, value in attributes.items() if key not in content_keys}  # type: ignore[union-attr]


def _strip_event(event: Event) -> Event:
    """Return *event* without content attributes, keeping its attribute limits and dropped count."""
    attributes = event.attributes
    content_keys = _content_keys(attributes)
    if not content_keys:
        return event
    kept = {key: value for key, value in attributes.items() if key not in content_keys}
    if isinstance(attributes, BoundedAttributes):
        bounded = BoundedAttributes(attributes.maxlen, kept, immutable=True, max_value_len=attributes.max_value_len)
        bounded.dropped = attributes.dropped
        return Event(name=event.name, attributes=bounded, timestamp=event.timestamp)
    return Event(name=event.name, attributes=kept, timestamp=event.timestamp)


def _strip_span_events(span: ReadableSpan) -> None:
    """Replace the span's events with content-free copies, keeping its ``dropped_events`` count."""
    events = getattr(span, "_events", None)
    if not events or not any(_content_keys(event.attributes) for event in events):
        return
    stripped = [_strip_event(event) for event in events]
    if isinstance(events, BoundedList):
        rebuilt = BoundedList.from_seq(events._dq.maxlen, stripped)
        rebuilt.dropped = events.dropped
        span._events = rebuilt
    else:
        span._events = tuple(stripped)


class ContentGatingSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Remove content attributes from spans before export while ``trace_content`` is off."""

    def __init__(self) -> None:
        self._override_span_ids: Set[Tuple[int, int]] = set()
        self._lock = threading.Lock()

    def on_start(self, span: Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Remember spans started while content capture was force-enabled for their context.

        The override lives in the context, which is gone by ``on_end``, so it is
        captured here.
        """
        try:
            if is_content_override_active(parent_context) or is_content_override_active():
                span_context = span.get_span_context()
                with self._lock:
                    self._override_span_ids.add((span_context.trace_id, span_context.span_id))
        except Exception:
            logger.debug("ContentGatingSpanProcessor: failed to inspect content override", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        """Strip content attributes from the span and its events before the exporter sees it."""
        try:
            span_context = span.get_span_context()
            key = (span_context.trace_id, span_context.span_id)
            with self._lock:
                if key in self._override_span_ids:
                    self._override_span_ids.discard(key)
                    return

            _strip_span_attributes(span)
            _strip_span_events(span)
        except Exception:
            # Failing open would export the very content this processor exists to
            # withhold, so drop every attribute rather than risk it.
            logger.exception("ContentGatingSpanProcessor: failed to strip content; dropping span attributes")
            try:
                span._attributes = {}
                span._events = ()
            except Exception:
                logger.debug("ContentGatingSpanProcessor: could not clear span attributes", exc_info=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op; this processor holds no spans."""
        return True

    def shutdown(self) -> None:
        """No-op; this processor holds no resources."""
