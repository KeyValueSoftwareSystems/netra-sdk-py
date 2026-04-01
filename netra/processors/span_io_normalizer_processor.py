import json
import logging
import re
from typing import Any, Callable, Dict, Optional

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

logger = logging.getLogger(__name__)

# Patterns for gen_ai indexed attributes
_PROMPT_RE = re.compile(r"^gen_ai\.prompts?\.(\d+)\.(role|content)$")
_COMPLETION_RE = re.compile(r"^gen_ai\.completions?\.(\d+)\.(role|content)$")

_TRACELOOP_PREFIX = "traceloop."
_NETRA_PREFIX = "netra."

SetAttributeFunc = Callable[[str, Any], None]


def _build_messages(index_map: Dict[int, Dict[str, str]]) -> str:
    """Serialize an index→message dict to a JSON array ordered by index.

    Args:
        index_map: Mapping of integer index to partial message dict.

    Returns:
        JSON string of the ordered message list.
    """
    return json.dumps([index_map[i] for i in sorted(index_map)])


def _extract_traceloop_input(raw: Any) -> str:
    """Extract the ``inputs`` payload from a traceloop entity input value.

    Traceloop serialises entity inputs as:
        '{"inputs": {...}, "tags": [...], "metadata": {...}, "kwargs": {...}}'

    We surface only the ``inputs`` dict as the canonical ``input`` attribute.
    If parsing fails the raw value is returned as-is.

    Args:
        raw: The raw attribute value (expected to be a JSON string).

    Returns:
        Serialized string of the inputs payload.
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        payload = parsed.get("inputs", parsed)
        return json.dumps(payload) if not isinstance(payload, str) else payload
    except Exception:
        return str(raw)


def _extract_traceloop_output(raw: Any) -> str:
    """Extract the ``outputs`` payload from a traceloop entity output value.

    Traceloop serialises entity outputs as:
        '{"outputs": {...}, "kwargs": {...}}'

    We surface only the ``outputs`` value as the canonical ``output`` attribute.
    If parsing fails the raw value is returned as-is.

    Args:
        raw: The raw attribute value (expected to be a JSON string).

    Returns:
        Serialized string of the outputs payload.
    """
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        payload = parsed.get("outputs", parsed)

        return json.dumps(payload) if not isinstance(payload, str) else payload
    except Exception:
        return str(raw)


class SpanIONormalizerProcessor(SpanProcessor):  # type: ignore[misc]
    """Normalises ``input`` / ``output`` attributes and remaps ``traceloop.*``
    keys to ``netra.*`` on all spans.

    All interception is done in ``on_start`` via a per-span closure that wraps
    ``span.set_attribute``, following the same pattern as
    ``InstrumentationSpanProcessor``.  No shared mutable state is used.
    """

    def on_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context] = None,
    ) -> None:
        """Wrap the span's ``set_attribute`` to intercept and normalise writes.

        Args:
            span: The span that was started.
            parent_context: The parent context (unused).
        """
        try:
            attrs = span.attributes or {}
            if "input" not in attrs:
                span.set_attribute("input", "")
            if "output" not in attrs:
                span.set_attribute("output", "")
            self._wrap_set_attribute(span)
        except Exception:
            logger.exception("SpanIONormalizerProcessor.on_start failed")

    def on_end(self, span: ReadableSpan) -> None:
        """No-op — all work is done live in the set_attribute closure."""

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
    def _wrap_set_attribute(span: Span) -> None:
        """Replace ``span.set_attribute`` with a normalising closure.

        Per-span accumulators for gen_ai prompts/completions are closure-scoped
        so each span owns its own independent state.

        Args:
            span: The span whose ``set_attribute`` will be replaced.
        """
        original: SetAttributeFunc = span.set_attribute

        # Per-span accumulators for gen_ai indexed attributes
        prompts: Dict[int, Dict[str, str]] = {}
        completions: Dict[int, Dict[str, str]] = {}

        def patched_set_attribute(key: str, value: Any) -> None:  # noqa: C901
            try:
                # 1. gen_ai.prompts.* / gen_ai.prompt.* → keep original + update input
                prompt_match = _PROMPT_RE.match(key)
                if prompt_match:
                    original(key, value)
                    idx = int(prompt_match.group(1))
                    field = prompt_match.group(2)
                    prompts.setdefault(idx, {})[field] = str(value)
                    original("input", _build_messages(prompts))
                    return

                # 2. gen_ai.completions.* / gen_ai.completion.* → keep original + update output
                completion_match = _COMPLETION_RE.match(key)
                if completion_match:
                    original(key, value)
                    idx = int(completion_match.group(1))
                    field = completion_match.group(2)
                    completions.setdefault(idx, {})[field] = str(value)
                    original("output", _build_messages(completions))
                    return

                # 3. traceloop.entity.input → input  (no traceloop key written)
                if key == "traceloop.entity.input":
                    original("input", _extract_traceloop_input(value))
                    return

                # 4. traceloop.entity.output → output  (no traceloop key written)
                if key == "traceloop.entity.output":
                    original("output", _extract_traceloop_output(value))
                    return

                # 5. Other traceloop.* → netra.*  (no traceloop key written)
                if key.startswith(_TRACELOOP_PREFIX):
                    new_key = _NETRA_PREFIX + key[len(_TRACELOOP_PREFIX) :]
                    original(new_key, value)
                    return

                # 6. Everything else — pass through unchanged
                original(key, value)

            except Exception:
                logger.debug("SpanIONormalizerProcessor: error processing key=%s", key, exc_info=True)
                try:
                    original(key, value)
                except Exception:
                    pass

        setattr(span, "set_attribute", patched_set_attribute)
