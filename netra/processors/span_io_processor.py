import json
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from netra.instrumentation.message_builder import build_messages, tool_call_entry
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# Failsafe for third-party / legacy instrumentors that still emit indexed
# gen_ai.prompt(s).N.* / gen_ai.completion(s).N.* keys.  Those keys are folded
# into structured ``input``/``output`` JSON and are never written to the span.
_INDEXED_MSG_RE = re.compile(r"^gen_ai\.(prompts?|completions?)\.(\d+)\.(.+)$")
_TOOL_CALL_FIELD_RE = re.compile(r"^tool_calls\.(\d+)\.(.+)$")

_TRACELOOP_PREFIX = "traceloop."
_NETRA_PREFIX = "netra."

_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"

SetAttributeFunc = Callable[[str, Any], None]


def _span_key(context: Any) -> Tuple[int, int]:
    """Identify a span by its trace and span ids (``SpanContext`` itself isn't hashable)."""
    return context.trace_id, context.span_id


def _build_messages(index_map: Dict[int, Dict[str, str]]) -> str:
    """Serialize an index→message dict to a JSON array ordered by index.

    Folded ``tool_calls.J.*`` fields become one tool-call entry each, after the message they belong to.
    """
    entries: List[Dict[str, Any]] = []
    for index in sorted(index_map):
        message: Dict[str, str] = {}
        tool_calls: Dict[int, Dict[str, str]] = {}
        for field, value in index_map[index].items():
            if tool_call_field := _TOOL_CALL_FIELD_RE.match(field):
                tool_calls.setdefault(int(tool_call_field.group(1)), {})[tool_call_field.group(2)] = value
            else:
                message[field] = value
        if "content" in message or not tool_calls:
            entries.append(message)
        for _, call in sorted(tool_calls.items()):
            entries.append(tool_call_entry(call.get("name", ""), call.get("arguments", ""), call.get("id")))
    return build_messages(entries)


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


class SpanIOProcessor(SpanProcessor):  # type: ignore[misc]
    """Normalises ``input`` / ``output`` attributes and remaps ``traceloop.*``
    keys to ``netra.*`` on all spans.

    All interception is done in ``on_start`` via a per-span closure that wraps
    ``span.set_attribute``, following the same pattern as
    ``InstrumentationSpanProcessor``.
    """

    def __init__(self) -> None:
        """Track, per span context, the writer of messages folded from indexed keys until the span ends."""
        self._pending_folds: Dict[Tuple[int, int], Callable[[Any], None]] = {}

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
            self._wrap_set_attribute(span, self._pending_folds)
            self._fold_creation_attributes(span)
        except Exception:
            logger.exception("SpanIOProcessor.on_start failed")

    def on_end(self, span: ReadableSpan) -> None:
        """Write messages folded from indexed keys, then promote ``netra.input``/``netra.output``.

        Called after all instrumentation has written its attributes, so user-set
        values always win. Falls back gracefully if OTel internals change.

        Args:
            span: The span that has ended.
        """
        try:
            build_folded_messages = self._pending_folds.pop(_span_key(span.context), None)
            attrs = getattr(span, "_attributes", None)
            if attrs is None:
                logger.debug(
                    "SpanIOProcessor.on_end: span._attributes not accessible; "
                    "netra.input/netra.output promotion skipped (span_id=%s)",
                    getattr(getattr(span, "context", None), "span_id", "unknown"),
                )
                return
            if not hasattr(attrs, "__setitem__") or not hasattr(attrs, "__delitem__"):
                logger.debug(
                    "SpanIOProcessor.on_end: span._attributes is not mutable (%s); "
                    "netra.input/netra.output promotion skipped",
                    type(attrs).__name__,
                )
                return

            if build_folded_messages:
                build_folded_messages(attrs)

            try:
                user_input = attrs.get("netra.user.input")
                if user_input:
                    attrs["input"] = user_input
                    del attrs["netra.user.input"]
            except Exception:
                logger.warning(
                    "SpanIOProcessor.on_end: could not promote netra.input → input",
                    exc_info=True,
                )

            try:
                user_output = attrs.get("netra.user.output")
                if user_output:
                    attrs["output"] = user_output
                    del attrs["netra.user.output"]
            except Exception:
                logger.warning(
                    "SpanIOProcessor.on_end: could not promote netra.output → output",
                    exc_info=True,
                )
        except Exception:
            logger.exception("SpanIOProcessor.on_end: unexpected error during netra.input/netra.output promotion")

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
    def _fold_creation_attributes(span: Span) -> None:
        """Fold indexed message keys passed at span creation, which bypass ``set_attribute``."""
        attrs = getattr(span, "_attributes", None)
        if not attrs:
            return
        indexed = {key: value for key, value in attrs.items() if _INDEXED_MSG_RE.match(key)}
        for key in indexed:
            del attrs[key]
        if indexed:
            span.set_attributes(indexed)

    @staticmethod
    def _wrap_set_attribute(span: Span, pending_folds: Dict[Tuple[int, int], Callable[[Any], None]]) -> None:
        """Replace ``span.set_attribute`` and ``span.set_attributes`` with normalising closures.

        Per-span accumulators for legacy indexed gen_ai prompts/completions are
        closure-scoped so each span owns its own independent state.  Indexed keys
        are never written; their fields are folded into structured ``input``/
        ``output`` JSON so third-party instrumentors still produce readable
        conversation data without bloating the attribute count.

        ``set_attributes`` (plural) in the OTel SDK writes directly to
        ``_attributes`` without calling ``set_attribute``, so it must also be
        wrapped to ensure every value passes through normalisation.

        The terminal write chains through to the previously-installed
        ``span.set_attribute`` rather than calling the class method directly.
        This preserves the processor chain — for example,
        ``InstrumentationSpanProcessor``'s truncation wrapper is called for
        every value that passes through here.  Recursion is avoided because
        ``InstrumentationSpanProcessor`` uses the class method
        (``type(span).set_attributes``) for *its* terminal write, which
        writes directly to ``_attributes`` without re-entering any instance
        wrapper.

        Args:
            span: The span whose ``set_attribute``/``set_attributes`` will be replaced.
            pending_folds: Where the span's folded-message writer is registered for ``on_end``.
        """
        _prev_set_attribute: SetAttributeFunc = span.set_attribute

        # Per-span accumulators for legacy indexed gen_ai attributes (failsafe only).
        prompts: Dict[int, Dict[str, str]] = {}
        completions: Dict[int, Dict[str, str]] = {}
        gen_ai_owns_input = False
        gen_ai_owns_output = False

        def _is_empty(v: Any) -> bool:
            return v is None or v == ""

        # Folded messages are pending until the span ends, so an owned side counts as set.
        def _input_is_empty() -> bool:
            return not gen_ai_owns_input and _is_empty((span.attributes or {}).get("input"))

        def _output_is_empty() -> bool:
            return not gen_ai_owns_output and _is_empty((span.attributes or {}).get("output"))

        def build_folded_messages(attributes: Any) -> None:
            """Write the folded messages into ``input``/``output`` once, when the span ends."""
            if gen_ai_owns_input and prompts:
                attributes["input"] = _build_messages(prompts)
            if gen_ai_owns_output and completions:
                attributes["output"] = _build_messages(completions)

        def patched_set_attribute(key: str, value: Any) -> None:  # noqa: C901
            nonlocal gen_ai_owns_input, gen_ai_owns_output
            try:
                # 1. Failsafe: fold indexed gen_ai.prompt(s).N.* / gen_ai.completion(s).N.*
                # into structured input/output, built in on_end. Never write the indexed key itself.
                indexed = _INDEXED_MSG_RE.match(key)
                if indexed:
                    pending_folds[_span_key(span.get_span_context())] = build_folded_messages
                    kind, idx_s, field = indexed.group(1), indexed.group(2), indexed.group(3)
                    idx = int(idx_s)
                    if kind.startswith("prompt"):
                        prompts.setdefault(idx, {})[field] = str(value)
                        gen_ai_owns_input = gen_ai_owns_input or _input_is_empty()
                    else:
                        completions.setdefault(idx, {})[field] = str(value)
                        gen_ai_owns_output = gen_ai_owns_output or _output_is_empty()
                    return

                # An explicit input/output written later wins over the folded messages.
                if key == "input":
                    gen_ai_owns_input = False
                elif key == "output":
                    gen_ai_owns_output = False

                # 2. traceloop.entity.input → input  (no traceloop key written)
                if key == "traceloop.entity.input":
                    if _input_is_empty():
                        _prev_set_attribute("input", _extract_traceloop_input(value))
                    return

                # 3. traceloop.entity.output → output  (no traceloop key written)
                if key == "traceloop.entity.output":
                    if _output_is_empty():
                        _prev_set_attribute("output", _extract_traceloop_output(value))
                    return

                # 4. Other traceloop.* → netra.*  (no traceloop key written)
                if key.startswith(_TRACELOOP_PREFIX):
                    new_key = _NETRA_PREFIX + key[len(_TRACELOOP_PREFIX) :]
                    _prev_set_attribute(new_key, value)
                    return

                # 5. gen_ai.usage token aliasing
                if key == _USAGE_INPUT_TOKENS:
                    _prev_set_attribute(_USAGE_PROMPT_TOKENS, value)
                    return

                if key == _USAGE_OUTPUT_TOKENS:
                    _prev_set_attribute(_USAGE_COMPLETION_TOKENS, value)
                    return

                # 6. db.statement → input (DB instrumentations: PyMySQL, etc.)
                # Keep db.statement; never map into output (query results / params are user data).
                if key == "db.statement":
                    _prev_set_attribute(key, value)
                    if _input_is_empty() and not _is_empty(value):
                        _prev_set_attribute("input", value if isinstance(value, str) else str(value))
                    return

                # 7. Everything else — pass through unchanged
                _prev_set_attribute(key, value)

            except Exception:
                logger.debug("SpanIOProcessor: error processing key=%s", key, exc_info=True)
                try:
                    _prev_set_attribute(key, value)
                except Exception:
                    logger.debug("SpanIOProcessor: error calling set_attribute key=%s", key, exc_info=True)

        def patched_set_attributes(attributes: Mapping[str, Any]) -> None:
            if not attributes:
                return
            for key, value in attributes.items():
                patched_set_attribute(key, value)

        setattr(span, "set_attribute", patched_set_attribute)
        setattr(span, "set_attributes", patched_set_attributes)
