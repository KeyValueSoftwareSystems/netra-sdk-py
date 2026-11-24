"""Attribute extraction for TypeSafe ``system_one`` calls.

``system_one`` does not take chat messages: it evaluates a ``state`` (text, an
asset, or JSON) against a mapping of named ``questions`` and returns one answer
per question.  Like the OpenAI Decisions API instrumentation, the call's content
is recorded only as the span's ``input`` and ``output`` attributes:

* ``input``: a JSON array of ``{"role", "content"}`` entries -- the state as a
  ``user`` entry, then a ``user`` entry whose ``content`` is the ``questions``
  object exactly as sent, keyed by question name.
* ``output``: a JSON array holding one ``assistant`` entry whose ``content`` is
  the ``answers`` object exactly as returned, keyed by question name.

Neither is recorded when content tracing is off, and both are kept within the
configured attribute length while staying valid JSON.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, SpanAttributes
from opentelemetry.trace import Span

from netra.config import Config, get_attribute_max_len
from netra.content_capture import is_content_capture_enabled
from netra.netra_logger import get_logger
from netra.span_wrapper import SpanType

logger = get_logger(__name__)

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_TYPESAFE = "TypeSafe"
NETRA_SPAN_TYPE_ATTR = f"{Config.LIBRARY_NAME}.span.type"
INPUT_ATTRIBUTE = "input"
OUTPUT_ATTRIBUTE = "output"
ERROR_TYPE_ATTR = "error.type"

_STATE_ROLE = "user"
_QUESTIONS_ROLE = "user"
_ANSWERS_ROLE = "assistant"
_JSON_ITEM_SEPARATOR = ", "
# The questions entry may use at most 1/N of the attribute, so the state they
# are asked about always keeps room however many questions there are.
_QUESTIONS_MAX_SHARE = 2
# A lone surrogate (half of a UTF-16 pair, e.g. from ``surrogateescape``-decoded
# input) cannot be encoded as UTF-8, so the OTLP exporter would fail on it.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed via OTel context keys."""
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def should_send_prompts() -> bool:
    """Whether call content may be recorded (``trace_content``, or a per-context override)."""
    return is_content_capture_enabled()


# --------------------------------------------------------------------------------------
# JSON encoding within a length budget
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Make SDK values JSON-serialisable.

    Question objects are pydantic models, and ``Image`` assets stringify to the
    opaque reference the SDK uses for them, so image bytes never reach the span.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return str(value)


def _dumps(value: Any) -> str:
    """Encode ``value`` as JSON, keeping non-ASCII text as-is.

    ``ensure_ascii=False`` keeps e.g. ``é`` as one character instead of the
    six-character ``\\u00e9``, so non-ASCII content is not cut far sooner than
    ASCII content and stays readable.  Lone surrogates, which that would let
    through and which cannot be encoded as UTF-8, are replaced by U+FFFD; the
    swap is one character for one, so encoded lengths are unchanged.
    """
    return _LONE_SURROGATE.sub("\ufffd", json.dumps(value, ensure_ascii=False, default=_json_default))


def _join_json_array(encoded_items: List[str]) -> str:
    """Join already-encoded JSON values into a JSON array, formatted as ``json.dumps`` would."""
    return "[" + _JSON_ITEM_SEPARATOR.join(encoded_items) + "]"


def _text_entry_within(role: str, content: str, max_len: int) -> Optional[str]:
    """Encode ``{"role", "content"}`` in at most ``max_len`` characters, shortening ``content`` to fit.

    Escaping makes encoded text longer than the raw text, so the content is cut
    in proportion to how far over the limit the encoding is and re-encoded.
    Content is first cut to ``max_len`` characters, which it can never exceed
    once encoded, so a multi-megabyte state is never encoded in full.

    Returns:
        The encoded entry, or None when not even an empty ``content`` fits.
    """
    overhead = len(_dumps({"role": role, "content": ""}))
    budget = max_len - overhead
    if budget < 0:
        return None
    content = content[:budget]
    while True:
        encoded = _dumps({"role": role, "content": content})
        encoded_len = len(encoded) - overhead
        if encoded_len <= budget:
            return encoded
        keep = len(content) * budget // encoded_len
        content = content[: min(keep, len(content) - 1)]


def _mapping_entry_within(role: str, mapping: Mapping[Any, Any], max_len: int) -> Optional[str]:
    """Encode ``{"role": role, "content": "<JSON object of mapping>"}`` in at most ``max_len`` characters.

    The object's JSON is escaped again inside ``content``, so each member is
    sized as such. Only the leading members that fit are kept, whole, so
    ``content`` stays valid JSON.

    Returns:
        The encoded entry, or None when the mapping is empty or not even its first member fits.
    """
    available = max_len - len(_dumps({"role": role, "content": "{}"}))
    kept: List[str] = []
    for key, value in mapping.items():
        encoded = _dumps(str(key)) + ": " + _dumps(value)
        escaped_len = len(_dumps(encoded)) - len('""')
        cost = escaped_len + (len(_JSON_ITEM_SEPARATOR) if kept else 0)
        if cost > available:
            break
        kept.append(encoded)
        available -= cost
    if not kept:
        return None
    return _dumps({"role": role, "content": "{" + _JSON_ITEM_SEPARATOR.join(kept) + "}"})


# --------------------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------------------


def extract_request(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    """Return ``(state, questions)`` whether they were passed positionally or by keyword."""
    state = args[0] if len(args) > 0 else kwargs.get("state")
    questions = args[1] if len(args) > 1 else kwargs.get("questions")
    return state, questions


def resolve_request_model(instance: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    """Return the model the call will use: the per-call override, else the client default."""
    model = kwargs.get("model")
    if model:
        return str(model)
    config = getattr(instance, "_config", None)
    default_model = getattr(config, "default_model", None)
    return str(default_model) if default_model else None


def build_input(state: Any, questions: Any) -> Optional[str]:
    """Build the ``input`` attribute: the state entry, then the questions entry.

    The questions entry's room is set aside first (at most half the attribute),
    so a long state never pushes the questions out; the state is then cut to
    what is left.

    Returns:
        A JSON array of ``{"role", "content"}`` entries, or None when there is nothing to record.
    """
    max_len = get_attribute_max_len()
    remaining = max_len - len("[]")

    questions_entry: Optional[str] = None
    if isinstance(questions, Mapping) and questions:
        questions_entry = _mapping_entry_within(_QUESTIONS_ROLE, questions, max_len // _QUESTIONS_MAX_SHARE - len("[]"))
        if questions_entry:
            remaining -= len(questions_entry) + len(_JSON_ITEM_SEPARATOR)

    entries: List[str] = []
    if state is not None and state != "":
        state_text = state if isinstance(state, str) else _dumps(state)
        if state_entry := _text_entry_within(_STATE_ROLE, state_text, remaining):
            entries.append(state_entry)
    if questions_entry:
        entries.append(questions_entry)
    return _join_json_array(entries) if entries else None


def set_request_attributes(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Record the span type, the requested model, and the state and questions as ``input``.

    Runs before the call and serialises caller content, so any failure is logged
    rather than allowed to abort the user's request.
    """
    if not span.is_recording():
        return
    try:
        span.set_attribute(NETRA_SPAN_TYPE_ATTR, SpanType.GENERATION.value)
        if model := resolve_request_model(instance, kwargs):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    except Exception:
        logger.debug("Failed to set request attributes for TypeSafe span", exc_info=True)

    if not should_send_prompts():
        return
    try:
        if input_json := build_input(*extract_request(args, kwargs)):
            span.set_attribute(INPUT_ATTRIBUTE, input_json)
    except Exception:
        logger.warning("netra.instrumentation.libraries.typesafe: failed to record input", exc_info=True)


# --------------------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------------------


def _response_as_dict(response: Any) -> Dict[str, Any]:
    """Dump the response model to a dict; custom ``response_model`` types included."""
    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump(mode="json")
        except Exception:
            dumped = response.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(response, Mapping):
        return dict(response)
    return {}


def build_output(response_dict: Dict[str, Any]) -> Optional[str]:
    """Build the ``output`` attribute: one assistant entry holding the ``answers`` object as returned.

    A custom ``response_model`` may not carry ``answers``; its body, minus the
    model and usage already recorded elsewhere, is recorded instead.

    Returns:
        A JSON array with one ``{"role", "content"}`` entry, or None when nothing fits.
    """
    answers = response_dict.get("answers")
    if not isinstance(answers, Mapping):
        answers = {k: v for k, v in response_dict.items() if k not in ("model", "usage")}
    entry = _mapping_entry_within(_ANSWERS_ROLE, answers, get_attribute_max_len() - len("[]"))
    return _join_json_array([entry]) if entry else None


def set_response_attributes(span: Span, response: Any) -> None:
    """Record the response model, token usage, and the answers as ``output``."""
    if not span.is_recording():
        return
    try:
        response_dict = _response_as_dict(response)

        if model := response_dict.get("model"):
            span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, str(model))

        usage = response_dict.get("usage")
        if isinstance(usage, Mapping):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if isinstance(input_tokens, int):
                span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
            if isinstance(output_tokens, int):
                span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, input_tokens + output_tokens)

        if should_send_prompts() and (output_json := build_output(response_dict)):
            span.set_attribute(OUTPUT_ATTRIBUTE, output_json)
    except Exception:
        # Runs after the call succeeded: never cost the user their response.
        logger.warning("netra.instrumentation.libraries.typesafe: failed to record output", exc_info=True)
