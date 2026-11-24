from typing import Any, Dict, List, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
)

from netra.config import get_attribute_max_len

# The size-bounded JSON encoding of decision input / questions / answers is shared
# with the OpenAI Decisions API capture, so both providers produce the same
# ``input`` / ``output`` shape and obey the attribute length limit the same way.
# These helpers are plain functions with no dependency on the ``openai`` package.
from netra.instrumentation.libraries.openai.utils import (
    _JSON_ITEM_SEPARATOR,
    INPUT_ATTRIBUTE,
    OUTPUT_ATTRIBUTE,
    _as_prompt_content,
    _encode_entry_within,
    _items_entry,
    _join_json_array,
    should_send_prompts,
)
from netra.instrumentation.span_utils import should_capture_content
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# Gemini Decisions API (``client.decisions.create``): the questions are recorded as the
# last ``input`` entry and the answers as the ``output`` entry, each a JSON array.
_DECISION_QUESTIONS_ROLE = "user"
_DECISION_ANSWERS_ROLE = "assistant"
# The questions entry may take at most 1/N of the ``input`` attribute, leaving the rest for the input.
_DECISION_QUESTIONS_MAX_SHARE = 2
# The API takes model names as ``models/{model}``; pricing and dashboards key off the bare name.
_MODEL_RESOURCE_PREFIX = "models/"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed for GenAI."""
    return bool(
        context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        or context_api.get_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY)
    )


def set_request_attributes(span: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Set request attributes (model, parameters, prompts) for Google GenAI."""
    if not span.is_recording():
        return

    if model := kwargs.get("model"):
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)

    if config := kwargs.get("config"):
        config_mappings = [
            ("temperature", SpanAttributes.LLM_REQUEST_TEMPERATURE),
            ("top_p", SpanAttributes.LLM_REQUEST_TOP_P),
            ("top_k", SpanAttributes.LLM_TOP_K),
            ("max_output_tokens", SpanAttributes.LLM_REQUEST_MAX_TOKENS),
            ("stop_sequences", SpanAttributes.LLM_CHAT_STOP_SEQUENCES),
            ("presence_penalty", SpanAttributes.LLM_PRESENCE_PENALTY),
            ("frequency_penalty", SpanAttributes.LLM_FREQUENCY_PENALTY),
        ]

        for attr_name, span_attr in config_mappings:
            if (value := getattr(config, attr_name, None)) is not None:
                span.set_attribute(span_attr, value)

    if should_capture_content() and (contents := kwargs.get("contents")):
        if isinstance(contents, str):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", contents)
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            return

        content_list = contents if isinstance(contents, list) else [contents]

        for index, content in enumerate(content_list):
            role = getattr(content, "role", "user")
            text = None

            if hasattr(content, "parts"):
                text = next((part.text for part in content.parts if hasattr(part, "text") and part.text), None)
            elif hasattr(content, "text"):
                text = content.text
                role = "user"
            elif hasattr(content, "function_call"):
                fc = content.function_call
                text = f"Function Call: {fc.name}, Args: {fc.args}"
                role = "model"
            else:
                text = str(content)
                role = "user"

            if text:
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", text)
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)


def set_response_attributes(span: Any, response: Any) -> None:
    """Set response attributes for Google GenAI."""
    if not span.is_recording():
        return

    capture = should_capture_content()
    if isinstance(response, str):
        if capture:
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response)
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        return

    usage = _extract_usage_metadata(response)
    if usage is not None:
        if (total := getattr(usage, "total_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)

        output = 0
        if isinstance(ct := getattr(usage, "candidates_token_count", None), int):
            output += ct
        if isinstance(tt := getattr(usage, "thoughts_token_count", None), int):
            output += tt
        if output > 0:
            span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output)

        if (cached := getattr(usage, "cached_content_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)

        if (prompt := getattr(usage, "prompt_token_count", None)) is not None:
            span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt)

        if not capture:
            return

        if isinstance(response, dict):
            if isinstance(text := response.get("content"), str) and text:
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", text)
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
                return

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list):
            for index, candidate in enumerate(candidates):
                content = getattr(candidate, "content", None)
                if content is None:
                    continue

                parts = getattr(content, "parts", None)
                if not isinstance(parts, list):
                    continue

                for part in parts:
                    if isinstance(text := getattr(part, "text", None), str):
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", text)
                        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", "assistant")
                        break


def _extract_usage_metadata(response: Any) -> Any:
    """Extract usage metadata from response."""
    if hasattr(response, "usage_metadata"):
        return response.usage_metadata
    if hasattr(response, "get"):
        chunk = response.get("chunk", {})
        return getattr(chunk, "usage_metadata", None)
    return None


def model_as_dict(input_object: Any) -> Any:
    """Convert OpenAI model object to dictionary"""
    if hasattr(input_object, "model_dump"):
        return input_object.model_dump()

    elif hasattr(input_object, "to_dict"):
        return input_object.to_dict()

    elif isinstance(input_object, dict):
        return input_object

    else:
        return {}


def _to_plain(value: Any) -> Any:
    """Convert a google-genai pydantic object to plain JSON-compatible data; pass anything else through.

    ``mode="json"`` turns bytes (inline media) into strings and ``exclude_none``
    drops the many unset optional fields, so the recorded JSON stays readable.
    """
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except TypeError:
            return value.model_dump()
    return value


def _decision_model_name(model: Any) -> Optional[str]:
    """Return the model name without the ``models/`` resource prefix, or None if not a string."""
    if not isinstance(model, str) or not model:
        return None
    return model[len(_MODEL_RESOURCE_PREFIX) :] if model.startswith(_MODEL_RESOURCE_PREFIX) else model


def _decision_input_entry(block: Any, max_content_len: int) -> Optional[Dict[str, Any]]:
    """Convert one Decisions API input content block to a ``{"role": "user", "content"}`` entry.

    Text blocks are recorded as their text. Media blocks (image, audio, video,
    document) are recorded as JSON with their inline base64 ``data`` replaced
    by a placeholder such as ``[image]``; ``uri`` and ``mime_type`` are kept.

    Returns:
        The entry, or None when the block carries nothing to record.
    """
    if isinstance(block, str):
        return {"role": "user", "content": block[:max_content_len]} if block else None

    block = _to_plain(block)
    if not isinstance(block, dict) or not block:
        return None

    block_type = block.get("type") or "text"
    if block_type == "text":
        text = block.get("text")
        if not isinstance(text, str) or not text:
            return None
        return {"role": "user", "content": text[:max_content_len]}

    media = {key: value for key, value in block.items() if key != "data"}
    if "data" in block:
        media["data"] = f"[{block_type}]"
    return {"role": "user", "content": _as_prompt_content(media, max_content_len)}


def _build_decision_input(input_data: Any, questions_entry: Optional[str] = None) -> Optional[str]:
    """Build the ``input`` attribute JSON for a Gemini Decisions API request.

    ``input`` is the shared context every question is evaluated against: a
    string, one content block, or a list of content blocks. Each becomes a
    ``user`` entry. ``questions_entry``, the already-encoded questions entry, is
    appended last so the questions read after the content they are asked about.
    Only lists and tuples are walked, so a one-shot iterable reaches the API intact.

    The result never exceeds the configured attribute length. The questions
    entry's room is set aside first, so a long input never pushes the questions
    out; each input entry is then cut to the budget the earlier entries left
    and later entries are dropped once it is used up, so the attribute stays valid JSON.

    Returns:
        A JSON array of ``{"role", "content"}`` entries, or None when there is nothing to record.
    """
    if isinstance(input_data, (list, tuple)):
        blocks: Any = input_data
    elif input_data is None:
        blocks = []
    else:
        blocks = [input_data]

    encoded_entries: List[str] = []
    remaining = get_attribute_max_len() - len("[]")
    if questions_entry:
        remaining -= len(questions_entry) + len(_JSON_ITEM_SEPARATOR)
    for block in blocks:
        separator_len = len(_JSON_ITEM_SEPARATOR) if encoded_entries else 0
        budget = remaining - separator_len
        if budget <= 0:
            break
        if not (entry := _decision_input_entry(block, budget)):
            continue
        if (encoded := _encode_entry_within(entry, budget)) is None:
            break
        encoded_entries.append(encoded)
        remaining = budget - len(encoded)

    if questions_entry:
        encoded_entries.append(questions_entry)
    return _join_json_array(encoded_entries) if encoded_entries else None


def _keyed_items(mapping: Any) -> List[Dict[str, Any]]:
    """Flatten a ``{id: question-or-answer}`` mapping into ``[{"id": id, ...fields}]``, in mapping order.

    Questions and answers are keyed by caller-chosen IDs; carrying the ID inside
    each item keeps answers matchable to their questions once both are arrays.
    Values that are not objects are recorded under ``value``.
    """
    items: List[Dict[str, Any]] = []
    for key, value in mapping.items():
        plain = _to_plain(value)
        items.append({"id": key, **plain} if isinstance(plain, dict) else {"id": key, "value": plain})
    return items


def _decision_questions_entry(questions: Any) -> Optional[str]:
    """Encode the Decisions API ``questions`` as the ``user`` entry that ends the ``input`` attribute.

    Its ``content`` is a JSON array with one ``{"id", "type", "instructions",
    "criteria"}`` object per question. The entry may use at most half the
    attribute length, so the input the questions are about still has room;
    whole questions are dropped from the end past that.

    Returns:
        The encoded entry, or None when there are no questions to record.
    """
    if not isinstance(questions, Mapping) or not questions:
        return None
    max_len = get_attribute_max_len() // _DECISION_QUESTIONS_MAX_SHARE - len("[]")
    return _items_entry(_keyed_items(questions), _DECISION_QUESTIONS_ROLE, max_len)


def set_decision_request_attributes(span: Any, kwargs: Dict[str, Any]) -> None:
    """Record a Gemini Decisions API request on ``span``, without ever failing the call.

    The model is always recorded. The input and questions are caller content, so
    they are recorded (as ``input``) only when content tracing is on. Any failure
    is logged instead of aborting the user's request.
    """
    if not span.is_recording():
        return

    if model := _decision_model_name(kwargs.get("model")):
        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        # The response carries no model; record the requested one so the span can be priced.
        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, model)

    if not should_send_prompts():
        return

    questions_entry: Optional[str] = None
    try:
        questions_entry = _decision_questions_entry(kwargs.get("questions"))
    except Exception:  # noqa: BLE001 - question capture is best-effort; never block the user's request
        logger.warning(
            "netra.instrumentation.libraries.google_genai: failed to record decision questions", exc_info=True
        )

    try:
        if input_json := _build_decision_input(kwargs.get("input"), questions_entry):
            span.set_attribute(INPUT_ATTRIBUTE, input_json)
    except Exception:  # noqa: BLE001 - input capture is best-effort; never block the user's request
        logger.warning("netra.instrumentation.libraries.google_genai: failed to record decision input", exc_info=True)


def _set_decision_usage_attributes(span: Any, usage: Any) -> None:
    """Record the token counts of a Decisions API ``usage`` object (Interactions-style field names)."""
    usage = _to_plain(usage)
    if not isinstance(usage, dict):
        return

    def count(key: str) -> Optional[int]:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if (prompt := count("total_input_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt)
    output = [n for n in (count("total_output_tokens"), count("total_thought_tokens")) if n is not None]
    if output:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, sum(output))
    if (total := count("total_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total)
    if (cached := count("total_cached_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cached)


def set_decision_response_attributes(span: Any, response: Any) -> None:
    """Record a Gemini Decisions API response on ``span``, without ever failing the call.

    Token usage and, when the input was blocked, the block reason (as the
    finish reason) are always recorded. The answers are recorded as ``output``
    only when content tracing is on: one assistant entry whose ``content`` is a
    JSON array of ``{"id", "type", ...}`` answers (``noul`` probability, ``choice``
    with its probabilities, or ``score`` with its levels), keeping only as many
    whole answers as fit the attribute length.

    A raw-response object (``decisions.with_raw_response.create``) has no parsed
    fields and records nothing.
    """
    if not span.is_recording():
        return

    try:
        _set_decision_usage_attributes(span, getattr(response, "usage", None))

        feedback = _to_plain(getattr(response, "prompt_feedback", None))
        if isinstance(feedback, dict) and (block_reason := feedback.get("block_reason")):
            span.set_attribute(SpanAttributes.LLM_RESPONSE_FINISH_REASON, str(block_reason))
    except Exception:  # noqa: BLE001 - runs after the call succeeded; never cost the user their response
        logger.warning("netra.instrumentation.libraries.google_genai: failed to record decision usage", exc_info=True)

    answers = getattr(response, "answers", None)
    if not isinstance(answers, Mapping) or not answers or not should_send_prompts():
        return
    try:
        entry = _items_entry(_keyed_items(answers), _DECISION_ANSWERS_ROLE, get_attribute_max_len() - len("[]"))
        if entry:
            span.set_attribute(OUTPUT_ATTRIBUTE, _join_json_array([entry]))
    except Exception:  # noqa: BLE001 - runs after the call succeeded; never cost the user their response
        logger.warning("netra.instrumentation.libraries.google_genai: failed to record decision answers", exc_info=True)
