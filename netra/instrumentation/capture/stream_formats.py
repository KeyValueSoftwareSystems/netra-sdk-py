"""Parsing of the wire formats streamed responses arrive in.

Server-sent events, NDJSON and bare concatenated JSON are what an HTTP response
body, an LLM completion stream and an agent event stream all look like on the
wire, so the parsing lives here rather than in any one instrumentation.

The entry point is :func:`parse_streaming_body`. It takes the prefix a
:class:`~netra.instrumentation.capture.bounded_capture.BoundedStreamBuffer` retained and returns a
:class:`~netra.instrumentation.capture.bounded_capture.BoundedValue` ready to hand to
:func:`~netra.instrumentation.capture.bounded_capture.serialize_within_budget`.

Parsing folds the buffer's truncation flag into its result: a prefix parses into
a partial value whether or not the parser itself stopped early, and a caller
that had to remember to OR those two flags together would eventually forget.
"""

import json
from typing import Any, List, NamedTuple, Optional

from netra.instrumentation.capture.bounded_capture import BoundedValue

_SSE_DATA_PREFIX = "data:"
_SSE_DONE_SENTINEL = "[DONE]"


class _PartialParse(NamedTuple):
    """What one parsing strategy produced, and whether it stopped early.

    Attributes:
        items: The values parsed so far, in stream order.
        truncated: True when parsing stopped at the budget with input left over.
    """

    items: List[Any]
    truncated: bool


def _trim_to_record_boundary(data: bytes) -> bytes:
    """Cut *data* back to the last complete record.

    A capture that stopped at a byte limit almost always ends mid-record, and
    half a frame can only be parsed into a junk string. Dropping it here is
    explicit; leaving it to a later length trim only works while the serialized
    value happens to overflow the budget.
    """
    for terminator in (b"\n\n", b"\n"):
        end = data.rfind(terminator)
        if end != -1:
            return data[: end + len(terminator)]
    return data


def _parse_sse(lines: List[str], budget: int) -> _PartialParse:
    """Parse ``data:`` lines into events, stopping once *budget* is reached.

    Args:
        lines: Stripped, non-empty lines of the body.
        budget: Characters of content worth building.

    Returns:
        The parsed events and whether parsing stopped before consuming *lines*.
    """
    events: List[Any] = []
    produced = 0
    for line in lines:
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data = line.removeprefix(_SSE_DATA_PREFIX).strip()
        if not data or data == _SSE_DONE_SENTINEL:
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            events.append(data)
        produced += len(data) + 2  # the entry plus the ", " that joins it
        if produced >= budget:
            return _PartialParse(events, truncated=True)
    return _PartialParse(events, truncated=False)


def _parse_json_sequence(text: str, budget: int) -> Optional[_PartialParse]:
    """Parse *text* as one or more JSON values, stopping once *budget* is reached.

    Covers a single JSON document, NDJSON, and bare concatenated objects.

    Args:
        text: The decoded body.
        budget: Characters of content worth building.

    Returns:
        The parsed values and whether parsing stopped early, or None when *text*
        is not a complete sequence of JSON values.
    """
    decoder = json.JSONDecoder()
    results: List[Any] = []
    produced = 0
    index = 0
    stripped = text.strip()
    try:
        while index < len(stripped):
            value, end_index = decoder.raw_decode(stripped, index)
            results.append(value)
            produced += end_index - index
            index = end_index
            while index < len(stripped) and stripped[index] in " \t\n\r":
                index += 1
            if produced >= budget:
                return _PartialParse(results, truncated=True)
    except json.JSONDecodeError:
        return None

    if results and index == len(stripped):
        return _PartialParse(results, truncated=False)
    return None


def _single_or_list(items: List[Any]) -> Any:
    """Unwrap a one-element parse so a lone JSON document is not recorded as a list."""
    return items[0] if len(items) == 1 else items


def parse_streaming_body(accumulated: bytes, total_bytes: int, *, truncated: bool, budget: int) -> BoundedValue:
    """Parse retained stream bytes into the value to record on a span.

    Handles SSE (``data: {...}``), NDJSON, plain concatenated JSON objects, and
    falls back to decoded text or a binary placeholder.

    Parsing stops once *budget* characters of content have been built. A capture
    buffer deliberately retains several times the attribute budget (see
    ``http.body._PARSE_COMPACTION_HEADROOM``), and turning all of it into Python
    objects only to discard most of them is the largest allocation left in this
    path -- one that stacks when many streams finish at the same moment.

    Args:
        accumulated: Bytes retained from the stream; may be a prefix.
        total_bytes: The real size of the stream, used for the binary placeholder
            and reported alongside the truncation marker.
        truncated: Whether *accumulated* is only a prefix.
        budget: Characters of content worth building.

    Returns:
        A :class:`~netra.instrumentation.capture.bounded_capture.BoundedValue` whose ``truncated`` flag
        already accounts for both a prefixed capture and a parser-side cut.
    """
    if truncated:
        accumulated = _trim_to_record_boundary(accumulated)

    try:
        text = accumulated.decode("utf-8")
    except UnicodeDecodeError:
        return BoundedValue(
            f"<binary content: {total_bytes} bytes>",
            truncated=truncated,
            total_size=total_bytes,
            is_placeholder=True,
        )

    def result(value: Any, *, parser_cut: bool) -> BoundedValue:
        return BoundedValue(value, truncated=truncated or parser_cut, total_size=total_bytes)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line.startswith(_SSE_DATA_PREFIX) for line in lines):
        events = _parse_sse(lines, budget)
        if events.items:
            if events.truncated:
                return result(events.items, parser_cut=True)
            return result(_single_or_list(events.items), parser_cut=False)

    values = _parse_json_sequence(text, budget)
    if values is not None:
        if values.truncated:
            return result(values.items, parser_cut=True)
        return result(_single_or_list(values.items), parser_cut=False)

    # Plain text: slicing to the budget is safe because JSON escaping only grows
    # the serialized form, so the slice still overflows and gets trimmed exactly.
    return result(text[:budget], parser_cut=len(text) > budget)
