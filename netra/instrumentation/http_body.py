"""Bounded capture, parsing and serialization of streaming HTTP response bodies.

The ``requests`` and ``httpx`` streaming wrappers tee every chunk the caller
reads so the body can be recorded on the span. This module is the pipeline that
sits behind that tee, in the order it runs:

* :class:`BoundedBodyBuffer` — retains a bounded prefix while counting the whole
  stream, so tracing a large download costs a fixed amount of memory.
* :func:`parse_streaming_body` — turns the retained prefix into the structured
  value recorded on the span (SSE, NDJSON, concatenated JSON, text, binary).
* :func:`serialize_bounded_output` — serializes the span envelope plus that body
  within the attribute budget, preserving the truncation marker.

:func:`build_streaming_output` runs all three and is what the ``requests`` and
``httpx`` adapters call; they own only their per-library header sanitization.
"""

import json
from typing import Any, Dict, List, NamedTuple, Optional, Union

from netra.config import get_attribute_max_len
from netra.utils import TRUNCATION_MARKER_KEY

# Appended to a truncated body so the cut is visible in the UI, not just implied
# by a flag somewhere above it.
TRUNCATION_ELLIPSIS = "..."

# A streaming body is parsed before it is serialized onto the span, and parsing
# can shrink it: an SSE event ``data: {...}\n\n`` loses its framing and becomes
# ``{...}, ``.  Retaining exactly the character budget would therefore export a
# *short* attribute for SSE, so the byte budget carries a headroom factor.  Four
# covers the densest realistic framing — a ~9-byte event wrapping a ~3-character
# payload — and still bounds retention to a few hundred kilobytes.
_PARSE_COMPACTION_HEADROOM = 4

# Shrinking works on the actual serialized string, so a couple of rounds is
# always enough; the bound only exists so a pathological body cannot spin.
_MAX_FIT_ROUNDS = 8

_SSE_DATA_PREFIX = "data:"
_SSE_DONE_SENTINEL = "[DONE]"


def _trim_partial_utf8_tail(data: bytes) -> bytes:
    """Drop a trailing incomplete UTF-8 sequence from *data*.

    Network chunk boundaries do not respect codepoint boundaries, so a body
    captured only up to a byte limit can end in the middle of a multi-byte
    character.  Decoding that raises ``UnicodeDecodeError``, which callers read
    as "this body is binary" — dropping the at-most-3-byte remnant keeps a
    truncated text body recognizable as text.

    Args:
        data: The retained body prefix.

    Returns:
        *data* unchanged if it already ends on a codepoint boundary, otherwise
        *data* without the incomplete trailing sequence.
    """
    for offset in range(1, min(4, len(data)) + 1):
        byte = data[-offset]
        if byte < 0x80:  # ASCII: the sequence ends here, nothing to trim
            return data
        if byte >= 0xC0:  # Lead byte: compare bytes seen against bytes required
            required = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return data if offset >= required else data[:-offset]
        # 0x80..0xBF is a continuation byte — keep walking back to the lead byte
    return data


class BoundedBodyBuffer:
    """Accumulates streaming HTTP body bytes up to the span-attribute size limit.

    A streaming response can be arbitrarily large, but the attribute it ends up
    in is capped at ``attribute_max_len`` by ``InstrumentationSpanProcessor``.
    Buffering a whole multi-gigabyte body only to discard all but the first
    50,000 characters is what makes tracing a large download run the process out
    of memory, so capture stops near that limit while ``total_bytes`` keeps
    counting everything that actually flowed.

    The contract is that the buffer is never the reason an exported attribute is
    short: it retains ``attribute_max_len * _PARSE_COMPACTION_HEADROOM`` bytes so
    that whatever the body parses into still overflows the character budget and
    gets trimmed at the usual place.
    """

    __slots__ = ("_max_bytes", "_parts", "_captured_bytes", "_total_bytes")

    def __init__(self, max_bytes: Optional[int] = None) -> None:
        """Initialize the buffer.

        Args:
            max_bytes: Maximum number of bytes to retain. Defaults to the
                configured ``attribute_max_len`` plus parse headroom. A value of
                zero or less disables retention while still counting
                ``total_bytes``.
        """
        if max_bytes is None:
            max_bytes = get_attribute_max_len() * _PARSE_COMPACTION_HEADROOM
        self._max_bytes = max_bytes
        self._parts: List[bytes] = []
        self._captured_bytes = 0
        self._total_bytes = 0

    def append(self, chunk: Union[bytes, bytearray, str]) -> None:
        """Record a stream chunk, retaining only the bytes that still fit.

        Chunk types other than bytes/bytearray/str are counted as nothing and
        ignored, matching what the streaming wrappers previously did.

        Args:
            chunk: A chunk as yielded by the wrapped response iterator.
        """
        if isinstance(chunk, str):
            data: Union[bytes, bytearray] = chunk.encode("utf-8")
        elif isinstance(chunk, (bytes, bytearray)):
            data = chunk
        else:
            return

        self._total_bytes += len(data)

        remaining = self._max_bytes - self._captured_bytes
        if remaining <= 0:
            return

        retained = data[:remaining] if len(data) > remaining else data
        # bytes() is a no-op for an exact bytes object and detaches a bytearray
        # the caller is free to mutate after yielding it.
        self._parts.append(bytes(retained))
        self._captured_bytes += len(retained)

    @property
    def total_bytes(self) -> int:
        """Total bytes seen on the stream, including bytes that were not retained."""
        return self._total_bytes

    @property
    def truncated(self) -> bool:
        """True when the stream carried more bytes than the cap allowed retaining."""
        return self._total_bytes > self._captured_bytes

    def getvalue(self) -> bytes:
        """Return the retained prefix of the body, ending on a UTF-8 boundary."""
        data = b"".join(self._parts)
        return _trim_partial_utf8_tail(data) if self.truncated else data


class ParsedBody(NamedTuple):
    """A parsed response body plus how it should be presented.

    Attributes:
        value: The parsed body.
        is_placeholder: True for a ``<binary content: N bytes>`` description.
            That is a description rather than content, so marking it with an
            ellipsis would read as though the description were cut short.
        truncated: True when *parsing* dropped content, independently of whether
            the capture buffer did. Callers must fold this into the span's
            truncation marker or the recorded body is silently short.
    """

    value: Any
    is_placeholder: bool = False
    truncated: bool = False


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
    body happens to overflow the budget.
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


def parse_streaming_body(accumulated: bytes, total_bytes: int, *, truncated: bool, budget: int) -> ParsedBody:
    """Parse retained streaming bytes into the value recorded on the span.

    Handles SSE (``data: {...}``), NDJSON, plain concatenated JSON objects, and
    falls back to decoded text or a binary placeholder.

    Parsing stops once *budget* characters of content have been built. The
    capture buffer deliberately retains several times the attribute budget (see
    ``_PARSE_COMPACTION_HEADROOM``), and turning all of it into Python objects
    only to discard most of them is the largest allocation left in this path --
    one that stacks when many streams finish at the same moment.

    Args:
        accumulated: Bytes retained from the response; may be a prefix.
        total_bytes: The real size of the body, used for the binary placeholder.
        truncated: Whether *accumulated* is only a prefix of the body.
        budget: Characters of content worth building.

    Returns:
        A :class:`ParsedBody` whose ``truncated`` flag reports parser-side cuts.
    """
    if truncated:
        accumulated = _trim_to_record_boundary(accumulated)

    try:
        text = accumulated.decode("utf-8")
    except UnicodeDecodeError:
        return ParsedBody(f"<binary content: {total_bytes} bytes>", is_placeholder=True)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line.startswith(_SSE_DATA_PREFIX) for line in lines):
        events = _parse_sse(lines, budget)
        if events.items:
            if events.truncated:
                return ParsedBody(events.items, truncated=True)
            return ParsedBody(_single_or_list(events.items))

    values = _parse_json_sequence(text, budget)
    if values is not None:
        if values.truncated:
            return ParsedBody(values.items, truncated=True)
        return ParsedBody(_single_or_list(values.items))

    # Plain text: slicing to the budget is safe because JSON escaping only grows
    # the serialized form, so the slice still overflows and gets trimmed exactly.
    return ParsedBody(text[:budget], truncated=len(text) > budget)


def _with_ellipsis(body: Any) -> Any:
    """Attach :data:`TRUNCATION_ELLIPSIS` to the tail of *body*."""
    if isinstance(body, str):
        return body + TRUNCATION_ELLIPSIS
    if isinstance(body, list):
        return [*body, TRUNCATION_ELLIPSIS]
    return body


def _shrink_body(body: Any, deficit: int) -> Optional[Any]:
    """Drop roughly *deficit* serialized characters from the tail of *body*.

    Args:
        body: The body to shrink.
        deficit: How many characters the serialized output is over budget.

    Returns:
        The shortened body, or None when it cannot shrink any further.
    """
    if isinstance(body, str):
        keep = len(body) - deficit
        return body[:keep] if keep > 0 else None
    if isinstance(body, list) and body:
        # Entries in a stream are near-uniform in size, so one estimate lands
        # close and the caller's re-measure absorbs whatever it missed.
        per_entry = max(1, len(json.dumps(body, default=str)) // len(body))
        keep = len(body) - max(1, -(-deficit // per_entry))
        return body[:keep] if keep > 0 else None
    return None


def serialize_bounded_output(
    envelope: Dict[str, Any],
    parsed: ParsedBody,
    *,
    total_bytes: int,
    truncated: bool,
    max_len: int,
) -> str:
    """Serialize a span output envelope plus its body, kept inside *max_len*.

    The body is trimmed here rather than left to ``InstrumentationSpanProcessor``
    because that processor cuts the serialized attribute at a fixed length --
    which would slice off the trailing ellipsis, the only part of the value that
    shows a reader the content was cut. Doing the final trim here also lets the
    marker be raised for a body that fit the capture buffer but still overflows
    the attribute budget.

    Args:
        envelope: Everything but the body (status, headers). Not mutated.
        parsed: The parsed body to place last.
        total_bytes: The real size of the body on the wire.
        truncated: Whether the body is only part of what was streamed.
        max_len: The attribute budget the result must fit within.

    Returns:
        The serialized output, at most *max_len* characters unless the envelope
        alone already exceeds it.
    """

    def render(body: Any, *, is_truncated: bool) -> str:
        # Insertion order is the wire order: the marker goes before the body so
        # it survives even if something downstream trims the tail anyway.
        data = dict(envelope)
        if is_truncated:
            data[TRUNCATION_MARKER_KEY] = True
            data["body_bytes"] = total_bytes
        data["body"] = _with_ellipsis(body) if is_truncated and not parsed.is_placeholder else body
        return json.dumps(data)

    fullest = render(parsed.value, is_truncated=truncated)
    if len(fullest) <= max_len:
        return fullest

    body, serialized = parsed.value, fullest
    for _ in range(_MAX_FIT_ROUNDS):
        shrunk = _shrink_body(body, len(serialized) - max_len)
        if shrunk is None:
            break
        body = shrunk
        serialized = render(body, is_truncated=True)
        if len(serialized) <= max_len:
            return serialized

    # Nothing fits -- the envelope alone is over budget (huge headers, or a
    # max_len smaller than it). Hand back the fullest version and let the
    # exporter's hard trim do what it would have done anyway, rather than
    # shipping a body we shrank for no gain.
    return fullest


def build_streaming_output(envelope: Dict[str, Any], body_buffer: BoundedBodyBuffer) -> str:
    """Parse and serialize a captured streaming body into a span ``output`` value.

    Args:
        envelope: Everything but the body (status, headers), already sanitized
            by the calling instrumentation. Not mutated.
        body_buffer: The body bytes teed off the stream as the caller read it.

    Returns:
        The serialized output, bounded by the configured ``attribute_max_len``.
    """
    max_len = get_attribute_max_len()
    parsed = parse_streaming_body(
        body_buffer.getvalue(),
        body_buffer.total_bytes,
        truncated=body_buffer.truncated,
        budget=max_len,
    )
    return serialize_bounded_output(
        envelope,
        parsed,
        total_bytes=body_buffer.total_bytes,
        truncated=body_buffer.truncated or parsed.truncated,
        max_len=max_len,
    )
