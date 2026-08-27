"""Bounded capture and budgeted serialization of oversized telemetry values.

Two problems recur wherever the SDK records a value whose size the application,
not the SDK, controls:

* **Capture cost.** An instrumentation that accumulates a stream retains every
  chunk the caller reads, even though the exported attribute is capped at
  ``attribute_max_len``.  Holding a multi-gigabyte download in order to export
  50,000 characters of it is what makes tracing run a process out of memory.
* **Where the cut lands.** A value that overflows the budget is trimmed by
  ``InstrumentationSpanProcessor`` at a fixed character count.  That slice lands
  mid-token: JSON stops parsing, and a truncation marker appended at the end is
  the first thing lost.

This module answers both without knowing what produced the value, so HTTP
bodies, LLM token streams and agent output can share one implementation:

* :class:`BoundedStreamBuffer` retains a bounded prefix while counting
  everything that flows through it, so capture cost is flat in the size of the
  stream.
* :func:`serialize_within_budget` serializes an envelope plus a payload inside a
  character budget, shrinking the payload *structurally* -- dropping whole list
  entries or dict values rather than slicing the serialized string -- so the
  result is still valid JSON and still carries its marker.

Nothing here reads the active config: every bound is passed in, because the
right bound depends on what the caller will do with the value afterwards (see
``http.body._PARSE_COMPACTION_HEADROOM`` for a caller that needs slack).

Callers that also need to parse a captured byte stream compose this with the
sibling :mod:`~netra.instrumentation.capture.stream_formats`;
:mod:`~netra.instrumentation.http.body` is the worked example of that
composition.
"""

import json
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Union

# Marker key set on any value the SDK cut short, wherever that happens. The
# Netra UI keys off it to show a value as partial, so every producer must use
# this exact string.
TRUNCATION_MARKER_KEY = "__truncated__"

# Appended to a truncated payload so the cut is visible in the UI, not just
# implied by a flag somewhere above it.
TRUNCATION_ELLIPSIS = "..."

# Shrinking works on the actual serialized string and re-measures every round,
# so a couple of rounds is always enough; the bound only exists so a
# pathological payload cannot spin.
_MAX_FIT_ROUNDS = 8


def _trim_partial_utf8_tail(data: bytes) -> bytes:
    """Drop a trailing incomplete UTF-8 sequence from *data*.

    Chunk boundaries do not respect codepoint boundaries, so a value captured
    only up to a byte limit can end in the middle of a multi-byte character.
    Decoding that raises ``UnicodeDecodeError``, which callers read as "this is
    binary" -- dropping the at-most-3-byte remnant keeps a truncated text value
    recognizable as text.

    Args:
        data: The retained prefix.

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
        # 0x80..0xBF is a continuation byte -- keep walking back to the lead byte
    return data


class BoundedStreamBuffer:
    """Accumulates streamed chunks up to a byte cap while counting all of them.

    A stream can be arbitrarily large, but the attribute it ends up in is capped
    by ``InstrumentationSpanProcessor``. Buffering the whole thing only to
    discard all but a prefix is what makes tracing a large payload run the
    process out of memory, so retention stops at *max_bytes* while
    :attr:`total_bytes` keeps counting everything that actually flowed.

    Chunks may be ``bytes``, ``bytearray`` or ``str``; text chunks are counted
    and retained as their UTF-8 encoding, and :meth:`getvalue` hands back a
    prefix that always ends on a codepoint boundary, so it decodes cleanly.

    The buffer is not synchronized. One stream is consumed by one reader, which
    is the only way the tee in front of it is correct in the first place.
    """

    __slots__ = ("_max_bytes", "_parts", "_captured_bytes", "_total_bytes")

    def __init__(self, max_bytes: int) -> None:
        """Initialize the buffer.

        Args:
            max_bytes: Maximum number of bytes to retain. Zero or less disables
                retention while still counting :attr:`total_bytes`.
        """
        self._max_bytes = max_bytes
        self._parts: List[bytes] = []
        self._captured_bytes = 0
        self._total_bytes = 0

    def append(self, chunk: Union[bytes, bytearray, str]) -> None:
        """Record a chunk, retaining only the bytes that still fit.

        Chunk types other than bytes/bytearray/str are counted as nothing and
        ignored: a stream of arbitrary objects has no byte length to speak of,
        and guessing one would corrupt the reported total.

        Args:
            chunk: A chunk as yielded by the wrapped iterator.
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
        """Total bytes seen, including bytes that were not retained."""
        return self._total_bytes

    @property
    def truncated(self) -> bool:
        """True when the stream carried more bytes than the cap allowed retaining."""
        return self._total_bytes > self._captured_bytes

    def getvalue(self) -> bytes:
        """Return the retained prefix, ending on a UTF-8 codepoint boundary."""
        data = b"".join(self._parts)
        return _trim_partial_utf8_tail(data) if self.truncated else data


class BoundedValue(NamedTuple):
    """A payload to record, plus what is known about how complete it is.

    Attributes:
        value: The payload itself.
        truncated: True when *value* is only part of what was produced. The
            producer folds every reason for a cut into this one flag, so a
            consumer cannot record a short value while reporting it as whole.
        total_size: The real size of the complete value in bytes, recorded
            alongside the marker so a reader can see how much was dropped.
            None when the producer does not know it.
        is_placeholder: True when *value* describes the content instead of being
            it (``<binary content: N bytes>``). Such a description is complete
            as written, so it never gets an ellipsis.
    """

    value: Any
    truncated: bool = False
    total_size: Optional[int] = None
    is_placeholder: bool = False


def _measure(value: Any) -> int:
    """Serialized character count of *value*, the unit every budget here is in."""
    return len(json.dumps(value, default=str))


def _with_ellipsis(value: Any) -> Any:
    """Attach :data:`TRUNCATION_ELLIPSIS` to the tail of *value*."""
    if isinstance(value, str):
        return value + TRUNCATION_ELLIPSIS
    if isinstance(value, list):
        return [*value, TRUNCATION_ELLIPSIS]
    return value


def _shrink_value(value: Any, deficit: int) -> Optional[Any]:
    """Return *value* with roughly *deficit* serialized characters removed.

    Shrinking is structural rather than a slice of the serialized string, so
    whatever comes back still serializes to valid JSON.

    Args:
        value: The payload to shrink.
        deficit: How many characters the serialized output is over budget.

    Returns:
        The shortened payload, or None when *value* has no slack left to give.
        None is the caller's signal to stop trying.
    """
    if isinstance(value, str):
        if not value:
            return None
        # A deficit larger than the string means the overflow is not the
        # string's to cover -- the envelope is oversized. Give up everything
        # rather than nothing, so the caller ends on its smallest rendering.
        return value[: max(0, len(value) - deficit)]
    if isinstance(value, list):
        return _shrink_list(value, deficit)
    if isinstance(value, dict):
        return _shrink_dict(value, deficit)
    # Numbers, booleans and None are already as short as they serialize.
    return None


_LIST_SEPARATOR_CHARS = 2  # ", " between entries, as json.dumps writes them


def _shrink_list(items: List[Any], deficit: int) -> Optional[Any]:
    """Drop entries from the tail of *items*, then shrink into what is left.

    Entries are measured individually rather than averaged. A stream's entries
    are near-uniform, so an average would do -- but a parsed body's are not, and
    one fat entry among many small ones makes the average claim every entry is
    droppable when only one of them carries the weight.

    Dropping from the tail keeps the earliest entries, which is what "this was
    cut short" means for anything that arrived in order.
    """
    if not items:
        return None

    sizes = [_measure(item) for item in items]
    keep = len(items)
    freed = 0
    while keep > 1 and freed < deficit:
        keep -= 1
        freed += sizes[keep] + _LIST_SEPARATOR_CHARS

    if freed >= deficit:
        return items[:keep]

    # One entry left and still over: shrink inside it, so the payload degrades
    # to a partial record rather than to an empty list.
    inner = _shrink_value(items[0], deficit - freed)
    if inner is not None:
        return [inner]
    # It cannot shrink either. Dropped entries are still progress; nothing is not.
    return items[:keep] if keep < len(items) else None


def _shrink_dict(mapping: Dict[str, Any], deficit: int) -> Optional[Any]:
    """Shrink values widest-first until *deficit* is covered, keeping every key.

    Covering the whole deficit in one call matters: shrinking only the single
    widest value frees a fixed amount per call regardless of how far over budget
    the payload is, so a dict of twenty fat values needs twenty rounds and the
    caller's round limit gives up long before that.
    """
    if not mapping:
        return None

    # Widest first: that is where the deficit lives, and spending it there keeps
    # the narrow keys intact so the shape stays recognizable to a reader.
    by_width = sorted(((_measure(value), key) for key, value in mapping.items()), reverse=True)

    shrunk_mapping = dict(mapping)
    remaining = deficit
    changed = False
    for size, key in by_width:
        if remaining <= 0:
            break
        shrunk = _shrink_value(shrunk_mapping[key], remaining)
        if shrunk is None:
            continue
        shrunk_mapping[key] = shrunk
        remaining -= size - _measure(shrunk)
        changed = True
    if changed:
        return shrunk_mapping

    # Every value is a scalar, so no entry can give up characters -- drop whole
    # entries from the tail instead, always leaving at least one behind.
    keys = list(mapping)
    freed = 0
    while len(keys) > 1 and freed < deficit:
        freed += _measure({keys[-1]: mapping[keys[-1]]})
        keys.pop()
    if len(keys) == len(mapping):
        return None
    return {key: mapping[key] for key in keys}


# Envelope keys the payload and its real size are written to. Constants rather
# than parameters: no caller has ever needed a different pair, and the Netra UI
# reads these exact names.
_PAYLOAD_KEY = "body"
_PAYLOAD_SIZE_KEY = "body_bytes"


def serialize_within_budget(envelope: Mapping[str, Any], payload: BoundedValue, *, max_len: int) -> str:
    """Serialize *envelope* plus *payload* as JSON kept inside *max_len*.

    The payload is trimmed here rather than left to
    ``InstrumentationSpanProcessor`` because that processor cuts the serialized
    attribute at a fixed length -- which would slice off the trailing ellipsis,
    the only part of the value that shows a reader the content was cut, and
    leave the JSON unparseable. Doing the final trim here also lets the marker
    be raised for a payload that was captured whole but still overflows the
    attribute budget.

    Args:
        envelope: Everything but the payload (status, headers, model, ...).
            Not mutated.
        payload: The value to place last, and what is known about it.
        max_len: The character budget the result must fit within.

    Returns:
        The serialized output, at most *max_len* characters. Two cases can
        exceed it, both because no shorter output exists: the envelope alone is
        over budget, or the payload is a placeholder, which describes content
        rather than being it and so cannot be cut down.
    """

    def render(value: Any, *, is_truncated: bool) -> str:
        # Insertion order is wire order: the marker and the real size go before
        # the payload so they survive even if something downstream trims the
        # tail anyway.
        data: Dict[str, Any] = dict(envelope)
        if is_truncated:
            data[TRUNCATION_MARKER_KEY] = True
            if payload.total_size is not None:
                data[_PAYLOAD_SIZE_KEY] = payload.total_size
        data[_PAYLOAD_KEY] = _with_ellipsis(value) if is_truncated and not payload.is_placeholder else value
        return json.dumps(data, default=str)

    rendered = render(payload.value, is_truncated=payload.truncated)
    if len(rendered) <= max_len:
        return rendered

    if payload.is_placeholder:
        # A placeholder describes the content rather than being it, so it has
        # no slack to give: half of "<binary content: N bytes>" is not a
        # smaller description, it is a broken one.
        return rendered

    value = payload.value
    for _ in range(_MAX_FIT_ROUNDS):
        shrunk = _shrink_value(value, len(rendered) - max_len)
        if shrunk is None:
            break
        value = shrunk
        rendered = render(value, is_truncated=True)
        if len(rendered) <= max_len:
            return rendered

    # Structural shrinking ran out of moves. It can bottom out with room still
    # left -- a list whose last surviving entry is a number has nothing further
    # to give -- so degrade to a flat text preview, which can be sliced to any
    # length. Two rounds always suffice: dropping N characters from the preview
    # drops at least N from the rendering, since escaping only ever grows.
    preview = json.dumps(value, default=str)
    for _ in range(2):
        candidate = render(preview, is_truncated=True)
        if len(candidate) <= max_len:
            return candidate
        preview = preview[: max(0, len(preview) - (len(candidate) - max_len))]

    # Even an empty payload does not fit, so the envelope alone exceeds
    # *max_len*: outsized headers, or a max_len smaller than the envelope.
    # Nothing this function controls can fix that. Hand back the smallest
    # rendering rather than the fullest -- the payload is what the exporter's
    # hard trim would have cut anyway, so the smaller one keeps more envelope.
    return render(preview, is_truncated=True)
