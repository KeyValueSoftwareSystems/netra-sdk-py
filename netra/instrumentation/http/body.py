"""Recording HTTP request and response bodies on a span, within budget.

This is the HTTP-shaped composition of two transport-agnostic pieces:
:mod:`netra.instrumentation.capture.bounded_capture` (bounded retention and budgeted serialization) and
:mod:`netra.instrumentation.capture.stream_formats` (SSE / NDJSON / JSON parsing). The
``requests``, ``httpx`` and ``fastapi`` instrumentations call in here; they own
only their per-library access to headers and raw bytes.

Two entry points, one pipeline behind both:

* :func:`build_streaming_output` for a body teed off a stream as the caller
  reads it, via a buffer from :func:`new_body_buffer`.
* :func:`build_response_output` for a body the HTTP library already holds whole.

The second exists because "already in memory" is not the same as "free to
record": parsing and re-serializing a 200 MB response so the exporter can keep
50,000 characters of it is Netra's own allocation, and it is avoidable by
running that body through the same bounded buffer a stream goes through.
"""

import json
from typing import Any, Mapping, Union

from netra.config import get_attribute_max_len
from netra.instrumentation.capture.bounded_capture import BoundedStreamBuffer, serialize_within_budget
from netra.instrumentation.capture.stream_formats import parse_streaming_body

# A body is parsed before it is serialized onto the span, and parsing can shrink
# it: an SSE event ``data: {...}\n\n`` loses its framing and becomes ``{...}, ``.
# Retaining exactly the character budget would therefore export a *short*
# attribute for SSE, so the byte budget carries a headroom factor. Four covers
# the densest realistic framing -- a ~9-byte event wrapping a ~3-character
# payload -- and still bounds retention to a few hundred kilobytes.
#
# Accepted limitation: this is a heuristic, not a guarantee. A stream diluted
# with framing the parser discards entirely -- ``event:`` lines, ``:`` comment
# keep-alives -- spends retained bytes on content that never reaches the span,
# and the exported attribute comes in under budget as a result. A measured case
# is pinned in ``tests/test_streaming_body_capture.py``. Raising the factor
# trades memory for a longer tail of that case; four is where we chose to sit.
_PARSE_COMPACTION_HEADROOM = 4


def new_body_buffer() -> BoundedStreamBuffer:
    """Return a capture buffer sized for an HTTP body that will be parsed.

    The headroom factor lives with the caller that needs it rather than in the
    buffer, because how much slack a capture needs depends on what the parser
    downstream will do to it.
    """
    return BoundedStreamBuffer(get_attribute_max_len() * _PARSE_COMPACTION_HEADROOM)


def build_streaming_output(envelope: Mapping[str, Any], body_buffer: BoundedStreamBuffer) -> str:
    """Parse and serialize a captured streaming body into a span attribute value.

    Args:
        envelope: Everything but the body (status, headers, url), already
            sanitized by the calling instrumentation. Not mutated.
        body_buffer: The body bytes teed off the stream as the caller read it.

    Returns:
        The serialized value, bounded by the configured ``attribute_max_len``.
    """
    max_len = get_attribute_max_len()
    parsed = parse_streaming_body(
        body_buffer.getvalue(),
        body_buffer.total_bytes,
        truncated=body_buffer.truncated,
        budget=max_len,
    )
    return serialize_within_budget(envelope, parsed, max_len=max_len)


def build_response_output(envelope: Mapping[str, Any], raw_body: Union[bytes, bytearray, str, None]) -> str:
    """Parse and serialize an already-received body under the same bound.

    Text bodies are accepted as well as bytes, so a caller holding a decoded
    body -- or a placeholder standing in for one it must not read -- gets the
    same budget enforcement without a second code path.

    Args:
        envelope: Everything but the body, already sanitized. Not mutated.
        raw_body: The raw body bytes or text, or None when there is no body.

    Returns:
        The serialized value, bounded by the configured ``attribute_max_len``.
        The ``body`` key is omitted entirely when *raw_body* is empty, so a
        bodiless response stays distinguishable from one carrying an empty body.
    """
    if not raw_body:
        return json.dumps(dict(envelope), default=str)
    buffer = new_body_buffer()
    buffer.append(raw_body)
    return build_streaming_output(envelope, buffer)
