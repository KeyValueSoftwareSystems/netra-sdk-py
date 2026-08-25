"""
Unit tests for bounded streaming-body capture in the HTTP instrumentations.

The ``requests`` and ``httpx`` streaming wrappers tee every chunk the caller
reads so the body can be recorded on the span.  They used to retain the whole
body, which made tracing a large download cost several times the download's size
in RAM even though the exported attribute is capped at ``attribute_max_len``.
Capture now stops at that same limit.

These tests pin the two invariants that matter:

    1. The caller still receives every byte, unaltered, no matter how large the
       body is.  Truncation applies to what Netra records, never to what the
       application reads.
    2. Peak retention is bounded by ``attribute_max_len`` and the recorded span
       still reports the *real* body size.

All three wrappers (requests, httpx sync, httpx async) share one pipeline, so
the behavioral tests run against each of them.
"""

import asyncio
import json
from typing import Any, Callable, Dict, Iterator, List

import pytest

from netra import config as config_module
from netra.config import _DEFAULT_ATTRIBUTE_MAX_LEN, Config, set_active_config
from netra.instrumentation.http_body import (
    _PARSE_COMPACTION_HEADROOM,
    TRUNCATION_ELLIPSIS,
    BoundedBodyBuffer,
    parse_streaming_body,
)
from netra.instrumentation.httpx.wrappers import AsyncStreamingWrapper
from netra.instrumentation.httpx.wrappers import StreamingWrapper as HttpxStreamingWrapper
from netra.instrumentation.requests.wrappers import StreamingWrapper as RequestsStreamingWrapper
from netra.utils import TRUNCATION_MARKER_KEY

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_active_config():
    """Isolate the process-global active config from other tests."""
    original = config_module._active_config
    config_module._active_config = None
    try:
        yield
    finally:
        config_module._active_config = original


class RecordingSpan:
    """Minimal Span stand-in that records the attributes set on it."""

    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}
        self.ended = False

    def is_recording(self) -> bool:
        return not self.ended

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def end(self) -> None:
        self.ended = True


class FakeResponse:
    """Stand-in for a streaming ``requests``/``httpx`` response."""

    def __init__(self, chunks: List[bytes], status_code: int = 200) -> None:
        self._chunk_source = chunks
        self.status_code = status_code
        self.headers = {"content-type": "application/octet-stream"}
        self.closed = False

    def _iter(self) -> Iterator[bytes]:
        yield from self._chunk_source

    # requests surface
    def iter_content(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        return self._iter()

    # httpx sync surface
    def iter_bytes(self, *args: Any, **kwargs: Any) -> Iterator[bytes]:
        return self._iter()

    # httpx async surface
    async def aiter_bytes(self, *args: Any, **kwargs: Any) -> Any:
        for chunk in self._chunk_source:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


def _drive_requests(response: FakeResponse, span: RecordingSpan) -> bytes:
    """Stream a response through the requests wrapper and return what the caller saw."""
    wrapper = RequestsStreamingWrapper(response=response, span=span)
    received = b"".join(wrapper.iter_content())
    wrapper.close()
    return received


def _drive_httpx(response: FakeResponse, span: RecordingSpan) -> bytes:
    """Stream a response through the httpx sync wrapper and return what the caller saw."""
    wrapper = HttpxStreamingWrapper(response=response, span=span)
    received = b"".join(wrapper.iter_bytes())
    wrapper.close()
    return received


def _drive_httpx_async(response: FakeResponse, span: RecordingSpan) -> bytes:
    """Stream a response through the httpx async wrapper and return what the caller saw."""
    wrapper = AsyncStreamingWrapper(response=response, span=span)

    async def consume() -> bytes:
        received = b""
        async for chunk in wrapper.aiter_bytes():
            received += chunk
        await wrapper.aclose()
        return received

    return asyncio.run(consume())


# Every wrapper runs the same capture -> parse -> serialize pipeline, so the
# behavioral tests below are asserted against all three rather than one.
Driver = Callable[[FakeResponse, RecordingSpan], bytes]

all_wrappers = pytest.mark.parametrize(
    "drive",
    [_drive_requests, _drive_httpx, _drive_httpx_async],
    ids=["requests", "httpx", "httpx-async"],
)


def _activate_limit(monkeypatch: pytest.MonkeyPatch, max_len: int) -> None:
    """Make ``get_attribute_max_len()`` return *max_len* for the current test."""
    monkeypatch.setenv("NETRA_ATTRIBUTE_MAX_LEN", str(max_len))
    set_active_config(Config())


def _span_output(span: RecordingSpan) -> Dict[str, Any]:
    """Parse the JSON ``output`` attribute the wrapper wrote."""
    return json.loads(span.attributes["output"])


class TestBoundedBodyBuffer:
    """The buffer retains a bounded prefix while counting the whole stream."""

    def test_retains_everything_when_body_is_under_the_cap(self):
        buffer = BoundedBodyBuffer(max_bytes=100)

        buffer.append(b"hello ")
        buffer.append(b"world")

        assert buffer.getvalue() == b"hello world"
        assert buffer.total_bytes == 11
        assert buffer.truncated is False

    @pytest.mark.parametrize(
        "chunk_size,chunk_count",
        [
            (1, 100),  # cap lands on a chunk boundary
            (7, 15),  # cap lands mid-chunk
            (1000, 1),  # a single chunk far larger than the cap
        ],
    )
    def test_retains_exactly_the_cap_and_counts_the_rest(self, chunk_size, chunk_count):
        buffer = BoundedBodyBuffer(max_bytes=10)

        for _ in range(chunk_count):
            buffer.append(b"a" * chunk_size)

        assert buffer.getvalue() == b"a" * 10
        assert buffer.total_bytes == chunk_size * chunk_count
        assert buffer.truncated is True

    def test_retention_is_flat_as_the_body_grows(self):
        small = BoundedBodyBuffer(max_bytes=64)
        large = BoundedBodyBuffer(max_bytes=64)

        for _ in range(10):
            small.append(b"x" * 1024)
        for _ in range(10_000):
            large.append(b"x" * 1024)

        assert len(small.getvalue()) == len(large.getvalue()) == 64
        assert large.total_bytes == 10_000 * 1024

    def test_truncated_multibyte_tail_is_dropped_so_text_still_decodes(self):
        # "€" is 3 bytes; a 10-byte cap over "abcdefgh€" cuts it after 2 of them.
        buffer = BoundedBodyBuffer(max_bytes=10)

        buffer.append("abcdefgh€x".encode("utf-8"))

        assert buffer.truncated is True
        assert buffer.getvalue().decode("utf-8") == "abcdefgh"

    def test_complete_multibyte_tail_is_kept(self):
        buffer = BoundedBodyBuffer(max_bytes=11)

        buffer.append("abcdefgh€x".encode("utf-8"))

        assert buffer.truncated is True
        assert buffer.getvalue().decode("utf-8") == "abcdefgh€"

    def test_untruncated_body_is_never_trimmed(self):
        payload = "€€€".encode("utf-8")
        buffer = BoundedBodyBuffer(max_bytes=len(payload))

        buffer.append(payload)

        assert buffer.truncated is False
        assert buffer.getvalue() == payload

    def test_str_chunks_are_counted_as_encoded_bytes(self):
        buffer = BoundedBodyBuffer(max_bytes=100)

        buffer.append("€")  # 1 character, 3 bytes

        assert buffer.total_bytes == 3
        assert buffer.getvalue() == "€".encode("utf-8")

    def test_bytearray_chunk_is_copied_not_aliased(self):
        buffer = BoundedBodyBuffer(max_bytes=100)
        chunk = bytearray(b"abc")

        buffer.append(chunk)
        chunk[0] = ord("z")

        assert buffer.getvalue() == b"abc"

    def test_non_bytes_chunk_is_ignored(self):
        buffer = BoundedBodyBuffer(max_bytes=100)

        buffer.append(None)  # type: ignore[arg-type]
        buffer.append(12345)  # type: ignore[arg-type]

        assert buffer.total_bytes == 0
        assert buffer.getvalue() == b""

    def test_zero_cap_retains_nothing_but_still_counts(self):
        buffer = BoundedBodyBuffer(max_bytes=0)

        buffer.append(b"payload")

        assert buffer.getvalue() == b""
        assert buffer.total_bytes == 7
        assert buffer.truncated is True

    def test_empty_buffer_reports_no_bytes(self):
        buffer = BoundedBodyBuffer(max_bytes=10)

        assert buffer.total_bytes == 0
        assert buffer.truncated is False

    def test_cap_defaults_to_configured_attribute_max_len_with_headroom(self, monkeypatch):
        _activate_limit(monkeypatch, 32)

        buffer = BoundedBodyBuffer()
        buffer.append(b"y" * 5000)

        assert len(buffer.getvalue()) == 32 * _PARSE_COMPACTION_HEADROOM

    def test_cap_defaults_to_sdk_default_before_init(self):
        assert config_module._active_config is None
        expected = _DEFAULT_ATTRIBUTE_MAX_LEN * _PARSE_COMPACTION_HEADROOM

        buffer = BoundedBodyBuffer()
        buffer.append(b"y" * (expected + 1000))

        assert len(buffer.getvalue()) == expected


class TestTruncatedRecordParsing:
    """A capture that stopped mid-record never records the partial record."""

    @pytest.mark.parametrize(
        "retained,expected",
        [
            (b'data: {"a": 1}\n\ndata: {"b"', {"a": 1}),  # half-written SSE frame
            (b'{"a": 1}\n{"b"', {"a": 1}),  # half-written NDJSON line
            (b'{"a": 1}\n\n', {"a": 1}),  # already on a boundary
        ],
    )
    def test_partial_trailing_record_is_dropped(self, retained, expected):
        parsed = parse_streaming_body(retained, total_bytes=10_000, truncated=True, budget=1_000)

        assert parsed.value == expected

    def test_body_with_no_record_boundary_is_kept_as_text(self):
        # Nothing to trim back to, so the fragment is recorded verbatim rather
        # than guessed at.
        retained = b'{"a": 1, "b"'

        parsed = parse_streaming_body(retained, total_bytes=10_000, truncated=True, budget=1_000)

        assert parsed.value == '{"a": 1, "b"'


class TestBudgetAwareParsing:
    """Parsing stops at the budget instead of building the whole buffer."""

    @all_wrappers
    def test_sse_events_beyond_the_budget_are_never_parsed(self, monkeypatch, drive: Driver):
        # The buffer retains max_len * headroom bytes; parsing all of it into
        # dicts is the largest allocation in the path, and it stacks when many
        # streams finish together.
        _activate_limit(monkeypatch, 1_000)
        span = RecordingSpan()
        event = b'data: {"choices": [{"delta": {"content": "token"}}]}\n\n'

        drive(FakeResponse([event * 5_000]), span)

        body = _span_output(span)["body"]
        retained_events = (1_000 * _PARSE_COMPACTION_HEADROOM) // len(event)
        # Far fewer entries than the buffer held -- the rest were never built.
        assert len(body) < retained_events // 2

    @all_wrappers
    def test_truncated_sse_records_no_partial_event(self, monkeypatch, drive: Driver):
        # The final retained frame is almost always half-written; it must not
        # show up as a junk string among the parsed events.
        _activate_limit(monkeypatch, 900)
        span = RecordingSpan()
        event = b'data: {"choices": [{"delta": {"content": "token"}}]}\n\n'

        drive(FakeResponse([event * 5_000]), span)

        body = _span_output(span)["body"]
        fragments = [entry for entry in body if isinstance(entry, str) and entry != TRUNCATION_ELLIPSIS]
        assert fragments == []
        assert body[-1] == TRUNCATION_ELLIPSIS

    @all_wrappers
    def test_parser_side_cut_still_raises_the_marker(self, monkeypatch, drive: Driver):
        # 1,120 bytes fits the capture buffer (400 * headroom) but the parser
        # stops at the 400-char budget, so the marker must come from the parser.
        _activate_limit(monkeypatch, 400)
        span = RecordingSpan()
        event = b'data: {"n": 1}\n\n'

        drive(FakeResponse([event * 70]), span)

        output = _span_output(span)
        assert output[TRUNCATION_MARKER_KEY] is True
        assert output["body"][-1] == TRUNCATION_ELLIPSIS


class TestStreamingCapture:
    """Each wrapper bounds what it records without touching what it yields."""

    @all_wrappers
    def test_caller_receives_every_byte_when_body_exceeds_the_cap(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 16)
        chunks = [bytes([i % 256]) * 1024 for i in range(64)]

        received = drive(FakeResponse(chunks), RecordingSpan())

        assert received == b"".join(chunks)

    @all_wrappers
    def test_recorded_body_is_bounded_by_the_cap(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 1_000)
        span = RecordingSpan()

        drive(FakeResponse([b"a" * 100_000]), span)

        output = _span_output(span)
        assert output[TRUNCATION_MARKER_KEY] is True
        assert output["body_bytes"] == 100_000
        assert output["body"].endswith(TRUNCATION_ELLIPSIS)
        assert set(output["body"][: -len(TRUNCATION_ELLIPSIS)]) == {"a"}
        assert len(span.attributes["output"]) <= 1_000

    @all_wrappers
    def test_truncation_flags_precede_the_body_so_they_survive_attribute_trimming(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 1_000)
        span = RecordingSpan()

        drive(FakeResponse([b"a" * 100_000]), span)

        serialized = span.attributes["output"]
        # InstrumentationSpanProcessor trims the tail of the serialized attribute;
        # the marker is only trustworthy if it appears before the body.
        assert serialized.index(f'"{TRUNCATION_MARKER_KEY}"') < serialized.index('"body":')
        assert serialized.index('"body_bytes"') < serialized.index('"body":')

    @all_wrappers
    def test_binary_placeholder_reports_the_streamed_size_not_the_retained_size(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 32)
        span = RecordingSpan()

        drive(FakeResponse([b"\xff\xfe" * 50_000]), span)

        assert _span_output(span)["body"] == "<binary content: 100000 bytes>"

    @all_wrappers
    def test_binary_placeholder_gets_no_ellipsis(self, monkeypatch, drive: Driver):
        # "<binary content: N bytes>" is a description, not a cut-off body; an
        # ellipsis would read as though the placeholder itself were truncated.
        _activate_limit(monkeypatch, 32)
        span = RecordingSpan()

        drive(FakeResponse([b"\xff\xfe" * 50_000]), span)

        output = _span_output(span)
        assert output["body"] == "<binary content: 100000 bytes>"
        assert output[TRUNCATION_MARKER_KEY] is True

    @all_wrappers
    def test_small_sse_body_is_recorded_unchanged(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)
        span = RecordingSpan()
        chunks = [b'data: {"delta": "hi"}\n\n', b'data: {"delta": "there"}\n\n', b"data: [DONE]\n\n"]

        drive(FakeResponse(chunks), span)

        output = _span_output(span)
        assert output["body"] == [{"delta": "hi"}, {"delta": "there"}]
        assert TRUNCATION_MARKER_KEY not in output

    @all_wrappers
    def test_truncated_sse_nearly_fills_the_attribute_budget(self, monkeypatch, drive: Driver):
        # Parsing strips SSE framing ("data: {...}\n\n" -> "{...}, "), so a buffer
        # holding exactly attribute_max_len bytes would serialize to far *less*
        # than the budget. The retention headroom is what keeps the trace full.
        max_len = 20_000
        _activate_limit(monkeypatch, max_len)
        span = RecordingSpan()
        event = b'data: {"choices": [{"delta": {"content": "tok"}}]}\n\n'

        drive(FakeResponse([event * 20_000]), span)

        serialized = span.attributes["output"]
        assert len(serialized) <= max_len
        assert len(serialized) > max_len * 0.9

    @all_wrappers
    def test_output_is_trimmed_here_so_the_ellipsis_survives_export(self, monkeypatch, drive: Driver):
        # InstrumentationSpanProcessor cuts any attribute at max_len. If the body
        # were left overflowing, that cut would take the ellipsis with it.
        max_len = 2_000
        _activate_limit(monkeypatch, max_len)
        span = RecordingSpan()

        drive(FakeResponse([b"z" * 500_000]), span)

        serialized = span.attributes["output"]
        assert len(serialized) <= max_len
        assert serialized.rstrip().endswith(f'{TRUNCATION_ELLIPSIS}"}}')

    @all_wrappers
    def test_body_within_the_capture_cap_but_over_the_budget_is_still_marked(self, monkeypatch, drive: Driver):
        # 400 bytes fits the capture buffer (100 * headroom) but not a 200-char
        # attribute, so the marker has to come from the serializer, not the buffer.
        _activate_limit(monkeypatch, 200)
        span = RecordingSpan()

        drive(FakeResponse([b"q" * 400]), span)

        output = _span_output(span)
        assert output[TRUNCATION_MARKER_KEY] is True
        assert output["body"].endswith(TRUNCATION_ELLIPSIS)

    @all_wrappers
    def test_sse_keep_alive_lines_are_not_recorded_as_empty_events(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)
        span = RecordingSpan()
        chunks = [b'data: {"delta": "hi"}\n\n', b"data:\n\n", b"data: [DONE]\n\n"]

        drive(FakeResponse(chunks), span)

        assert _span_output(span)["body"] == {"delta": "hi"}

    @all_wrappers
    def test_truncated_sse_events_end_with_an_ellipsis_element(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 400)
        span = RecordingSpan()
        event = b'data: {"i": 1}\n\n'

        drive(FakeResponse([event * 50]), span)

        body = _span_output(span)["body"]
        assert isinstance(body, list)
        assert body[-1] == TRUNCATION_ELLIPSIS
        assert body[0] == {"i": 1}
        assert len(span.attributes["output"]) <= 400

    @all_wrappers
    def test_untruncated_body_gets_no_ellipsis(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)
        span = RecordingSpan()

        drive(FakeResponse([b"short body"]), span)

        output = _span_output(span)
        assert output["body"] == "short body"
        assert TRUNCATION_MARKER_KEY not in output

    @all_wrappers
    def test_span_is_ended_once_the_stream_is_closed(self, monkeypatch, drive: Driver):
        _activate_limit(monkeypatch, 1_000)
        span = RecordingSpan()
        response = FakeResponse([b"chunk"])

        drive(response, span)

        assert response.closed is True
        assert span.ended is True
