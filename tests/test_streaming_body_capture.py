"""
Unit tests for bounded HTTP body capture and budgeted serialization.

The ``requests`` and ``httpx`` streaming wrappers tee every chunk the caller
reads so the body can be recorded on the span.  They used to retain the whole
body, which made tracing a large download cost several times the download's size
in RAM even though the exported attribute is capped at ``attribute_max_len``.
Capture now stops at that same limit, and non-streaming bodies go through the
same bound rather than being parsed and re-serialized whole.

These tests pin the invariants that matter:

    1. The caller still receives every byte, unaltered, no matter how large the
       body is.  Truncation applies to what Netra records, never to what the
       application reads.
    2. Peak retention is bounded by ``attribute_max_len`` and the recorded span
       still reports the *real* body size.
    3. The serialized attribute fits the budget and stays parseable whatever
       shape the body parses into -- not just the shapes someone thought to
       write a case for.  ``TestBudgetInvariantAcrossArbitraryShapes``
       generates the rest.

All three wrappers (requests, httpx sync, httpx async) share one pipeline, so
the behavioral tests run against each of them.
"""

import asyncio
import json
import random
import string
from typing import Any, Callable, Dict, Iterator, List

import pytest

from netra import config as config_module
from netra.config import _DEFAULT_ATTRIBUTE_MAX_LEN, Config, set_active_config
from netra.instrumentation.capture.bounded_capture import (
    TRUNCATION_ELLIPSIS,
    TRUNCATION_MARKER_KEY,
    BoundedStreamBuffer,
    BoundedValue,
    serialize_within_budget,
)
from netra.instrumentation.capture.stream_formats import parse_streaming_body
from netra.instrumentation.http.body import (
    _PARSE_COMPACTION_HEADROOM,
    build_response_output,
    new_body_buffer,
)
from netra.instrumentation.libraries.httpx.wrappers import AsyncStreamingWrapper
from netra.instrumentation.libraries.httpx.wrappers import StreamingWrapper as HttpxStreamingWrapper
from netra.instrumentation.libraries.requests.wrappers import StreamingWrapper as RequestsStreamingWrapper

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


class TestBoundedStreamBuffer:
    """The buffer retains a bounded prefix while counting the whole stream."""

    def test_retains_everything_when_body_is_under_the_cap(self):
        buffer = BoundedStreamBuffer(max_bytes=100)

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
        buffer = BoundedStreamBuffer(max_bytes=10)

        for _ in range(chunk_count):
            buffer.append(b"a" * chunk_size)

        assert buffer.getvalue() == b"a" * 10
        assert buffer.total_bytes == chunk_size * chunk_count
        assert buffer.truncated is True

    def test_retention_is_flat_as_the_body_grows(self):
        small = BoundedStreamBuffer(max_bytes=64)
        large = BoundedStreamBuffer(max_bytes=64)

        for _ in range(10):
            small.append(b"x" * 1024)
        for _ in range(10_000):
            large.append(b"x" * 1024)

        assert len(small.getvalue()) == len(large.getvalue()) == 64
        assert large.total_bytes == 10_000 * 1024

    def test_truncated_multibyte_tail_is_dropped_so_text_still_decodes(self):
        # "€" is 3 bytes; a 10-byte cap over "abcdefgh€" cuts it after 2 of them.
        buffer = BoundedStreamBuffer(max_bytes=10)

        buffer.append("abcdefgh€x".encode("utf-8"))

        assert buffer.truncated is True
        assert buffer.getvalue().decode("utf-8") == "abcdefgh"

    def test_complete_multibyte_tail_is_kept(self):
        buffer = BoundedStreamBuffer(max_bytes=11)

        buffer.append("abcdefgh€x".encode("utf-8"))

        assert buffer.truncated is True
        assert buffer.getvalue().decode("utf-8") == "abcdefgh€"

    def test_untruncated_body_is_never_trimmed(self):
        payload = "€€€".encode("utf-8")
        buffer = BoundedStreamBuffer(max_bytes=len(payload))

        buffer.append(payload)

        assert buffer.truncated is False
        assert buffer.getvalue() == payload

    def test_str_chunks_are_counted_as_encoded_bytes(self):
        buffer = BoundedStreamBuffer(max_bytes=100)

        buffer.append("€")  # 1 character, 3 bytes

        assert buffer.total_bytes == 3
        assert buffer.getvalue() == "€".encode("utf-8")

    def test_bytearray_chunk_is_copied_not_aliased(self):
        buffer = BoundedStreamBuffer(max_bytes=100)
        chunk = bytearray(b"abc")

        buffer.append(chunk)
        chunk[0] = ord("z")

        assert buffer.getvalue() == b"abc"

    def test_non_bytes_chunk_is_ignored(self):
        buffer = BoundedStreamBuffer(max_bytes=100)

        buffer.append(None)  # type: ignore[arg-type]
        buffer.append(12345)  # type: ignore[arg-type]

        assert buffer.total_bytes == 0
        assert buffer.getvalue() == b""

    def test_zero_cap_retains_nothing_but_still_counts(self):
        buffer = BoundedStreamBuffer(max_bytes=0)

        buffer.append(b"payload")

        assert buffer.getvalue() == b""
        assert buffer.total_bytes == 7
        assert buffer.truncated is True

    def test_empty_buffer_reports_no_bytes(self):
        buffer = BoundedStreamBuffer(max_bytes=10)

        assert buffer.total_bytes == 0
        assert buffer.truncated is False

    def test_cap_defaults_to_configured_attribute_max_len_with_headroom(self, monkeypatch):
        _activate_limit(monkeypatch, 32)

        buffer = new_body_buffer()
        buffer.append(b"y" * 5000)

        assert len(buffer.getvalue()) == 32 * _PARSE_COMPACTION_HEADROOM

    def test_cap_defaults_to_sdk_default_before_init(self):
        assert config_module._active_config is None
        expected = _DEFAULT_ATTRIBUTE_MAX_LEN * _PARSE_COMPACTION_HEADROOM

        buffer = new_body_buffer()
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


class TestBudgetIsAlwaysHonored:
    """Whatever shape the body parses into, the serialized value fits the budget.

    The shrinker used to handle only strings and multi-entry lists. Anything
    else -- most importantly a single JSON document, which is what a plain
    ``application/json`` response streamed via ``iter_content`` parses to --
    returned unshrunk and blew the budget by whatever the body happened to
    weigh. ``InstrumentationSpanProcessor`` then sliced the attribute at a
    fixed length, leaving unparseable JSON with the ellipsis cut off.
    """

    @staticmethod
    def _serialize(envelope, value, max_len: int) -> str:
        return serialize_within_budget(envelope, BoundedValue(value, truncated=True), max_len=max_len)

    @all_wrappers
    def test_single_json_document_over_budget_is_shrunk_not_passed_through(self, monkeypatch, drive: Driver):
        # Small enough to fit the capture buffer whole, so the buffer reports
        # truncated=False and only the serializer can enforce the budget.
        _activate_limit(monkeypatch, 2_000)
        document = json.dumps({"items": [{"id": i, "v": "z" * 100} for i in range(60)]}).encode()
        assert len(document) < 2_000 * _PARSE_COMPACTION_HEADROOM
        span = RecordingSpan()

        drive(FakeResponse([document]), span)

        serialized = span.attributes["output"]
        assert len(serialized) <= 2_000
        json.loads(serialized)  # still parseable -- no mid-token slice
        assert _span_output(span)[TRUNCATION_MARKER_KEY] is True

    def test_dict_body_is_shrunk_when_the_envelope_pushes_it_over(self):
        envelope = {"status_code": 200, "headers": {"x-request-id": "r" * 300}}
        body = {"summary": "a" * 900, "code": 7}

        serialized = self._serialize(envelope, body, max_len=1_000)

        assert len(serialized) <= 1_000
        recovered = json.loads(serialized)
        assert recovered[TRUNCATION_MARKER_KEY] is True
        # The widest value gave up characters; the narrow one stayed.
        assert recovered["body"]["code"] == 7
        assert len(recovered["body"]["summary"]) < 900

    def test_nested_structure_shrinks_into_its_widest_branch(self):
        body = {"meta": {"id": "abc"}, "rows": [{"text": "q" * 400} for _ in range(5)]}

        serialized = self._serialize({}, body, max_len=600)

        assert len(serialized) <= 600
        recovered = json.loads(serialized)["body"]
        assert recovered["meta"] == {"id": "abc"}
        assert len(recovered["rows"]) < 5

    def test_flat_scalar_dict_drops_entries_when_nothing_can_shrink(self):
        body = {f"k{i}": i for i in range(200)}

        serialized = self._serialize({}, body, max_len=300)

        assert len(serialized) <= 300
        recovered = json.loads(serialized)["body"]
        assert 0 < len(recovered) < 200
        assert recovered["k0"] == 0  # entries are dropped from the tail

    def test_a_scalar_body_is_left_alone_when_it_already_fits(self):
        serialized = self._serialize({"status_code": 200}, 42, max_len=100)

        assert json.loads(serialized)["body"] == 42

    def test_oversized_envelope_returns_the_smallest_rendering_not_the_fullest(self):
        # The envelope alone busts the budget, so nothing this layer does can
        # fit it. It must still not hand back the *largest* candidate.
        envelope = {"headers": {"h": "x" * 5_000}}

        serialized = self._serialize(envelope, ["entry" * 50] * 40, max_len=1_000)
        unshrunk = self._serialize(envelope, ["entry" * 50] * 40, max_len=10_000_000)

        assert len(serialized) < len(unshrunk)
        assert json.loads(serialized)[TRUNCATION_MARKER_KEY] is True


class TestNonStreamingBodiesAreBoundedToo:
    """A body the HTTP library already holds is still bounded before recording.

    "Already in memory" is not "free to record": parsing and re-serializing a
    200 MB response so the exporter can keep 50,000 characters of it is Netra's
    own allocation on top of the library's.
    """

    def test_large_body_is_recorded_within_budget_and_marked(self, monkeypatch):
        _activate_limit(monkeypatch, 1_000)
        body = json.dumps([{"i": i, "v": "y" * 80} for i in range(5_000)]).encode()

        serialized = build_response_output({"status_code": 200}, body)

        assert len(serialized) <= 1_000
        recovered = json.loads(serialized)
        assert recovered[TRUNCATION_MARKER_KEY] is True
        assert recovered["body_bytes"] == len(body)

    def test_parsing_never_sees_more_than_the_capture_cap(self, monkeypatch):
        _activate_limit(monkeypatch, 500)
        cap = 500 * _PARSE_COMPACTION_HEADROOM
        body = b"y" * (cap * 20)

        buffer = new_body_buffer()
        buffer.append(body)

        assert len(buffer.getvalue()) == cap
        assert buffer.total_bytes == len(body)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (b'{"key": "value"}', {"key": "value"}),
            (b"plain text", "plain text"),
            ("decoded text", "decoded text"),
        ],
    )
    def test_small_bodies_round_trip_unchanged(self, monkeypatch, raw, expected):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)

        recovered = json.loads(build_response_output({"status_code": 200}, raw))

        assert recovered["body"] == expected
        assert TRUNCATION_MARKER_KEY not in recovered

    @pytest.mark.parametrize("empty", [None, b"", ""])
    def test_a_bodiless_response_omits_the_body_key(self, monkeypatch, empty):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)

        recovered = json.loads(build_response_output({"status_code": 204}, empty))

        assert "body" not in recovered
        assert recovered == {"status_code": 204}

    def test_binary_body_is_described_rather_than_decoded(self, monkeypatch):
        _activate_limit(monkeypatch, _DEFAULT_ATTRIBUTE_MAX_LEN)

        recovered = json.loads(build_response_output({}, bytes(range(256))))

        assert recovered["body"] == "<binary content: 256 bytes>"


class TestCaptureHeadroomLimitation:
    """The parse headroom is a heuristic, and this pins where it falls short.

    ``_PARSE_COMPACTION_HEADROOM`` assumes retained bytes mostly become exported
    characters. A stream padded with framing the SSE parser discards outright
    breaks that assumption, and the attribute lands *under* budget. This is a
    documented trade-off, not a bug -- the test exists so that a change to the
    factor shows up as a deliberate edit rather than a silent drift.
    """

    def test_framing_the_parser_discards_leaves_budget_unused(self, monkeypatch):
        _activate_limit(monkeypatch, 5_000)
        # Every event spends ~55 bytes of capture to yield ~12 characters.
        event = b'event: ping\n:keepalive-padding-comment\ndata: {"t":1}\n\n'
        span = RecordingSpan()

        _drive_httpx(FakeResponse([event * 4_000]), span)

        serialized = span.attributes["output"]
        assert len(serialized) <= 5_000
        # Well short of the budget, purely because of discarded framing.
        assert len(serialized) < 5_000 * 0.8


class TestTextBufferAccess:
    """A text producer can use the same buffer without hand-rolling decoding."""

    def test_text_chunks_round_trip_through_gettext(self):
        buffer = BoundedStreamBuffer(max_bytes=1_000)

        for chunk in ("héllo ", "wörld"):
            buffer.append(chunk)

        assert buffer.gettext() == "héllo wörld"
        assert buffer.truncated is False

    def test_a_multibyte_character_is_never_split_across_the_cap(self):
        # "é" is two bytes; a cap of 5 lands mid-character on the third one.
        buffer = BoundedStreamBuffer(max_bytes=5)

        buffer.append("ééé")

        assert buffer.gettext() == "éé"
        assert buffer.truncated is True


class TestBudgetInvariantAcrossArbitraryShapes:
    """The budget holds for payload shapes nobody thought to write a case for.

    The shrinker was rewritten twice during review because each hand-picked
    example passed while a neighbouring shape blew the budget by 200 KB: an
    averaged entry size collapses on a list holding one fat entry among many
    small, and shrinking one dict value per round needs one round per fat key.
    This generates the neighbours.
    """

    @staticmethod
    def _random_value(rng: random.Random, depth: int = 0) -> Any:
        kinds = ["str", "int", "list", "dict", "str", "list"] if depth < 3 else ["str", "int"]
        kind = rng.choice(kinds)
        if kind == "str":
            return "".join(rng.choice(string.printable[:70]) for _ in range(rng.randint(0, 400)))
        if kind == "int":
            return rng.randint(-(10**6), 10**6)
        if kind == "list":
            return [
                TestBudgetInvariantAcrossArbitraryShapes._random_value(rng, depth + 1)
                for _ in range(rng.randint(0, 30))
            ]
        return {
            f"k{i}": TestBudgetInvariantAcrossArbitraryShapes._random_value(rng, depth + 1)
            for i in range(rng.randint(0, 15))
        }

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_output_fits_the_budget_and_stays_parseable(self, seed):
        rng = random.Random(seed)

        for _ in range(60):
            envelope = {"status_code": 200}
            if rng.random() < 0.25:
                envelope["headers"] = {"h": "x" * rng.randint(0, 3_000)}
            payload = BoundedValue(
                self._random_value(rng),
                truncated=rng.random() < 0.5,
                total_size=rng.randint(0, 10**6),
            )
            max_len = rng.choice([200, 1_000, 5_000])

            serialized = serialize_within_budget(envelope, payload, max_len=max_len)

            json.loads(serialized)  # never a mid-token slice
            # The only permitted overshoot is an envelope that leaves no room
            # for even an elided body -- no shorter output exists.
            floor = len(
                json.dumps({**envelope, TRUNCATION_MARKER_KEY: True, "body_bytes": payload.total_size, "body": "..."})
            )
            assert len(serialized) <= max_len or floor >= max_len

    def test_a_list_whose_weight_sits_in_one_entry_still_fits(self):
        # The averaged-size shrinker judged every entry droppable here and then
        # tried to shrink into the first, which was an empty dict.
        body = [{}, 1, 2, ["z" * 15_000], 3, "tail"]

        serialized = serialize_within_budget({}, BoundedValue(body, truncated=True), max_len=200)

        assert len(serialized) <= 200
        json.loads(serialized)

    def test_a_dict_of_many_fat_values_fits_without_exhausting_the_rounds(self):
        # One value shrunk per round needed 20 rounds; the limit is 8.
        body = {f"field{i}": "q" * 400 for i in range(20)}

        serialized = serialize_within_budget({}, BoundedValue(body, truncated=True), max_len=300)

        assert len(serialized) <= 300
        json.loads(serialized)
