"""Regression tests for the LLM instrumentation call paths (audit findings H-15, H-16, H-22).

These tests use fake SDK objects and an in-memory exporter, so they need none of the
provider SDKs installed.

- H-15: request values the SDKs pass for unset or model-typed parameters (``NOT_GIVEN``,
  ``omit``, ``Reasoning(...)``) must never break the user's call.
- H-16: LiteLLM Responses streams with function-call or reasoning output must not raise
  inside the user's loop, and all text must be kept.
- H-22: every stream wrapper exports exactly one span however the stream ends, marks
  mid-stream failures as ERROR, and returns itself from ``with``.
"""

import asyncio
import gc
import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from netra.instrumentation.capture.stream_utils import _force_finalize_inner_stream
from netra.instrumentation.libraries.cerebras import wrappers as cerebras_wrappers
from netra.instrumentation.libraries.google_genai import wrappers as genai_wrappers
from netra.instrumentation.libraries.groq import wrappers as groq_wrappers
from netra.instrumentation.libraries.litellm import wrappers as litellm_wrappers
from netra.instrumentation.libraries.openai import wrappers as openai_wrappers
from netra.instrumentation.llm_common import is_unset_sentinel, to_jsonable


# Fakes
class NotGiven:
    """Stand-in for ``openai.NOT_GIVEN`` (detected by class name)."""

    def __bool__(self) -> bool:
        return False


class Omit:
    """Stand-in for ``openai.omit`` (detected by class name)."""

    def __bool__(self) -> bool:
        return False


class Reasoning(BaseModel):
    effort: Optional[str] = None
    summary: Optional[str] = None


class FakeStream:
    """Sync stream shaped like a Stainless SDK ``Stream`` (``__enter__`` returns itself)."""

    def __init__(self, chunks: List[Any], error: Optional[Exception] = None) -> None:
        self._chunks = iter(chunks)
        self._error = error
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                raise self._error
            raise

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class FakeAsyncStream:
    """Async stream shaped like a Stainless SDK ``AsyncStream``."""

    def __init__(self, chunks: List[Any], error: Optional[Exception] = None) -> None:
        self._chunks = iter(chunks)
        self._error = error
        self.closed = False

    def __aiter__(self) -> "FakeAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                raise self._error
            raise StopAsyncIteration

    async def __aenter__(self) -> "FakeAsyncStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Any:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _chat_chunks() -> List[Dict[str, Any]]:
    return [
        {"model": "m-1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
        {"model": "m-1", "choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"model": "m-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


def _genai_chunks() -> List[Any]:
    return [SimpleNamespace(text="Hel"), SimpleNamespace(text="lo")]


CHAT_KWARGS = {"model": "m-1", "messages": [{"role": "user", "content": "hi"}], "stream": True}
GENAI_KWARGS = {"model": "gemini-x", "contents": "hi"}

# (name, sync wrapper factory, async wrapper factory, kwargs, chunk factory, span name)
PROVIDERS = [
    ("openai", openai_wrappers.chat_wrapper, openai_wrappers.achat_wrapper, CHAT_KWARGS, _chat_chunks, "openai.chat"),
    (
        "litellm",
        litellm_wrappers.completion_wrapper,
        litellm_wrappers.acompletion_wrapper,
        CHAT_KWARGS,
        _chat_chunks,
        "litellm.completion",
    ),
    ("groq", groq_wrappers.chat_wrapper, groq_wrappers.achat_wrapper, CHAT_KWARGS, _chat_chunks, "groq.chat"),
    (
        "cerebras",
        cerebras_wrappers.chat_wrapper,
        cerebras_wrappers.achat_wrapper,
        CHAT_KWARGS,
        _chat_chunks,
        "cerebras.chat.completions",
    ),
    (
        "google_genai",
        genai_wrappers.content_stream_wrapper,
        genai_wrappers.acontent_stream_wrapper,
        GENAI_KWARGS,
        _genai_chunks,
        "genai.generate_content_stream",
    ),
]
PROVIDER_IDS = [p[0] for p in PROVIDERS]


def _open_sync(tracer: Any, factory: Callable[..., Any], kwargs: Dict[str, Any], stream: Any) -> Any:
    return factory(tracer)(lambda *a, **k: stream, None, (), dict(kwargs))


def _open_async(tracer: Any, factory: Callable[..., Any], kwargs: Dict[str, Any], stream: Any) -> Any:
    async def wrapped(*args: Any, **kw: Any) -> Any:
        return stream

    return factory(tracer)(wrapped, None, (), dict(kwargs))


def _only_span(exporter: InMemorySpanExporter, name: str) -> Any:
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


# H-22: sync stream lifecycle
@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
class TestSyncStreamLifecycle:
    def test_full_consumption_exports_one_ok_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, sync_factory, _, kwargs, chunks, span_name = provider
        stream = _open_sync(tracer, sync_factory, kwargs, FakeStream(chunks()))

        received = list(stream)

        assert len(received) == len(chunks())
        span = _only_span(exporter, span_name)
        assert span.status.status_code == StatusCode.OK
        assert stream._netra_output == "Hello"

    def test_break_then_close_exports_one_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, sync_factory, _, kwargs, chunks, span_name = provider
        inner = FakeStream(chunks())
        stream = _open_sync(tracer, sync_factory, kwargs, inner)

        for _ in stream:
            break
        assert exporter.get_finished_spans() == ()
        stream.close()
        stream.close()  # idempotent

        _only_span(exporter, span_name)
        assert inner.closed

    def test_break_then_garbage_collection_exports_one_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, sync_factory, _, kwargs, chunks, span_name = provider
        stream = _open_sync(tracer, sync_factory, kwargs, FakeStream(chunks()))

        for _ in stream:
            break
        del stream
        gc.collect()

        _only_span(exporter, span_name)

    def test_mid_stream_error_exports_one_error_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, sync_factory, _, kwargs, chunks, span_name = provider
        stream = _open_sync(tracer, sync_factory, kwargs, FakeStream(chunks(), error=ConnectionError("dropped")))

        with pytest.raises(ConnectionError, match="dropped"):
            for _ in stream:
                pass

        span = _only_span(exporter, span_name)
        assert span.status.status_code == StatusCode.ERROR
        assert any(e.name == "exception" for e in span.events)

    def test_with_block_yields_wrapper_and_exports_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, sync_factory, _, kwargs, chunks, span_name = provider
        inner = FakeStream(chunks())
        stream = _open_sync(tracer, sync_factory, kwargs, inner)

        with stream as entered:
            assert entered is stream
            next(iter(entered))

        span = _only_span(exporter, span_name)
        assert span.status.status_code == StatusCode.OK
        assert inner.closed

    def test_force_finalize_after_break_ends_span_once(self, provider: Any, tracer: Any, exporter: Any) -> None:
        """``stream_utils`` finalises an abandoned inner stream; the span must still end only once."""
        _, sync_factory, _, kwargs, chunks, span_name = provider
        stream = _open_sync(tracer, sync_factory, kwargs, FakeStream(chunks()))

        iterator = iter(stream)
        next(iterator)
        assert stream._netra_output is None
        _force_finalize_inner_stream(iterator, stream)
        stream._finalize_span()

        _only_span(exporter, span_name)
        assert stream._netra_output == "Hel"


# H-22: async stream lifecycle
@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
class TestAsyncStreamLifecycle:
    def test_full_consumption_exports_one_ok_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, _, async_factory, kwargs, chunks, span_name = provider

        async def run() -> List[Any]:
            stream = await _open_async(tracer, async_factory, kwargs, FakeAsyncStream(chunks()))
            return [c async for c in stream]

        assert len(asyncio.run(run())) == len(chunks())
        assert _only_span(exporter, span_name).status.status_code == StatusCode.OK

    def test_break_then_close_exports_one_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, _, async_factory, kwargs, chunks, span_name = provider
        inner = FakeAsyncStream(chunks())

        async def run() -> None:
            stream = await _open_async(tracer, async_factory, kwargs, inner)
            async for _ in stream:
                break
            assert exporter.get_finished_spans() == ()
            await stream.close()
            await stream.aclose()

        asyncio.run(run())
        _only_span(exporter, span_name)
        assert inner.closed

    def test_break_then_garbage_collection_exports_one_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, _, async_factory, kwargs, chunks, span_name = provider

        async def run() -> None:
            stream = await _open_async(tracer, async_factory, kwargs, FakeAsyncStream(chunks()))
            async for _ in stream:
                break
            del stream
            gc.collect()

        asyncio.run(run())
        _only_span(exporter, span_name)

    def test_mid_stream_error_exports_one_error_span(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, _, async_factory, kwargs, chunks, span_name = provider

        async def run() -> None:
            stream = await _open_async(
                tracer, async_factory, kwargs, FakeAsyncStream(chunks(), error=ConnectionError("dropped"))
            )
            async for _ in stream:
                pass

        with pytest.raises(ConnectionError, match="dropped"):
            asyncio.run(run())
        assert _only_span(exporter, span_name).status.status_code == StatusCode.ERROR

    def test_async_with_block_yields_wrapper(self, provider: Any, tracer: Any, exporter: Any) -> None:
        _, _, async_factory, kwargs, chunks, span_name = provider
        inner = FakeAsyncStream(chunks())

        async def run() -> None:
            stream = await _open_async(tracer, async_factory, kwargs, inner)
            async with stream as entered:
                assert entered is stream
                await entered.__anext__()

        asyncio.run(run())
        assert _only_span(exporter, span_name).status.status_code == StatusCode.OK
        assert inner.closed


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
def test_telemetry_errors_never_reach_the_users_loop(provider: Any, tracer: Any, exporter: Any) -> None:
    """A malformed chunk (e.g. ``"index": null``, or a chunk that isn't a dict) is skipped, not raised."""
    name, sync_factory, _, kwargs, chunks, span_name = provider
    weird = [{"choices": [{"index": None, "delta": {"content": "x"}}]}, {"choices": "nope"}, object()]
    stream = _open_sync(tracer, sync_factory, kwargs, FakeStream(weird + chunks()))

    assert len(list(stream)) == len(weird) + len(chunks())
    assert _only_span(exporter, span_name).status.status_code == StatusCode.OK


def test_failure_before_stream_starts_propagates_and_ends_span(tracer: Any, exporter: Any) -> None:
    """Cerebras used to swallow this and return None to the caller."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError, match="rate limited"):
        cerebras_wrappers.chat_wrapper(tracer)(wrapped, None, (), dict(CHAT_KWARGS))
    assert _only_span(exporter, "cerebras.chat.completions").status.status_code == StatusCode.ERROR


def test_groq_non_streaming_error_is_not_replaced(tracer: Any, exporter: Any) -> None:
    """Groq used to replace the user's exception with UnboundLocalError."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError, match="rate limited"):
        groq_wrappers.chat_wrapper(tracer)(wrapped, None, (), {"model": "m", "messages": []})
    assert _only_span(exporter, "groq.chat").status.status_code == StatusCode.ERROR


# H-15: request values that must not break the call
SENTINELS = [
    pytest.param(NotGiven(), None, id="NOT_GIVEN"),
    pytest.param(Omit(), None, id="omit"),
    pytest.param(Reasoning(effort="low"), {"effort": "low"}, id="Reasoning-model"),
    pytest.param({"effort": "low"}, {"effort": "low"}, id="dict"),
]


def _responses_completed_event() -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "model": "gpt-x",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }


@pytest.mark.parametrize("reasoning,expected", SENTINELS)
@pytest.mark.parametrize("streaming", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_openai_responses_accepts_sentinels_and_models(
    reasoning: Any, expected: Any, streaming: bool, is_async: bool, tracer: Any, exporter: Any
) -> None:
    kwargs = {
        "model": "gpt-x",
        "input": "hi",
        "reasoning": reasoning,
        "temperature": NotGiven(),
        "max_output_tokens": Omit(),
        "stream": streaming,
    }
    calls = []

    if is_async:

        async def run() -> Any:
            async def wrapped(*a: Any, **k: Any) -> Any:
                calls.append(k)
                return FakeAsyncStream([_responses_completed_event()]) if streaming else {"output_text": "ok"}

            result = await openai_wrappers.aresponses_wrapper(tracer)(wrapped, None, (), kwargs)
            if streaming:
                return [e async for e in result]
            return result

        result = asyncio.run(run())
    else:

        def wrapped(*a: Any, **k: Any) -> Any:
            calls.append(k)
            return FakeStream([_responses_completed_event()]) if streaming else {"output_text": "ok"}

        result = openai_wrappers.responses_wrapper(tracer)(wrapped, None, (), kwargs)
        if streaming:
            result = list(result)

    assert len(calls) == 1, "the request must reach the SDK"
    assert result
    span = _only_span(exporter, "openai.response")
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    if expected is None:
        assert "llm.request.reasoning_effort" not in attributes
    else:
        assert json.loads(attributes["llm.request.reasoning_effort"]) == expected
    assert "llm.request.temperature" not in attributes
    assert "llm.request.max_tokens" not in attributes


def test_openai_attribute_failure_does_not_fail_the_call(
    tracer: Any, exporter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise TypeError("telemetry bug")

    monkeypatch.setattr(openai_wrappers, "set_request_attributes", boom)
    monkeypatch.setattr(openai_wrappers, "set_response_attributes", boom)

    result = openai_wrappers.chat_wrapper(tracer)(
        lambda *a, **k: {"choices": []}, None, (), {"model": "m", "messages": []}
    )

    assert result == {"choices": []}
    assert _only_span(exporter, "openai.chat").status.status_code == StatusCode.OK


def test_sentinel_helpers() -> None:
    assert is_unset_sentinel(NotGiven()) and is_unset_sentinel(Omit())
    assert not is_unset_sentinel(None) and not is_unset_sentinel(0)
    assert to_jsonable(NotGiven()) is None
    assert to_jsonable(Reasoning(effort="high")) == {"effort": "high"}
    assert to_jsonable({"a": 1}) == {"a": 1}
    assert to_jsonable(object()).startswith("<object object")


# H-16: LiteLLM Responses streams
def _litellm_completed(output: List[Dict[str, Any]], status: str = "completed") -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"status": status, "model": "gpt-x", "output": output, "usage": {"input_tokens": 3}},
    }


FUNCTION_CALL = {"type": "function_call", "name": "get_weather", "arguments": '{"city": "Paris"}', "call_id": "c1"}
REASONING = {"type": "reasoning", "content": None, "summary": [{"type": "summary_text", "text": "thinking"}]}


def _message(*texts: str) -> Dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t} for t in texts]}


LITELLM_CASES = [
    pytest.param([FUNCTION_CALL, _message("done")], "done", True, id="function_call-then-message"),
    pytest.param([REASONING, _message("answer")], "answer", False, id="reasoning-item"),
    pytest.param([_message("first "), _message("second")], "first second", False, id="two-message-items"),
    pytest.param([_message("A", "B")], "AB", False, id="two-text-parts"),
]


def _completion_attributes(span: Any) -> Dict[str, Any]:
    return {k: v for k, v in (span.attributes or {}).items() if k.startswith("gen_ai.completion")}


@pytest.mark.parametrize("output,expected_text,has_function_call", LITELLM_CASES)
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_litellm_responses_stream_handles_all_output_items(
    output: List[Dict[str, Any]],
    expected_text: str,
    has_function_call: bool,
    is_async: bool,
    tracer: Any,
    exporter: Any,
) -> None:
    events = [{"type": "response.output_text.delta", "delta": "x"}, _litellm_completed(output)]
    kwargs = {"model": "gpt-x", "input": "hi", "stream": True}

    if is_async:

        async def run() -> List[Any]:
            stream = await _open_async(tracer, litellm_wrappers.aresponses_wrapper, kwargs, FakeAsyncStream(events))
            return [e async for e in stream]

        received = asyncio.run(run())
    else:
        received = list(_open_sync(tracer, litellm_wrappers.responses_wrapper, kwargs, FakeStream(events)))

    assert len(received) == len(events)
    span = _only_span(exporter, "litellm.responses")
    assert span.status.status_code == StatusCode.OK
    completions = _completion_attributes(span)
    contents = [v for k, v in completions.items() if k.endswith(".content")]
    assert expected_text in contents
    assert "thinking" not in "".join(contents)
    if has_function_call:
        assert {"name": "get_weather", "arguments": '{"city": "Paris"}'} in [
            json.loads(c) for c in contents if c.startswith("{")
        ]


def test_litellm_responses_stream_accepts_incomplete_status(tracer: Any, exporter: Any) -> None:
    events = [_litellm_completed([_message("partial")], status="incomplete")]
    stream = _open_sync(
        tracer, litellm_wrappers.responses_wrapper, {"model": "m", "input": "hi", "stream": True}, FakeStream(events)
    )
    list(stream)

    contents = _completion_attributes(_only_span(exporter, "litellm.responses")).values()
    assert "partial" in contents
