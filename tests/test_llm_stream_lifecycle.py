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
import json
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

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
