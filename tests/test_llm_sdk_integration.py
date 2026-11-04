"""End-to-end checks against the real provider SDKs (H-15, H-22).

Each test is skipped unless the SDK is installed. Requests go to an ``httpx.MockTransport``,
never the network.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

httpx = pytest.importorskip("httpx")

_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


def _sse(events: List[Dict[str, Any]], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def _transport(handler: Callable[[Any], Any]) -> Any:
    requests: List[Any] = []

    def record(request: Any) -> Any:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(record)
    transport.requests = requests  # type: ignore[attr-defined]
    return transport


def _spans(exporter: InMemorySpanExporter, name: str) -> List[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


CHAT_CHUNKS = [
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m-1",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
    },
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m-1",
        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
    },
]


# OpenAI
@pytest.fixture(scope="module")
def openai_mod() -> Any:
    openai = pytest.importorskip("openai")
    from netra.instrumentation.libraries.openai import NetraOpenAIInstrumentor

    NetraOpenAIInstrumentor().instrument(tracer_provider=_PROVIDER)
    return openai


RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-x",
    "output": [
        {
            "type": "message",
            "id": "m1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def _responses_handler(request: Any) -> Any:
    if json.loads(request.content).get("stream"):
        events = [
            {
                "type": "response.output_text.delta",
                "item_id": "m1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hi",
                "sequence_number": 1,
                "logprobs": [],
            },
            {"type": "response.completed", "response": RESPONSE, "sequence_number": 2},
        ]
        return httpx.Response(200, content=_sse(events, done=False), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=RESPONSE)


def _reasoning_values(openai: Any) -> List[Any]:
    from openai.types.shared import Reasoning

    values = [openai.NOT_GIVEN, Reasoning(effort="low")]
    if hasattr(openai, "omit"):
        values.append(openai.omit)
    return values


@pytest.mark.parametrize("streaming", [False, True], ids=["non-stream", "stream"])
def test_openai_responses_with_reasoning_sentinels_sync(openai_mod: Any, exporter: Any, streaming: bool) -> None:
    transport = _transport(_responses_handler)
    client = openai_mod.OpenAI(api_key="x", http_client=httpx.Client(transport=transport))

    for reasoning in _reasoning_values(openai_mod):
        result = client.responses.create(model="gpt-x", input="hi", reasoning=reasoning, stream=streaming)
        if streaming:
            list(result)

    assert len(transport.requests) == len(_reasoning_values(openai_mod))
    spans = _spans(exporter, "openai.response")
    assert len(spans) == len(_reasoning_values(openai_mod))
    assert all(s.status.status_code == StatusCode.OK for s in spans)


@pytest.mark.parametrize("streaming", [False, True], ids=["non-stream", "stream"])
def test_openai_responses_with_reasoning_sentinels_async(openai_mod: Any, exporter: Any, streaming: bool) -> None:
    transport = _transport(_responses_handler)

    async def run() -> None:
        client = openai_mod.AsyncOpenAI(api_key="x", http_client=httpx.AsyncClient(transport=transport))
        for reasoning in _reasoning_values(openai_mod):
            result = await client.responses.create(model="gpt-x", input="hi", reasoning=reasoning, stream=streaming)
            if streaming:
                async for _ in result:
                    pass

    asyncio.run(run())
    spans = _spans(exporter, "openai.response")
    assert len(spans) == len(_reasoning_values(openai_mod))
    assert all(s.status.status_code == StatusCode.OK for s in spans)


def test_openai_mid_stream_error_is_reported_as_error(openai_mod: Any, exporter: Any) -> None:
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self) -> Any:
            yield _sse(CHAT_CHUNKS[:1], done=False)
            raise httpx.ReadError("connection dropped")

    transport = _transport(
        lambda request: httpx.Response(200, stream=BrokenStream(), headers={"content-type": "text/event-stream"})
    )
    client = openai_mod.OpenAI(api_key="x", http_client=httpx.Client(transport=transport))

    stream = client.chat.completions.create(model="m-1", messages=[{"role": "user", "content": "hi"}], stream=True)
    with pytest.raises(httpx.ReadError):
        for _ in stream:
            pass

    spans = _spans(exporter, "openai.chat")
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


# Groq and Cerebras: ``with client.chat.completions.create(stream=True) as s``
def _chat_stream_transport() -> Any:
    return _transport(
        lambda request: httpx.Response(200, content=_sse(CHAT_CHUNKS), headers={"content-type": "text/event-stream"})
    )


def test_groq_with_block_is_instrumented(exporter: Any) -> None:
    groq = pytest.importorskip("groq")
    from netra.instrumentation.libraries.groq import NetraGroqInstrumentor
    from netra.instrumentation.libraries.groq.wrappers import StreamingWrapper

    NetraGroqInstrumentor().instrument(tracer_provider=_PROVIDER)
    client = groq.Groq(api_key="x", http_client=httpx.Client(transport=_chat_stream_transport()))

    with client.chat.completions.create(model="m-1", messages=[{"role": "user", "content": "hi"}], stream=True) as s:
        assert isinstance(s, StreamingWrapper)
        text = "".join(c.choices[0].delta.content or "" for c in s)

    assert text == "Hello"
    spans = _spans(exporter, "groq.chat")
    assert len(spans) == 1 and spans[0].status.status_code == StatusCode.OK


def test_cerebras_with_block_and_early_break(exporter: Any) -> None:
    cerebras_sdk = pytest.importorskip("cerebras.cloud.sdk")
    from netra.instrumentation.libraries.cerebras import NetraCerebrasInstrumentor
    from netra.instrumentation.libraries.cerebras.wrappers import StreamingWrapper

    NetraCerebrasInstrumentor().instrument(tracer_provider=_PROVIDER)
    client = cerebras_sdk.Cerebras(
        api_key="x", http_client=httpx.Client(transport=_chat_stream_transport()), warm_tcp_connection=False
    )

    with client.chat.completions.create(model="m-1", messages=[{"role": "user", "content": "hi"}], stream=True) as s:
        assert isinstance(s, StreamingWrapper)
        for _ in s:
            break

    spans = _spans(exporter, "cerebras.chat.completions")
    assert len(spans) == 1 and spans[0].status.status_code == StatusCode.OK
