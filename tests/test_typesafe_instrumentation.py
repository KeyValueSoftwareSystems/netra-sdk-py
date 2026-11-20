"""
Unit tests for NetraTypeSafeInstrumentor.

The real `typesafe-sdk` package is not a test dependency (it needs a newer
pydantic than the dev environment pins): a minimal stand-in `typesafe_sdk`
module is registered in sys.modules before instrumenting, mirroring the
`system_one` signatures and response shape of typesafe-sdk 0.7.5.
"""

import asyncio
import json
import sys
import types
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterator, List, Mapping, Optional

import pytest
import wrapt
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import BaseModel

from netra.instrumentation.instruments import DEFAULT_INSTRUMENTS, DEFAULT_INSTRUMENTS_FOR_ROOT, InstrumentSet
from netra.instrumentation.libraries.typesafe import NetraTypeSafeInstrumentor
from netra.instrumentation.libraries.typesafe import utils as typesafe_utils
from netra.instrumentation.wiring.registry import CUSTOM_INSTRUMENTORS
from netra.instrumentation.wiring.triggers import INSTRUMENT_TRIGGERS

DEFAULT_MODEL = "jev-default"

# --------------------------------------------------------------------------------------
# Stand-ins for the SDK
# --------------------------------------------------------------------------------------


class Noul(BaseModel):
    type: str = "noul"
    instructions: Optional[str] = None


class Image:
    """Like the SDK's Image: stringifies to an opaque reference, never to its bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __str__(self) -> str:
        return "<image:abc123>"


class Usage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class SystemOneResponse(BaseModel):
    model: str
    usage: Usage
    answers: Dict[str, Any] = {}


class TypeSafeAPIError(Exception):
    pass


ANSWERS = {"billing": {"type": "noul", "noul": 0.93}}


def _respond(model: Optional[str], response_model: Any) -> Any:
    body = {"model": model or DEFAULT_MODEL, "usage": {"input_tokens": 40, "output_tokens": 3}, "answers": ANSWERS}
    return (response_model or SystemOneResponse).model_validate(body)


class TypeSafeClient:
    def __init__(self, *, model: Optional[str] = None, fail: bool = False) -> None:
        self._config = NS(default_model=model or DEFAULT_MODEL)
        self._fail = fail
        self.calls = 0

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: Optional[str] = None,
        response_model: Any = None,
        **_: Any,
    ) -> Any:
        self.calls += 1
        if self._fail:
            raise TypeSafeAPIError("400 bad request")
        return _respond(model or self._config.default_model, response_model)


class AsyncTypeSafeClient(TypeSafeClient):
    async def system_one(  # type: ignore[override]
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: Optional[str] = None,
        response_model: Any = None,
        **_: Any,
    ) -> Any:
        self.calls += 1
        if self._fail:
            raise TypeSafeAPIError("400 bad request")
        return _respond(model or self._config.default_model, response_model)


def _install_fake_module() -> Optional[types.ModuleType]:
    module = types.ModuleType("typesafe_sdk")
    module.TypeSafeClient = TypeSafeClient  # type: ignore[attr-defined]
    module.AsyncTypeSafeClient = AsyncTypeSafeClient  # type: ignore[attr-defined]
    previous = sys.modules.get("typesafe_sdk")
    sys.modules["typesafe_sdk"] = module
    return previous


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    previous = _install_fake_module()
    span_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    instrumentor = NetraTypeSafeInstrumentor()
    instrumentor._instrument(tracer_provider=provider)
    try:
        yield span_exporter
    finally:
        instrumentor._uninstrument()
        if previous is None:
            sys.modules.pop("typesafe_sdk", None)
        else:
            sys.modules["typesafe_sdk"] = previous


QUESTIONS = {"billing": Noul(instructions="Is this about billing?")}


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def _input(attrs: Any) -> List[Dict[str, Any]]:
    """The ``input`` attribute, with the questions entry's (the last one's) nested JSON decoded."""
    entries = json.loads(attrs["input"])
    entries[-1]["content"] = json.loads(entries[-1]["content"])
    return entries


def _output(attrs: Any) -> Dict[str, Any]:
    (entry,) = json.loads(attrs["output"])
    assert entry["role"] == "assistant"
    return json.loads(entry["content"])


def test_sync_call_records_generation_span(exporter: InMemorySpanExporter) -> None:
    result = TypeSafeClient().system_one(state="I was charged twice.", questions=QUESTIONS)

    assert result.answers == ANSWERS
    span = _only_span(exporter)
    attrs = span.attributes
    assert span.name == "typesafe.system_one"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    assert span.instrumentation_scope.name == "netra.instrumentation.typesafe"
    assert attrs["gen_ai.system"] == "TypeSafe"
    assert attrs["llm.request.type"] == "decision"
    assert attrs["netra.span.type"] == "GENERATION"
    assert attrs["gen_ai.request.model"] == DEFAULT_MODEL
    assert attrs["gen_ai.response.model"] == DEFAULT_MODEL
    assert attrs["gen_ai.usage.prompt_tokens"] == 40
    assert attrs["gen_ai.usage.completion_tokens"] == 3
    assert attrs["llm.usage.total_tokens"] == 43
    assert "llm.response.duration" in attrs

    assert _input(attrs) == [
        {"role": "user", "content": "I was charged twice."},
        {"role": "user", "content": {"billing": {"type": "noul", "instructions": "Is this about billing?"}}},
    ]
    assert _output(attrs) == ANSWERS


def test_content_recorded_only_as_input_and_output(exporter: InMemorySpanExporter) -> None:
    TypeSafeClient().system_one(state="x", questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert not [key for key in attrs if key.startswith(("gen_ai.prompt", "gen_ai.completion", "typesafe."))]


def test_positional_arguments_and_per_call_model(exporter: InMemorySpanExporter) -> None:
    TypeSafeClient().system_one({"message": "hi"}, QUESTIONS, model="jev-x")

    attrs = _only_span(exporter).attributes
    assert attrs["gen_ai.request.model"] == "jev-x"
    assert json.loads(_input(attrs)[0]["content"]) == {"message": "hi"}


def test_image_asset_recorded_by_reference_only(exporter: InMemorySpanExporter) -> None:
    image = Image(b"\x89PNG-raw-image-bytes")
    TypeSafeClient().system_one(state={"photo": image}, questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert json.loads(_input(attrs)[0]["content"]) == {"photo": "<image:abc123>"}
    assert "PNG" not in attrs["input"]


def test_dict_questions_are_recorded(exporter: InMemorySpanExporter) -> None:
    questions = {"tone": {"type": "choice", "criteria": {"calm": None, "angry": None}}}
    TypeSafeClient().system_one(state="x", questions=questions)

    assert _input(_only_span(exporter).attributes)[1]["content"] == questions


def test_custom_response_model(exporter: InMemorySpanExporter) -> None:
    class Custom(BaseModel):
        model: str
        billing_flag: Optional[bool] = None

    TypeSafeClient().system_one(state="x", questions=QUESTIONS, response_model=Custom)

    attrs = _only_span(exporter).attributes
    assert attrs["gen_ai.response.model"] == DEFAULT_MODEL
    assert "gen_ai.usage.prompt_tokens" not in attrs
    # No ``answers`` field: the rest of the body, minus model/usage, is the output.
    assert _output(attrs) == {"billing_flag": None}


def test_no_content_when_content_tracing_is_off(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    TypeSafeClient().system_one(state="secret", questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert "input" not in attrs
    assert "output" not in attrs
    assert attrs["gen_ai.usage.prompt_tokens"] == 40
    assert attrs["gen_ai.request.model"] == DEFAULT_MODEL


def test_long_state_is_cut_but_questions_and_json_survive(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(typesafe_utils, "get_attribute_max_len", lambda: 300)
    TypeSafeClient().system_one(state='"quoted"\n' * 1000, questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert len(attrs["input"]) <= 300
    state_entry, questions_entry = _input(attrs)
    assert state_entry["content"].startswith('"quoted"')
    assert questions_entry["content"] == {"billing": {"type": "noul", "instructions": "Is this about billing?"}}


def test_answers_that_do_not_fit_are_dropped_whole(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(typesafe_utils, "get_attribute_max_len", lambda: 120)
    many = {f"q{i}": {"type": "noul", "noul": 0.5} for i in range(20)}
    monkeypatch.setattr(sys.modules[__name__], "ANSWERS", many)
    TypeSafeClient().system_one(state="x", questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert len(attrs["output"]) <= 120
    kept = _output(attrs)
    assert 0 < len(kept) < 20
    assert list(kept) == [f"q{i}" for i in range(len(kept))]
    assert kept["q0"] == {"type": "noul", "noul": 0.5}


def test_non_ascii_content_is_recorded_unescaped(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = {"facturación": {"type": "noul", "noul": 0.9, "note": "請求 😀"}}
    monkeypatch.setattr(sys.modules[__name__], "ANSWERS", answers)
    questions = {"facturación": Noul(instructions="¿Es sobre facturación? 請求")}
    TypeSafeClient().system_one(state="Me cobraron dos veces 😀 二重請求", questions=questions)

    attrs = _only_span(exporter).attributes
    # Kept as the characters themselves, not as six-character \uXXXX escapes.
    assert "\\u" not in attrs["input"]
    assert "\\u" not in attrs["output"]
    assert _input(attrs) == [
        {"role": "user", "content": "Me cobraron dos veces 😀 二重請求"},
        {"role": "user", "content": {"facturación": {"type": "noul", "instructions": "¿Es sobre facturación? 請求"}}},
    ]
    assert _output(attrs) == answers


def test_long_non_ascii_state_is_cut_within_limit(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(typesafe_utils, "get_attribute_max_len", lambda: 300)
    TypeSafeClient().system_one(state="二重請求" * 1000, questions=QUESTIONS)

    attrs = _only_span(exporter).attributes
    assert len(attrs["input"]) <= 300
    state_entry, _ = _input(attrs)
    # Unescaped, one character costs one: far more of the state fits than the
    # ~20 characters six-character escapes would have left room for.
    assert len(state_entry["content"]) > 100
    assert set(state_entry["content"]) <= set("二重請求")


def test_lone_surrogates_are_replaced_so_export_can_encode(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ANSWERS", {"q\ud800": "a\udfff"})
    TypeSafeClient().system_one(state="bad\ud800", questions={"k\udc80": {"type": "noul"}})

    attrs = _only_span(exporter).attributes
    # OTLP encodes attributes as UTF-8, which lone surrogates would break.
    attrs["input"].encode("utf-8")
    attrs["output"].encode("utf-8")
    assert _input(attrs) == [
        {"role": "user", "content": "bad�"},
        {"role": "user", "content": {"k�": {"type": "noul"}}},
    ]
    # pydantic's JSON-mode dump already rewrites a surrogate in a dict key its
    # own way; the value reaches us intact and is replaced here.
    assert list(_output(attrs).values()) == ["a�"]


def test_error_is_recorded_and_reraised(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(TypeSafeAPIError):
        TypeSafeClient(fail=True).system_one(state="x", questions=QUESTIONS)

    span = _only_span(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "TypeSafeAPIError"
    assert "output" not in span.attributes
    assert "input" in span.attributes
    assert [event.name for event in span.events] == ["exception"]


def test_async_call(exporter: InMemorySpanExporter) -> None:
    result = asyncio.run(AsyncTypeSafeClient().system_one(state="x", questions=QUESTIONS, model="jev-async"))

    assert result.model == "jev-async"
    span = _only_span(exporter)
    assert span.status.status_code == StatusCode.OK
    assert span.attributes["gen_ai.request.model"] == "jev-async"
    assert _output(span.attributes) == ANSWERS


def test_async_error(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(TypeSafeAPIError):
        asyncio.run(AsyncTypeSafeClient(fail=True).system_one(state="x", questions=QUESTIONS))

    span = _only_span(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "TypeSafeAPIError"


def test_suppressed_instrumentation_creates_no_span(exporter: InMemorySpanExporter) -> None:
    from opentelemetry import context as context_api
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

    token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        client = TypeSafeClient()
        client.system_one(state="x", questions=QUESTIONS)
    finally:
        context_api.detach(token)

    assert client.calls == 1
    assert exporter.get_finished_spans() == ()


def test_uninstrument_removes_wrappers(exporter: InMemorySpanExporter) -> None:
    module = sys.modules["typesafe_sdk"]
    assert isinstance(vars(module.TypeSafeClient)["system_one"], wrapt.FunctionWrapper)
    assert isinstance(vars(module.AsyncTypeSafeClient)["system_one"], wrapt.FunctionWrapper)

    NetraTypeSafeInstrumentor()._uninstrument()

    assert not isinstance(vars(module.TypeSafeClient)["system_one"], wrapt.FunctionWrapper)
    assert not isinstance(vars(module.AsyncTypeSafeClient)["system_one"], wrapt.FunctionWrapper)


def test_registered_and_enabled_by_default() -> None:
    assert InstrumentSet.TYPESAFE in DEFAULT_INSTRUMENTS
    assert InstrumentSet.TYPESAFE in DEFAULT_INSTRUMENTS_FOR_ROOT
    assert INSTRUMENT_TRIGGERS[InstrumentSet.TYPESAFE] == ("typesafe_sdk",)
    (spec,) = CUSTOM_INSTRUMENTORS[InstrumentSet.TYPESAFE]
    assert spec.required_distributions == ("typesafe-sdk",)
    assert spec.module == "netra.instrumentation.libraries.typesafe"
    assert spec.class_name == "NetraTypeSafeInstrumentor"
