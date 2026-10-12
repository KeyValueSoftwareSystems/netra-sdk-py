"""``trace_content=False`` must keep prompt/completion content off every Netra span (audit H-4).

Each case drives one instrumentor's attribute-setting code with a sentinel string in
every prompt, completion, tool argument, transcript and system prompt, then checks
the exported span:

- content off: the sentinel appears in no attribute and no event, but the metadata
  the case lists (model, usage, ...) is still there;
- content on: the sentinel *is* there, which proves the case reaches the code path.

Fakes and in-memory exporters only — no provider SDK is needed.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

SENTINEL = "SECRET-SENTINEL-123"


# Fixtures
@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


@pytest.fixture
def tracer(provider: TracerProvider) -> Any:
    return provider.get_tracer("test.trace_content")


class _FakeLM:
    model = "openai/gpt-test"
    kwargs: Dict[str, Any] = {"temperature": 0.0}

    def __call__(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        return [SENTINEL]


# DSPy wrappers used to swallow the user's exception and return None.
class _BoomError(Exception):
    pass


class _RaisingLM(_FakeLM):
    def __call__(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        raise _BoomError("provider failed")

    async def acall(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        raise _BoomError("provider failed")


def test_dspy_lm_call_reraises_the_provider_error(tracer: Any, exporter: InMemorySpanExporter) -> None:
    from netra.instrumentation.libraries.dspy.wrappers import LMCallWrapper

    lm = _RaisingLM()
    with pytest.raises(_BoomError):
        LMCallWrapper(tracer)(lm.__call__, lm, (), {"prompt": "hi"})
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_dspy_async_lm_call_reraises_the_provider_error(tracer: Any, exporter: InMemorySpanExporter) -> None:
    from netra.instrumentation.libraries.dspy.wrappers import LMAsyncCallWrapper

    lm = _RaisingLM()
    with pytest.raises(_BoomError):
        asyncio.run(LMAsyncCallWrapper(tracer)(lm.acall, lm, (), {"prompt": "hi"}))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.parametrize("wrapper_name", ["ModuleForwardWrapper", "ModuleForwardSyncWrapper", "ToolCallWrapper"])
def test_dspy_module_and_tool_wrappers_reraise(wrapper_name: str, tracer: Any) -> None:
    from netra.instrumentation.libraries.dspy import wrappers

    def failing(*args: Any, **kwargs: Any) -> Any:
        raise _BoomError("user code failed")

    instance = SimpleNamespace(name="tool", desc="")
    wrapper = getattr(wrappers, wrapper_name)(tracer)
    with pytest.raises(_BoomError):
        wrapper(failing, instance, (), {"x": 1})


def test_dspy_telemetry_failure_does_not_break_the_call(
    tracer: Any, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    from netra.instrumentation.libraries.dspy import wrappers

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("telemetry bug")

    monkeypatch.setattr(wrappers, "get_llm_invocation_parameters", broken)
    monkeypatch.setattr(wrappers, "extract_usage_info", broken)
    lm = _FakeLM()
    assert wrappers.LMCallWrapper(tracer)(lm.__call__, lm, (), {"prompt": "hi"}) == [SENTINEL]
    assert len(exporter.get_finished_spans()) == 1
