"""PydanticAI instrumentation must wrap ``run_sync`` rather than replace it.

The instrumented ``run_sync`` used to build and close its own event loop, which
left the caller's thread with a closed default loop, bypassed pydantic-ai's
nested-run guard and skipped agent-name inference. It also called
``result.usage()``, which raises on pydantic-ai 2.x where ``usage`` is a
property, so every instrumented run failed.
"""

import asyncio
from typing import Any, Iterator, List

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.libraries.pydantic_ai.utils import set_pydantic_response_attributes


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


class _Usage:
    input_tokens = 7
    output_tokens = 3
    total_tokens = 10
    details: dict = {}


class _ResultWithUsageProperty:
    """pydantic-ai >= 2: ``usage`` is a property."""

    model_name = "test-model"
    output = "hello"

    @property
    def usage(self) -> _Usage:
        return _Usage()


class _ResultWithUsageMethod:
    """pydantic-ai < 2: ``usage`` is a method."""

    model_name = "test-model"
    output = "hello"

    def usage(self) -> _Usage:
        return _Usage()


class _ResultWithBrokenUsage:
    output = "hello"

    @property
    def usage(self) -> Any:
        raise RuntimeError("shape changed")


@pytest.mark.unit
@pytest.mark.parametrize("result_cls", [_ResultWithUsageProperty, _ResultWithUsageMethod])
def test_response_attributes_read_usage_as_property_or_method(
    provider: TracerProvider, exporter: InMemorySpanExporter, result_cls: type
) -> None:
    with provider.get_tracer(__name__).start_as_current_span("run") as span:
        set_pydantic_response_attributes(span, result_cls())

    attributes = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == "7"
    assert attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == "3"
    assert attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == "10"


@pytest.mark.unit
def test_response_attributes_never_raise(provider: TracerProvider) -> None:
    with provider.get_tracer(__name__).start_as_current_span("run") as span:
        set_pydantic_response_attributes(span, _ResultWithBrokenUsage())


# --- end-to-end against the real library (skipped when pydantic-ai is absent) ---


@pytest.fixture
def instrumented(provider: TracerProvider) -> Iterator[None]:
    pytest.importorskip("pydantic_ai")
    from netra.instrumentation.libraries.pydantic_ai import NetraPydanticAIInstrumentor

    instrumentor = NetraPydanticAIInstrumentor()
    # skip_dep_check: the dependency may be installed as pydantic-ai-slim.
    instrumentor.instrument(tracer_provider=provider, skip_dep_check=True)
    try:
        yield
    finally:
        instrumentor.uninstrument()


def _test_agent() -> Any:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    return Agent(TestModel())


def _names(exporter: InMemorySpanExporter) -> List[str]:
    return [span.name for span in exporter.get_finished_spans()]


@pytest.mark.unit
def test_run_sync_leaves_the_threads_event_loop_usable(instrumented: None) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = _test_agent().run_sync("hi")

        assert result.output
        assert asyncio.get_event_loop() is loop
        assert not loop.is_closed()
        assert loop.run_until_complete(asyncio.sleep(0, result="ok")) == "ok"
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.mark.unit
def test_run_sync_returns_the_upstream_result_type(instrumented: None) -> None:
    from pydantic_ai.run import AgentRunResult

    assert isinstance(_test_agent().run_sync("hi"), AgentRunResult)


@pytest.mark.unit
def test_run_sync_infers_the_agent_name_from_the_caller(instrumented: None) -> None:
    support_agent = _test_agent()
    support_agent.run_sync("hi")
    assert support_agent.name == "support_agent"


@pytest.mark.unit
def test_nested_run_sync_still_raises_upstream_user_error(instrumented: None) -> None:
    from pydantic_ai.exceptions import UserError

    outer = _test_agent()
    inner = _test_agent()

    @outer.tool_plain
    def call_inner() -> str:
        return str(inner.run_sync("nested").output)

    with pytest.raises(UserError):
        outer.run_sync("hi")


@pytest.mark.unit
def test_run_sync_span_is_the_parent_of_the_run_span(instrumented: None, exporter: InMemorySpanExporter) -> None:
    _test_agent().run_sync("hi")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert "pydantic_ai.agent.run_sync" in spans, _names(exporter)
    assert "pydantic_ai.agent.run" in spans, _names(exporter)
    run_sync, run = spans["pydantic_ai.agent.run_sync"], spans["pydantic_ai.agent.run"]
    assert run.parent is not None and run.parent.span_id == run_sync.context.span_id
    assert run_sync.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
