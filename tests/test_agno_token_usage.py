"""Agno spans must not carry token usage.

The underlying provider instrumentation (google_genai, openai, anthropic, ...)
emits its own child span with the same usage numbers, so stamping usage on the
Agno agent/team/LLM spans double-counts every call in the trace.
"""

from typing import Any, Dict, List

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.agno.utils import set_response_attributes
from netra.instrumentation.agno.wrappers import (
    LlmSpanStreamingWrapper,
    model_response_capture_wrapper,
)

USAGE_ATTRIBUTES = (
    SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
)

METRICS = {"input_tokens": 4744, "output_tokens": 278, "reasoning_tokens": 0, "total_tokens": 5022}


class FakeAgnoResponse:
    """Stand-in for an Agno RunOutput / ModelResponse carrying usage metrics."""

    def __init__(self, metrics: Dict[str, int]) -> None:
        self.metrics = metrics
        self.content = "hello"


class FakeModel:
    """Stand-in for an Agno Model instance."""

    id = "gemini-3.1-pro-preview"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Any:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


def _finished_attributes(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"expected exactly one span, got {[s.name for s in spans]}"
    return dict(spans[0].attributes or {})


@pytest.mark.unit
def test_agent_run_span_omits_token_usage(tracer: Any, exporter: InMemorySpanExporter) -> None:
    with tracer.start_as_current_span("agno.agent.run.Single-Skill Host") as span:
        set_response_attributes(span, FakeAgnoResponse(METRICS))

    attributes = _finished_attributes(exporter)
    assert attributes.get("output")
    for usage_attribute in USAGE_ATTRIBUTES:
        assert usage_attribute not in attributes


@pytest.mark.unit
def test_non_streaming_llm_span_omits_token_usage(tracer: Any, exporter: InMemorySpanExporter) -> None:
    assistant_message = FakeAgnoResponse(METRICS)
    wrapper = model_response_capture_wrapper(tracer)

    wrapper(lambda *a, **kw: FakeAgnoResponse(METRICS), FakeModel(), ([], assistant_message), {})

    attributes = _finished_attributes(exporter)
    assert attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gemini-3.1-pro-preview"
    for usage_attribute in USAGE_ATTRIBUTES:
        assert usage_attribute not in attributes


@pytest.mark.unit
def test_streaming_llm_span_omits_token_usage(tracer: Any, exporter: InMemorySpanExporter) -> None:
    span = tracer.start_span("gemini-3.1-pro-preview")
    chunks: List[FakeAgnoResponse] = [FakeAgnoResponse(METRICS)]

    stream = LlmSpanStreamingWrapper(span=span, response=iter(chunks), ctx_token=None)
    consumed: List[Any] = list(stream)

    assert consumed == chunks
    attributes = _finished_attributes(exporter)
    for usage_attribute in USAGE_ATTRIBUTES:
        assert usage_attribute not in attributes
