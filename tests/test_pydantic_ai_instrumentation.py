"""PydanticAI instrumentation must wrap ``run_sync`` rather than replace it.

The instrumented ``run_sync`` used to build and close its own event loop, which
left the caller's thread with a closed default loop, bypassed pydantic-ai's
nested-run guard and skipped agent-name inference. It also called
``result.usage()``, which raises on pydantic-ai 2.x where ``usage`` is a
property, so every instrumented run failed.
"""

from typing import Any

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
