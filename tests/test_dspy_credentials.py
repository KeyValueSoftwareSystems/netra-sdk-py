"""DSPy spans must not export provider credentials.

``dspy.LM(..., api_key=..., api_base=...)`` keeps both in ``lm.kwargs``, and the
LM wrappers used to dump the merged ``lm.kwargs`` + call kwargs into
``llm.headers`` and every bound argument into ``gen_ai.request.input``.
"""

import asyncio
import json
from typing import Any, Dict, Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.libraries.dspy.utils import get_llm_invocation_parameters, redact_sensitive_arguments
from netra.instrumentation.libraries.dspy.wrappers import LMAsyncCallWrapper, LMCallWrapper

SECRET = "sk-FAKE-SECRET-123"
INTERNAL_HOST = "https://internal-proxy.corp/v1"


class FakeLM:
    model = "openai/gpt-4o-mini"

    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": 1000,
            "api_key": SECRET,
            "api_base": INTERNAL_HOST,
        }

    def __call__(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        return ["hello"]

    async def acall(self, prompt: Optional[str] = None, messages: Any = None, **kwargs: Any) -> Any:
        return ["hello"]


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter: InMemorySpanExporter) -> Any:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


def _exported_text(exporter: InMemorySpanExporter) -> str:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return json.dumps(dict(spans[0].attributes or {}))


@pytest.mark.unit
def test_invocation_parameters_are_allowlisted() -> None:
    params = get_llm_invocation_parameters(
        FakeLM(), {"prompt": "hi", "api_key": "sk-CALLTIME", "temperature": 0.7, "self": object()}
    )
    assert params == {"temperature": 0.7, "max_tokens": 1000}


@pytest.mark.unit
def test_redact_sensitive_arguments_drops_credentials_recursively() -> None:
    redacted = redact_sensitive_arguments(
        {
            "prompt": "hi",
            "kwargs": {"api_key": "x", "openai_api_key": "y", "client_secret": "z", "max_tokens": 5},
            "extra_headers": {"Authorization": "Bearer t"},
        }
    )
    assert redacted == {"prompt": "hi", "kwargs": {"max_tokens": 5}}


@pytest.mark.unit
def test_sync_lm_call_does_not_export_credentials(tracer: Any, exporter: InMemorySpanExporter) -> None:
    lm = FakeLM()
    LMCallWrapper(tracer)(lm.__call__, lm, (), {"prompt": "hi", "api_key": "sk-CALLTIME", "temperature": 0.5})

    exported = _exported_text(exporter)
    for secret in (SECRET, "sk-CALLTIME", INTERNAL_HOST):
        assert secret not in exported
    attributes = dict(exporter.get_finished_spans()[0].attributes or {})
    assert json.loads(attributes[SpanAttributes.LLM_HEADERS]) == {"temperature": 0.5, "max_tokens": 1000}


@pytest.mark.unit
def test_async_lm_call_does_not_export_credentials(tracer: Any, exporter: InMemorySpanExporter) -> None:
    lm = FakeLM()
    asyncio.run(LMAsyncCallWrapper(tracer)(lm.acall, lm, (), {"prompt": "hi", "api_key": "sk-CALLTIME"}))

    exported = _exported_text(exporter)
    for secret in (SECRET, "sk-CALLTIME", INTERNAL_HOST):
        assert secret not in exported
