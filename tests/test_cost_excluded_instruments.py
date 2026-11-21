"""Tests for ``cost_excluded_instruments``: usage attributes are stripped from spans of excluded instrumentations."""

import json
from typing import Any, Dict, FrozenSet, Iterator, Mapping, Optional
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra import Netra
from netra.instrumentation.instruments import ALL_INSTRUMENTS, NetraInstruments
from netra.processors import InstrumentationSpanProcessor
from netra.utils import resolve_cost_excluded_instruments

OPENAI_SCOPE = "opentelemetry.instrumentation.openai"
NETRA_OPENAI_SCOPE = "netra.instrumentation.openai"
ANTHROPIC_SCOPE = "opentelemetry.instrumentation.anthropic"
MANUAL_SCOPE = "combat_sdk"

CUSTOM_USAGE = json.dumps([{"model": "anthropic/claude-sonnet-4.6", "usage_type": "total", "cost_in_usd": 0.14}])

USAGE_ATTRIBUTES: Dict[str, Any] = {
    "netra.usage": CUSTOM_USAGE,
    "gen_ai.usage.prompt_tokens": 120,
    "gen_ai.usage.completion_tokens": 30,
    "gen_ai.usage.total_tokens": 150,
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 30,
    "gen_ai.usage.cache_read_input_tokens": 10,
    "gen_ai.usage.cache_read.input_tokens": 10,
    "gen_ai.usage.cache_creation_input_tokens_5m": 4,
    "gen_ai.usage.prompt.character_count": 512,
    "llm.usage.total_tokens": 150,
    "gen_ai.audio.duration": 1.5,
}

KEPT_ATTRIBUTES: Dict[str, Any] = {
    "gen_ai.request.model": "anthropic/claude-sonnet-4.6",
    "gen_ai.prompt.0.content": "hello",
    "gen_ai.performance.time_to_first_token": 0.2,
    "netra.usage.source": "framework",
}


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _provider(exporter: InMemorySpanExporter, excluded: FrozenSet[str]) -> TracerProvider:
    """Build a provider with the production order: Netra's processor before the exporting one."""
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor(cost_excluded_instrument_names=excluded))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _emit(
    provider: TracerProvider,
    scope: str,
    *,
    set_later: Optional[Mapping[str, Any]] = None,
    at_creation: Optional[Mapping[str, Any]] = None,
) -> None:
    tracer = provider.get_tracer(scope)
    span = tracer.start_span("openai.chat", attributes=dict(at_creation or {}))
    for key, value in (set_later or {}).items():
        span.set_attribute(key, value)
    span.end()


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _attributes(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


@pytest.mark.parametrize("scope", [OPENAI_SCOPE, NETRA_OPENAI_SCOPE])
def test_excluded_instrument_span_loses_usage_attributes(exporter: InMemorySpanExporter, scope: str) -> None:
    provider = _provider(exporter, frozenset({"openai"}))

    _emit(provider, scope, set_later={**USAGE_ATTRIBUTES, **KEPT_ATTRIBUTES})

    attributes = _attributes(_only_span(exporter))
    assert set(USAGE_ATTRIBUTES) & set(attributes) == set()


@pytest.mark.parametrize(
    ("scope", "instrument"),
    [
        ("netra.instrumentation.cohere", NetraInstruments.COHEREAI),
        ("netra.instrumentation.mistralai", NetraInstruments.MISTRALAI),
        ("netra.instrumentation.pydantic_ai_slim", NetraInstruments.PYDANTIC_AI),
        ("netra.instrumentation.weaviate", NetraInstruments.WEAVIATEDB),
        ("opentelemetry.instrumentation.chromadb", NetraInstruments.CHROMA),
        ("opentelemetry.instrumentation.crewai.instrumentation", NetraInstruments.CREW),
        ("opentelemetry.instrumentation.llamaindex", NetraInstruments.LLAMA_INDEX),
        ("opentelemetry.instrumentation.mcp.instrumentation", NetraInstruments.MCP),
        ("opentelemetry.instrumentation.qdrant", NetraInstruments.QDRANTDB),
    ],
)
@pytest.mark.parametrize("by_all", [False, True], ids=["named", "all"])
def test_span_whose_scope_tail_differs_from_instrument_value_loses_usage(
    exporter: InMemorySpanExporter, scope: str, instrument: NetraInstruments, by_all: bool
) -> None:
    excluded = {NetraInstruments.ALL} if by_all else {instrument}
    provider = _provider(exporter, resolve_cost_excluded_instruments(excluded))

    _emit(provider, scope, set_later=USAGE_ATTRIBUTES)

    assert set(USAGE_ATTRIBUTES) & set(_attributes(_only_span(exporter))) == set()


def test_excluded_instrument_span_keeps_model_and_content(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, frozenset({"openai"}))

    _emit(provider, OPENAI_SCOPE, set_later={**USAGE_ATTRIBUTES, **KEPT_ATTRIBUTES})

    attributes = _attributes(_only_span(exporter))
    assert {key: attributes.get(key) for key in KEPT_ATTRIBUTES} == KEPT_ATTRIBUTES
    assert attributes["netra.instrumentation.name"] == "openai"


def test_usage_passed_at_span_creation_is_stripped(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, frozenset({"openai"}))

    _emit(provider, OPENAI_SCOPE, at_creation={**USAGE_ATTRIBUTES, **KEPT_ATTRIBUTES})

    attributes = _attributes(_only_span(exporter))
    assert set(USAGE_ATTRIBUTES) & set(attributes) == set()
    assert attributes["gen_ai.request.model"] == KEPT_ATTRIBUTES["gen_ai.request.model"]


def test_non_excluded_instrument_span_keeps_usage(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, frozenset({"openai"}))

    _emit(provider, ANTHROPIC_SCOPE, set_later=USAGE_ATTRIBUTES)

    attributes = _attributes(_only_span(exporter))
    assert {key: attributes.get(key) for key in USAGE_ATTRIBUTES} == USAGE_ATTRIBUTES


def test_empty_exclusion_keeps_usage(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, frozenset())

    _emit(provider, OPENAI_SCOPE, set_later=USAGE_ATTRIBUTES)

    attributes = _attributes(_only_span(exporter))
    assert {key: attributes.get(key) for key in USAGE_ATTRIBUTES} == USAGE_ATTRIBUTES


def test_all_sentinel_excludes_every_instrumentation(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, resolve_cost_excluded_instruments({NetraInstruments.ALL}))

    _emit(provider, OPENAI_SCOPE, set_later=USAGE_ATTRIBUTES)
    _emit(provider, ANTHROPIC_SCOPE, set_later=USAGE_ATTRIBUTES)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert [set(USAGE_ATTRIBUTES) & set(_attributes(span)) for span in spans] == [set(), set()]


def test_manual_span_keeps_usage_when_all_excluded(exporter: InMemorySpanExporter) -> None:
    provider = _provider(exporter, resolve_cost_excluded_instruments({NetraInstruments.ALL}))

    _emit(provider, MANUAL_SCOPE, set_later={"netra.usage": CUSTOM_USAGE})

    assert _attributes(_only_span(exporter))["netra.usage"] == CUSTOM_USAGE


@pytest.mark.parametrize(
    ("instruments", "expected"),
    [
        (None, frozenset()),
        (set(), frozenset()),
        ({NetraInstruments.OPENAI}, frozenset({"openai"})),
        ({NetraInstruments.OPENAI, NetraInstruments.ANTHROPIC}, frozenset({"openai", "anthropic"})),
        ({NetraInstruments.ALL}, frozenset(member.value for member in ALL_INSTRUMENTS)),
        ({NetraInstruments.ALL, NetraInstruments.OPENAI}, frozenset(member.value for member in ALL_INSTRUMENTS)),
    ],
)
def test_resolve_cost_excluded_instruments(
    instruments: Optional[set[NetraInstruments]], expected: FrozenSet[str]
) -> None:
    assert resolve_cost_excluded_instruments(instruments) == expected


class TestNetraInitWiring:
    @pytest.fixture(autouse=True)
    def reset_netra(self) -> Iterator[None]:
        with Netra._init_lock:
            Netra._initialized = False
        yield
        with Netra._init_lock:
            Netra._initialized = False

    @pytest.mark.parametrize(
        ("cost_excluded_instruments", "expected_names"),
        [
            (None, frozenset()),
            ({NetraInstruments.OPENAI}, frozenset({"openai"})),
        ],
    )
    # configure_package_logging is patched because it sets the ``netra`` logger
    # level process-wide, which would hide warnings from later tests' caplog.
    @patch("netra.configure_package_logging")
    @patch("netra.init_instrumentations")
    @patch("netra.Tracer")
    def test_init_passes_cost_excluded_names_to_tracer(
        self,
        mock_tracer: MagicMock,
        _mock_init_instrumentations: MagicMock,
        _mock_configure_package_logging: MagicMock,
        cost_excluded_instruments: Optional[set[NetraInstruments]],
        expected_names: FrozenSet[str],
    ) -> None:
        Netra.init(cost_excluded_instruments=cost_excluded_instruments)

        mock_tracer.assert_called_once()
        assert mock_tracer.call_args.kwargs["cost_excluded_instrument_names"] == expected_names
