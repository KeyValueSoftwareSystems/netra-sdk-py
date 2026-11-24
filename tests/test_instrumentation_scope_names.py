"""Tests for resolving a span's instrumentation scope to its ``InstrumentSet`` value.

The resolution feeds the ``netra.instrumentation.name`` attribute, ``root_instruments``
filtering and ``cost_excluded_instruments``, so a scope that resolves to the wrong
name silently breaks all three.
"""

import re
from pathlib import Path
from typing import Dict, Optional

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import netra
from netra.instrumentation.instruments import InstrumentSet, instrumentation_name_for_scope
from netra.instrumentation.wiring.registry import CUSTOM_INSTRUMENTORS
from netra.processors import InstrumentationSpanProcessor

pytestmark = pytest.mark.unit

INSTRUMENTATION_NAME = "netra.instrumentation.name"
NETRA_LIBRARIES_MODULE = "netra.instrumentation.libraries."

# Scopes whose package is not the instrumentation's value, or whose tracer is
# created in a submodule. Taken from the tracer names the instrumentations
# actually use (Netra's ``_TRACER_NAME`` constants and ``get_tracer(__name__)``
# in the installed Traceloop / OTel contrib packages).
SCOPES_NOT_NAMED_AFTER_THEIR_INSTRUMENT = [
    ("netra.instrumentation.cohere", InstrumentSet.COHEREAI),
    ("netra.instrumentation.mistralai", InstrumentSet.MISTRALAI),
    ("netra.instrumentation.pydantic_ai_slim", InstrumentSet.PYDANTIC_AI),
    ("netra.instrumentation.weaviate", InstrumentSet.WEAVIATEDB),
    ("opentelemetry.instrumentation.aiohttp_client", InstrumentSet.AIOHTTP),
    ("opentelemetry.instrumentation.chromadb", InstrumentSet.CHROMA),
    ("opentelemetry.instrumentation.crewai.instrumentation", InstrumentSet.CREW),
    ("opentelemetry.instrumentation.kafka", InstrumentSet.KAFKA_PYTHON),
    ("opentelemetry.instrumentation.llamaindex", InstrumentSet.LLAMA_INDEX),
    ("opentelemetry.instrumentation.mcp.instrumentation", InstrumentSet.MCP),
    ("opentelemetry.instrumentation.openai.v1", InstrumentSet.OPENAI),
    ("opentelemetry.instrumentation.pika.pika_instrumentor", InstrumentSet.PIKA),
    ("opentelemetry.instrumentation.qdrant", InstrumentSet.QDRANTDB),
]


@pytest.mark.parametrize(("scope_name", "instrument"), SCOPES_NOT_NAMED_AFTER_THEIR_INSTRUMENT)
def test_scope_not_named_after_its_instrument_resolves_to_the_instrument(
    scope_name: str, instrument: InstrumentSet
) -> None:
    assert instrumentation_name_for_scope(scope_name) == instrument.value


@pytest.mark.parametrize(
    ("scope_name", "expected"),
    [
        ("netra.instrumentation.openai", "openai"),
        ("opentelemetry.instrumentation.anthropic", "anthropic"),
        ("livekit-agents", "livekit"),
        # An instrumentation no member claims still reads as an instrumentation,
        # so root filtering keeps treating it as one it doesn't allow.
        ("opentelemetry.instrumentation.somethingnew", "somethingnew"),
        ("opentelemetry.instrumentation.somethingnew.sub", "somethingnew"),
        # Decorators, Netra.start_span and user tracers are not instrumentations.
        ("combat_sdk", None),
        ("my.app.tracer", None),
        ("netra.session_manager", None),
        ("openai", None),
    ],
)
def test_instrumentation_name_for_scope(scope_name: str, expected: Optional[str]) -> None:
    assert instrumentation_name_for_scope(scope_name) == expected


def test_no_scope_package_names_a_different_instrument() -> None:
    claimed: Dict[str, InstrumentSet] = {member.value: member for member in InstrumentSet}
    for member in InstrumentSet:
        for package in member.scope_packages:
            assert claimed.setdefault(package, member) is member, (package, member, claimed[package])


def _netra_tracer_names(library_module: str) -> set[str]:
    """Tracer names a Netra-authored instrumentation package creates, read from source.

    Read rather than imported: importing an instrumentation imports the library it patches.
    """
    package_dir = Path(netra.__file__).parent.joinpath(*library_module.split(".")[1:])
    names = set()
    for path in package_dir.rglob("*.py"):
        source = path.read_text()
        names.update(re.findall(r'^_TRACER_NAME\s*=\s*"([^"]+)"', source, re.M))
        names.update(re.findall(r'get_tracer\(\s*"([^"]+)"', source))
    return names


NETRA_AUTHORED_INSTRUMENTATIONS = sorted(
    {
        (member, spec.module)
        for member, specs in CUSTOM_INSTRUMENTORS.items()
        for spec in specs
        if spec.module.startswith(NETRA_LIBRARIES_MODULE)
    },
    key=lambda pair: pair[1],
)


@pytest.mark.parametrize(
    ("instrument", "library_module"),
    NETRA_AUTHORED_INSTRUMENTATIONS,
    ids=[module for _, module in NETRA_AUTHORED_INSTRUMENTATIONS],
)
def test_every_netra_tracer_resolves_to_its_registered_instrument(
    instrument: InstrumentSet, library_module: str
) -> None:
    # Guards new instrumentations: a tracer named after the library rather than
    # the InstrumentSet value needs ``scope_packages`` on the member.
    tracer_names = _netra_tracer_names(library_module)

    assert tracer_names, f"no tracer name found under {library_module}"
    assert {name: instrumentation_name_for_scope(name) for name in tracer_names} == {
        name: instrument.value for name in tracer_names
    }


def _stamped_name(scope_name: str) -> Optional[str]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.get_tracer(scope_name).start_span("chat").end()
    (span,) = exporter.get_finished_spans()
    return _attributes(span).get(INSTRUMENTATION_NAME)


def _attributes(span: ReadableSpan) -> Dict[str, object]:
    return dict(span.attributes or {})


@pytest.mark.parametrize(("scope_name", "instrument"), SCOPES_NOT_NAMED_AFTER_THEIR_INSTRUMENT)
def test_span_from_scope_not_named_after_its_instrument_is_stamped_with_the_instrument(
    scope_name: str, instrument: InstrumentSet
) -> None:
    assert _stamped_name(scope_name) == instrument.value


@pytest.mark.parametrize("scope_name", ["combat_sdk", "my.app.tracer", "opentelemetry.instrumentation.somethingnew"])
def test_span_from_unrecognised_scope_is_not_stamped(scope_name: str) -> None:
    assert _stamped_name(scope_name) is None
