"""Assert-based checks for SpanIOProcessor db.statement → input mapping."""

from opentelemetry.sdk.trace import TracerProvider

from netra.processors.instrumentation_span_processor import InstrumentationSpanProcessor
from netra.processors.span_io_processor import SpanIOProcessor


def _start_span_with_io_processor():
    # ORDER MATTERS: InstrumentationSpanProcessor must precede SpanIOProcessor
    # (same as Tracer setup) so terminal writes go through the class method.
    provider = TracerProvider()
    provider.add_span_processor(InstrumentationSpanProcessor())
    processor = SpanIOProcessor()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test")
    return tracer.start_span("db.query"), processor


def test_db_statement_maps_to_input_when_empty():
    span, _ = _start_span_with_io_processor()
    span.set_attribute("db.statement", "SELECT 1")

    attrs = dict(span.attributes or {})
    assert attrs.get("input") == "SELECT 1"
    assert attrs.get("db.statement") == "SELECT 1"
    assert attrs.get("output") == ""
    span.end()


def test_db_statement_does_not_overwrite_existing_input():
    span, _ = _start_span_with_io_processor()
    span.set_attribute("input", "already")
    span.set_attribute("db.statement", "SELECT 1")

    attrs = dict(span.attributes or {})
    assert attrs.get("input") == "already"
    assert attrs.get("db.statement") == "SELECT 1"
    span.end()


def test_db_statement_parameters_do_not_map_to_input():
    span, _ = _start_span_with_io_processor()
    span.set_attribute("db.statement.parameters", "('secret',)")

    attrs = dict(span.attributes or {})
    assert attrs.get("input") == ""
    assert attrs.get("output") == ""
    assert attrs.get("db.statement.parameters") == "('secret',)"
    span.end()


def test_db_system_and_name_alone_leave_input_empty():
    span, _ = _start_span_with_io_processor()
    span.set_attribute("db.system", "mysql")
    span.set_attribute("db.name", "app")

    attrs = dict(span.attributes or {})
    assert attrs.get("input") == ""
    assert attrs.get("db.system") == "mysql"
    assert attrs.get("db.name") == "app"
    span.end()


def test_empty_db_statement_does_not_replace_input():
    span, _ = _start_span_with_io_processor()
    span.set_attribute("db.statement", "")

    attrs = dict(span.attributes or {})
    assert attrs.get("input") == ""
    assert attrs.get("db.statement") == ""
    span.end()
