from typing import Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from netra.session import SessionSpanProcessor
from netra.span_aggregation import SpanAggregationProcessor


class CombinedSpanProcessor(SpanProcessor):  # type: ignore[misc]
    def __init__(self) -> None:
        self.session_processor = SessionSpanProcessor()
        self.aggregation_processor = SpanAggregationProcessor()

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        self.session_processor.on_start(span, parent_context)
        self.aggregation_processor.on_start(span, parent_context)

    def on_end(self, span: trace.Span) -> None:
        self.session_processor.on_end(span)
        self.aggregation_processor.on_end(span)

    def force_flush(self, timeout_millis: int = 30000) -> None:
        self.session_processor.force_flush(timeout_millis)
        self.aggregation_processor.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self.session_processor.shutdown()
        self.aggregation_processor.shutdown()
