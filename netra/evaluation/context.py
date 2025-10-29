import logging
from typing import Any, Dict, Optional

from opentelemetry import baggage, trace

from netra.config import Config
from netra.span_wrapper import SpanType, SpanWrapper

from .client import _EvaluationHttpClient
from .models import DatasetItem, Run

logger = logging.getLogger(__name__)


class RunEntryContext:
    """Context around a single test entry that starts a parent evaluation span and posts agent_triggered."""

    def __init__(self, client: _EvaluationHttpClient, cfg: Config, run: Run, entry: DatasetItem) -> None:
        self._client = client
        self._cfg = cfg
        self.run = run
        self.entry = entry
        self._span_cm: Optional[SpanWrapper] = None
        self._trace_id_hex: Optional[str] = None

    @staticmethod
    def _trace_id_hex_from_span(span: trace.Span) -> str:
        ctx = span.get_span_context()
        return f"{ctx.trace_id:032x}"

    @staticmethod
    def _get_session_id_from_baggage() -> Any:
        try:
            return baggage.get_baggage("session_id")
        except Exception:
            return None

    def __enter__(self) -> "RunEntryContext":
        attrs: Dict[str, str] = {
            "netra.eval.dataset_id": self.run.dataset_id,
            "netra.eval.run_id": self.run.id,
            "netra.eval.test_id": self.entry.id,
        }
        self._span_cm = SpanWrapper("evaluation.entry", attributes=attrs, as_type=SpanType.SPAN)
        self._span_cm.__enter__()

        if self._span_cm.span is not None:
            self._trace_id_hex = self._trace_id_hex_from_span(self._span_cm.span)
        session_id = self._get_session_id_from_baggage()

        try:
            self._client.post_entry_status(
                self.run.id, self.entry.id, status="agent_triggered", trace_id=self._trace_id_hex, session_id=session_id
            )
        except Exception as e:
            logger.debug("Failed to POST agent_triggered: %s", e, exc_info=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> Any:  # type:ignore[no-untyped-def]
        try:
            if self._span_cm is not None:
                self._span_cm.__exit__(exc_type, exc, tb)
        finally:
            self._span_cm = None
        return False

    @property
    def trace_id(self) -> Optional[str]:
        return self._trace_id_hex

    @property
    def span(self) -> Optional[trace.Span]:
        return self._span_cm.span if self._span_cm else None
