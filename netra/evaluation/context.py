import logging
from types import TracebackType
from typing import Dict, Optional, Type, cast

from opentelemetry import trace

from netra.config import Config
from netra.span_wrapper import SpanType, SpanWrapper

from .client import _EvaluationHttpClient
from .models import DatasetItem, EntryStatus, Run
from .utils import get_session_id_from_baggage, get_trace_id_from_span

logger = logging.getLogger(__name__)


class RunEntryContext:
    """Context around a single test entry that starts a parent evaluation span and posts agent_triggered."""

    def __init__(self, client: _EvaluationHttpClient, cfg: Config, run: Run, entry: DatasetItem) -> None:
        self._client = client
        self._config = cfg
        self.run = run
        self.entry = entry
        self._span_wrapper: Optional[SpanWrapper] = None
        self._trace_id: Optional[str] = None

    def __enter__(self) -> "RunEntryContext":
        prefix = f"{Config.LIBRARY_NAME}.eval"
        attributes: Dict[str, str] = {
            f"{prefix}.dataset_id": self.run.dataset_id,
            f"{prefix}.run_id": self.run.id,
            f"{prefix}.test_id": self.entry.id,
        }
        self._span_wrapper = SpanWrapper(f"{self.run.name}", attributes=attributes, as_type=SpanType.SPAN)
        self._span_wrapper.__enter__()

        if self._span_wrapper.span is not None:
            self._trace_id = get_trace_id_from_span(self._span_wrapper.span)
        session_id = get_session_id_from_baggage()

        try:
            self._client.post_entry_status(
                self.run.id,
                self.entry.id,
                status=EntryStatus.AGENT_TRIGGERED,
                trace_id=self._trace_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.debug("netra.evaluation: Failed to POST agent_triggered: %s", exc, exc_info=True)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[Exception],
        tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is not None:
            try:
                session_id = get_session_id_from_baggage()
                self._client.post_entry_status(
                    self.run.id,
                    self.entry.id,
                    status=EntryStatus.FAILED,
                    trace_id=self._trace_id,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.debug("netra.evaluation: Failed to POST agent failure status: %s", exc, exc_info=True)
        try:
            if self._span_wrapper is not None:
                self._span_wrapper.__exit__(exc_type, cast(Optional[Exception], exc), tb)
        finally:
            self._span_wrapper = None
        return True if exc_type is not None else False

    @property
    def trace_id(self) -> Optional[str]:
        return self._trace_id

    @property
    def span(self) -> Optional[trace.Span]:
        return self._span_wrapper.span if self._span_wrapper else None
