"""
Unit tests for SessionSpanProcessor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from contextlib import contextmanager
from typing import Iterator, Optional
from unittest.mock import Mock, patch

import pytest
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from netra.processors.session_span_processor import SessionSpanProcessor
from netra.session_manager import ATTR_SESSION_ID, ATTR_TENANT_ID, ATTR_USER_ID


class TestSessionSpanProcessor:
    """Test SessionSpanProcessor core functionality."""

    def test_initialization(self):
        """Test SessionSpanProcessor initialization."""
        # Act
        processor = SessionSpanProcessor()

        # Assert
        assert processor is not None
        assert hasattr(processor, "on_start")
        assert hasattr(processor, "on_end")
        assert hasattr(processor, "force_flush")
        assert hasattr(processor, "shutdown")

    @patch("netra.processors.session_span_processor.SessionManager")
    @patch("netra.processors.session_span_processor.otel_context")
    @patch("netra.processors.session_span_processor.baggage")
    @patch("netra.processors.session_span_processor.Config")
    def test_on_start_with_session_attributes(self, mock_config, mock_baggage, mock_context, mock_session_manager):
        """Test on_start method with session attributes present."""
        # Arrange
        processor = SessionSpanProcessor()
        mock_span = Mock()
        mock_parent_context = Mock()

        # Configure mocks
        mock_config.LIBRARY_NAME = "netra"
        mock_config.LIBRARY_VERSION = "1.0.0"
        mock_config.SDK_NAME = "netra-sdk"

        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx

        mock_baggage.get_baggage.side_effect = lambda key, ctx: {
            "session_id": "session123",
            "user_id": "user456",
            "tenant_id": "tenant789",
            "custom_keys": "key1,key2",
            "custom.key1": "value1",
            "custom.key2": "value2",
        }.get(key)

        # Act
        processor.on_start(mock_span, mock_parent_context)

        # Assert
        mock_session_manager.set_current_span.assert_called_once_with(mock_span)
        mock_context.get_current.assert_called_once()

        # Verify span attributes are set
        expected_calls = [
            (("library.name", "netra"),),
            (("library.version", "1.0.0"),),
            (("sdk.name", "netra-sdk"),),
            (("netra.session_id", "session123"),),
            (("netra.user_id", "user456"),),
            (("netra.tenant_id", "tenant789"),),
            (("netra.custom.key1", "value1"),),
            (("netra.custom.key2", "value2"),),
        ]

        assert mock_span.set_attribute.call_count == len(expected_calls)
        for call_args in expected_calls:
            mock_span.set_attribute.assert_any_call(*call_args[0])

    @patch("netra.processors.session_span_processor.SessionManager")
    @patch("netra.processors.session_span_processor.otel_context")
    @patch("netra.processors.session_span_processor.baggage")
    @patch("netra.processors.session_span_processor.Config")
    def test_on_start_with_minimal_attributes(self, mock_config, mock_baggage, mock_context, mock_session_manager):
        """Test on_start method with only basic library attributes."""
        # Arrange
        processor = SessionSpanProcessor()
        mock_span = Mock()

        # Configure mocks
        mock_config.LIBRARY_NAME = "netra"
        mock_config.LIBRARY_VERSION = "1.0.0"
        mock_config.SDK_NAME = "netra-sdk"

        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx

        # No session attributes available
        mock_baggage.get_baggage.return_value = None

        # Act
        processor.on_start(mock_span)

        # Assert
        mock_session_manager.set_current_span.assert_called_once_with(mock_span)

        # Verify only basic library attributes are set
        expected_calls = [(("library.name", "netra"),), (("library.version", "1.0.0"),), (("sdk.name", "netra-sdk"),)]

        assert mock_span.set_attribute.call_count == len(expected_calls)
        for call_args in expected_calls:
            mock_span.set_attribute.assert_any_call(*call_args[0])

    @patch("netra.processors.session_span_processor.logger")
    @patch("netra.processors.session_span_processor.SessionManager")
    @patch("netra.processors.session_span_processor.otel_context")
    def test_on_start_with_exception_handling(self, mock_context, mock_session_manager, mock_logger):
        """Test on_start method handles exceptions gracefully."""
        # Arrange
        processor = SessionSpanProcessor()
        mock_span = Mock()

        # Configure mock to raise exception
        mock_context.get_current.side_effect = Exception("Test exception")

        # Act
        processor.on_start(mock_span)

        # Assert
        mock_logger.exception.assert_called_once()
        assert "Error setting span attributes:" in str(mock_logger.exception.call_args)

    def test_on_end_method(self):
        """Test on_end method (no-op implementation)."""
        # Arrange
        processor = SessionSpanProcessor()
        mock_span = Mock()

        # Act & Assert (should not raise any exception)
        processor.on_end(mock_span)

    def test_force_flush_method(self):
        """Test force_flush method (no-op implementation)."""
        # Arrange
        processor = SessionSpanProcessor()

        # Act & Assert (should not raise any exception)
        processor.force_flush()
        processor.force_flush(timeout_millis=5000)

    def test_shutdown_method(self):
        """Test shutdown method (no-op implementation)."""
        # Arrange
        processor = SessionSpanProcessor()

        # Act & Assert (should not raise any exception)
        processor.shutdown()


@contextmanager
def _ambient_baggage(**items: str) -> Iterator[None]:
    """Attach baggage to the ambient OTel context for the duration of the block.

    Args:
        items: Baggage key/value pairs to attach.
    """
    ctx = otel_context.get_current()
    for key, value in items.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def _context_with_baggage(**items: str) -> Context:
    """Build a standalone context carrying baggage, never attached to any task.

    Stands in for the context a framework snapshots and later hands back as an
    explicit ``context=`` parent — LiveKit's ``AgentSession._root_span_context``.

    Args:
        items: Baggage key/value pairs to put in the context.

    Returns:
        A context carrying *items* and nothing else.
    """
    ctx = Context()
    for key, value in items.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    return ctx


class TestSessionSpanProcessorBaggageResolution:
    """Where ``on_start`` reads session baggage from, exercised through a real provider.

    A span started with an explicit ``context=`` is not created *inside* that
    context — the SDK fires ``on_start`` before making it current — so the parent
    context has to be consulted separately or those spans lose their session id.
    """

    @pytest.fixture  # type: ignore[misc]
    def span_exporter(self) -> InMemorySpanExporter:
        """An in-memory exporter collecting the spans a test produces."""
        return InMemorySpanExporter()

    @pytest.fixture  # type: ignore[misc]
    def tracer(self, span_exporter: InMemorySpanExporter) -> Iterator[trace.Tracer]:
        """A tracer whose provider runs ``SessionSpanProcessor`` then exports.

        Args:
            span_exporter: The exporter to collect finished spans into.
        """
        provider = TracerProvider()
        provider.add_span_processor(SessionSpanProcessor())
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        yield provider.get_tracer(__name__)
        provider.shutdown()

    @staticmethod
    def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
        """Return the single span the test produced.

        Args:
            span_exporter: The exporter the tracer fixture wrote to.
        """
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1, f"expected exactly one exported span, got {[span.name for span in spans]}"
        return spans[0]

    @staticmethod
    def _attribute(span: ReadableSpan, key: str) -> Optional[object]:
        """Read one attribute off a finished span.

        Args:
            span: The exported span.
            key: The attribute name.
        """
        return (span.attributes or {}).get(key)

    def test_session_id_read_from_parent_context_when_ambient_context_has_none(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """A span parented onto a baggage-carrying context is stamped from that context."""
        assert baggage.get_baggage("session_id") is None, "ambient context must be bare for this test to mean anything"
        parent_context = _context_with_baggage(session_id="RM_room_sid")

        with tracer.start_as_current_span("agent_turn", context=parent_context):
            pass

        assert self._attribute(self._only_span(span_exporter), ATTR_SESSION_ID) == "RM_room_sid"

    def test_session_id_read_from_context_snapshotted_before_it_was_detached(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """The LiveKit shape: baggage is attached, snapshotted, detached, then used as a parent.

        Reproduces a turn triggered from the entrypoint task after
        ``await session.start(...)`` has returned and unwound the session scope —
        the ambient context is bare, but the snapshot LiveKit kept is not.
        """
        token = otel_context.attach(baggage.set_baggage("session_id", "RM_room_sid"))
        root_span_context = otel_context.get_current()
        otel_context.detach(token)
        assert baggage.get_baggage("session_id") is None, "detach must have unwound the ambient baggage"

        with tracer.start_as_current_span("agent_turn", context=root_span_context):
            pass

        assert self._attribute(self._only_span(span_exporter), ATTR_SESSION_ID) == "RM_room_sid"

    def test_ambient_session_id_wins_over_parent_context(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """A later ``Netra.set_session_id()`` still overrides an earlier snapshot.

        The fallback must be additive only: it may fill in a missing id, never
        replace one the ambient context already resolved.
        """
        parent_context = _context_with_baggage(session_id="snapshotted-at-session-start")

        with _ambient_baggage(session_id="set-by-the-user-later"):
            with tracer.start_as_current_span("agent_turn", context=parent_context):
                pass

        assert self._attribute(self._only_span(span_exporter), ATTR_SESSION_ID) == "set-by-the-user-later"

    def test_session_id_still_resolves_from_ambient_context_with_no_parent_context(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """The ordinary path — no explicit parent context — is unchanged."""
        with _ambient_baggage(session_id="ambient-session"):
            with tracer.start_as_current_span("llm_request"):
                pass

        assert self._attribute(self._only_span(span_exporter), ATTR_SESSION_ID) == "ambient-session"

    def test_user_tenant_and_custom_keys_also_resolve_from_parent_context(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """Every session field falls back, not just the session id."""
        parent_context = _context_with_baggage(
            user_id="user-42",
            tenant_id="tenant-7",
            custom_keys="campaign",
            **{"custom.campaign": "spring-sale"},
        )

        with tracer.start_as_current_span("agent_turn", context=parent_context):
            pass

        span = self._only_span(span_exporter)
        assert self._attribute(span, ATTR_USER_ID) == "user-42"
        assert self._attribute(span, ATTR_TENANT_ID) == "tenant-7"
        assert self._attribute(span, "netra.custom.campaign") == "spring-sale"

    def test_no_session_attributes_when_neither_context_carries_baggage(
        self, tracer: trace.Tracer, span_exporter: InMemorySpanExporter
    ) -> None:
        """A parent context without baggage adds nothing."""
        with tracer.start_as_current_span("agent_turn", context=Context()):
            pass

        span = self._only_span(span_exporter)
        assert self._attribute(span, ATTR_SESSION_ID) is None
        assert self._attribute(span, ATTR_USER_ID) is None
        assert self._attribute(span, ATTR_TENANT_ID) is None
