"""
Unit tests for SessionManager class.
Tests the happy path scenarios for session management functionality.
"""

from unittest.mock import Mock, patch

from netra.config import Config
from netra.session_manager import SessionManager


class TestSessionManagerSpanTracking:
    """Test span tracking functionality."""

    def test_set_and_get_current_span(self):
        """Test setting and getting current span."""
        # Arrange
        mock_span = Mock()

        # Act
        SessionManager.set_current_span(mock_span)
        result = SessionManager.get_current_span()

        # Assert
        assert result == mock_span

    def test_get_current_span_when_none(self):
        """Test getting current span when none is set."""
        # Arrange
        SessionManager.set_current_span(None)

        # Act
        result = SessionManager.get_current_span()

        # Assert
        assert result is None


class TestSessionManagerContext:
    """Test session context management."""

    @patch("netra.session_manager.otel_context")
    @patch("netra.session_manager.baggage")
    def test_set_session_id_context(self, mock_baggage, mock_context):
        """Test setting session ID in context."""
        # Arrange
        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx
        mock_baggage.set_baggage.return_value = mock_ctx

        # Act
        SessionManager.set_session_context("session_id", "test-session-123")

        # Assert
        mock_context.get_current.assert_called_once()
        mock_baggage.set_baggage.assert_called_once_with("session_id", "test-session-123", mock_ctx)
        mock_context.attach.assert_called_once_with(mock_ctx)

    @patch("netra.session_manager.otel_context")
    @patch("netra.session_manager.baggage")
    def test_set_user_id_context(self, mock_baggage, mock_context):
        """Test setting user ID in context."""
        # Arrange
        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx
        mock_baggage.set_baggage.return_value = mock_ctx

        # Act
        SessionManager.set_session_context("user_id", "user-456")

        # Assert
        mock_context.get_current.assert_called_once()
        mock_baggage.set_baggage.assert_called_once_with("user_id", "user-456", mock_ctx)
        mock_context.attach.assert_called_once_with(mock_ctx)

    @patch("netra.session_manager.otel_context")
    @patch("netra.session_manager.baggage")
    def test_set_tenant_id_context(self, mock_baggage, mock_context):
        """Test setting tenant ID in context."""
        # Arrange
        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx
        mock_baggage.set_baggage.return_value = mock_ctx

        # Act
        SessionManager.set_session_context("tenant_id", "tenant-789")

        # Assert
        mock_context.get_current.assert_called_once()
        mock_baggage.set_baggage.assert_called_once_with("tenant_id", "tenant-789", mock_ctx)
        mock_context.attach.assert_called_once_with(mock_ctx)

    @patch("netra.session_manager.otel_context")
    @patch("netra.session_manager.baggage")
    def test_set_attribute_on_active_span_serializes_and_sets(self, mock_baggage, mock_context):
        """Test setting custom attribute on active span (no baggage)."""
        # Arrange: mock an active recording span
        with patch("netra.session_manager.trace") as mock_trace:
            mock_span = Mock()
            mock_span.is_recording.return_value = True
            mock_trace.get_current_span.return_value = mock_span

            # Act: set attributes of various types
            SessionManager.set_attribute_on_active_span(f"{Config.LIBRARY_NAME}.custom.model", "gpt-4")
            SessionManager.set_attribute_on_active_span(f"{Config.LIBRARY_NAME}.custom.temperature", 0.7)
            SessionManager.set_attribute_on_active_span(f"{Config.LIBRARY_NAME}.custom.params", {"a": 1, "b": [1, 2]})

            # Assert: attributes set on span, not baggage
            calls = [c.args for c in mock_span.set_attribute.call_args_list]
            assert (f"{Config.LIBRARY_NAME}.custom.model", "gpt-4") in calls
            assert (f"{Config.LIBRARY_NAME}.custom.temperature", "0.7") in calls
            # dict gets JSON-serialized
            assert (f"{Config.LIBRARY_NAME}.custom.params", '{"a": 1, "b": [1, 2]}') in calls
            mock_baggage.set_baggage.assert_not_called()


class TestSessionManagerCustomEvents:
    """Test custom event functionality."""

    @patch("netra.session_manager.datetime")
    def test_set_custom_event_with_current_span(self, mock_datetime):
        """Test setting custom event when current span exists."""
        # Arrange
        mock_span = Mock()
        mock_datetime.now.return_value.timestamp.return_value = 1234567890.123456789
        expected_timestamp = int(1234567890.123456789 * 1_000_000_000)
        SessionManager.set_current_span(mock_span)

        attributes = {"key1": "value1", "key2": "value2"}

        # Act
        SessionManager.set_custom_event("test_event", attributes)

        # Assert
        mock_span.add_event.assert_called_once_with(
            name="test_event", attributes=attributes, timestamp=expected_timestamp
        )

    @patch("netra.session_manager.datetime")
    @patch("netra.session_manager.trace")
    @patch("netra.session_manager.otel_context")
    @patch("netra.session_manager.Config")
    def test_set_custom_event_without_current_span(self, mock_config, mock_context, mock_trace, mock_datetime):
        """Test setting custom event when no current span exists (fallback)."""
        # Arrange
        SessionManager.set_current_span(None)
        mock_datetime.now.return_value.timestamp.return_value = 1234567890.123456789
        expected_timestamp = int(1234567890.123456789 * 1_000_000_000)

        mock_ctx = Mock()
        mock_context.get_current.return_value = mock_ctx

        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        mock_config.LIBRARY_NAME = "netra"

        attributes = {"error": "test_error"}

        # Act
        SessionManager.set_custom_event("error_event", attributes)

        # Assert
        mock_context.get_current.assert_called_once()
        mock_trace.get_tracer.assert_called_once_with("netra.session_manager")
        mock_tracer.start_as_current_span.assert_called_once_with("netra.error_event", context=mock_ctx)
        mock_span.add_event.assert_called_once_with(
            name="error_event", attributes=attributes, timestamp=expected_timestamp
        )

    def test_set_custom_event_with_empty_attributes(self):
        """Test setting custom event with empty attributes."""
        # Arrange
        mock_span = Mock()
        SessionManager.set_current_span(mock_span)

        # Act
        SessionManager.set_custom_event("empty_event", {})

        # Assert
        mock_span.add_event.assert_called_once()
        call_args = mock_span.add_event.call_args
        assert call_args[1]["name"] == "empty_event"
        assert call_args[1]["attributes"] == {}
        assert "timestamp" in call_args[1]
