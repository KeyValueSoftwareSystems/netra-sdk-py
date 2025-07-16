"""
Unit tests for SessionSpanProcessor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from unittest.mock import Mock, patch

from netra.processors.session_span_processor import SessionSpanProcessor


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
