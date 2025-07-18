"""
Unit tests for ErrorDetectionProcessor class.
Tests focusing on core functionality, error detection, and span wrapping.
"""

from unittest.mock import Mock, patch

from netra.processors.error_detection_processor import ErrorDetectionProcessor


class TestErrorDetectionProcessor:
    """Test ErrorDetectionProcessor core functionality."""

    def test_initialization(self):
        """Test ErrorDetectionProcessor initialization."""
        # Act
        processor = ErrorDetectionProcessor()

        # Assert
        assert processor is not None
        assert hasattr(processor, "on_start")
        assert hasattr(processor, "on_end")
        assert hasattr(processor, "force_flush")
        assert hasattr(processor, "shutdown")

    def test_get_span_id_success(self):
        """Test _get_span_id method with valid span."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        # Act
        result = processor._get_span_id(mock_span)

        # Assert
        expected_span_id = f"{123456789:032x}-{987654321:016x}"
        assert result == expected_span_id
        mock_span.get_span_context.assert_called_once()

    def test_get_span_id_with_exception(self):
        """Test _get_span_id method when span context raises exception."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span.get_span_context.side_effect = Exception("Test exception")

        # Act
        result = processor._get_span_id(mock_span)

        # Assert
        assert result is None
        mock_span.get_span_context.assert_called_once()

    @patch("netra.processors.error_detection_processor.httpx")
    @patch("netra.processors.error_detection_processor.Netra")
    def test_status_code_processing_with_error(self, mock_netra, mock_httpx):
        """Test _status_code_processing method with error status code."""
        # Arrange
        processor = ErrorDetectionProcessor()
        error_status_code = 500
        mock_httpx.codes.is_error.return_value = True

        # Act
        processor._status_code_processing(error_status_code)

        # Assert
        mock_httpx.codes.is_error.assert_called_once_with(error_status_code)
        mock_netra.set_custom_event.assert_called_once_with(
            event_name="error_detected", attributes={"has_error": True, "status_code": error_status_code}
        )

    @patch("netra.processors.error_detection_processor.httpx")
    @patch("netra.processors.error_detection_processor.Netra")
    def test_status_code_processing_with_success(self, mock_netra, mock_httpx):
        """Test _status_code_processing method with success status code."""
        # Arrange
        processor = ErrorDetectionProcessor()
        success_status_code = 200
        mock_httpx.codes.is_error.return_value = False

        # Act
        processor._status_code_processing(success_status_code)

        # Assert
        mock_httpx.codes.is_error.assert_called_once_with(success_status_code)
        mock_netra.set_custom_event.assert_not_called()

    @patch("netra.processors.error_detection_processor.httpx")
    @patch("netra.processors.error_detection_processor.Netra")
    def test_wrap_span_methods_http_status_code_error(self, mock_netra, mock_httpx):
        """Test _wrap_span_methods with HTTP status code that triggers error detection."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        original_set_attribute = Mock()
        mock_span.set_attribute = original_set_attribute
        span_id = "test-span-id"

        mock_httpx.codes.is_error.return_value = True

        # Act
        processor._wrap_span_methods(mock_span, span_id)

        # Call the wrapped method
        mock_span.set_attribute("http.status_code", 404)

        # Assert
        mock_httpx.codes.is_error.assert_called_once_with(404)
        mock_netra.set_custom_event.assert_called_once_with(
            event_name="error_detected", attributes={"has_error": True, "status_code": 404}
        )
        original_set_attribute.assert_called_once_with("http.status_code", 404)

    def test_wrap_span_methods_non_status_code_attribute(self):
        """Test _wrap_span_methods with non-status code attribute."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        original_set_attribute = Mock()
        mock_span.set_attribute = original_set_attribute
        span_id = "test-span-id"

        # Act
        processor._wrap_span_methods(mock_span, span_id)

        # Call the wrapped method with non-status code attribute
        mock_span.set_attribute("custom.attribute", "value")

        # Assert
        original_set_attribute.assert_called_once_with("custom.attribute", "value")

    def test_on_start_with_valid_span(self):
        """Test on_start method with valid span."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        with patch.object(processor, "_wrap_span_methods") as mock_wrap:
            # Act
            processor.on_start(mock_span)

            # Assert
            expected_span_id = f"{123456789:032x}-{987654321:016x}"
            mock_wrap.assert_called_once_with(mock_span, expected_span_id)

    def test_on_start_with_invalid_span(self):
        """Test on_start method with invalid span (no span ID)."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span.get_span_context.side_effect = Exception("Test exception")

        with patch.object(processor, "_wrap_span_methods") as mock_wrap:
            # Act
            processor.on_start(mock_span)

            # Assert
            mock_wrap.assert_not_called()

    def test_on_start_with_parent_context(self):
        """Test on_start method with parent context."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_parent_context = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        with patch.object(processor, "_wrap_span_methods") as mock_wrap:
            # Act
            processor.on_start(mock_span, mock_parent_context)

            # Assert
            expected_span_id = f"{123456789:032x}-{987654321:016x}"
            mock_wrap.assert_called_once_with(mock_span, expected_span_id)

    def test_on_end_method(self):
        """Test on_end method (no-op implementation)."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()

        # Act & Assert (should not raise any exception)
        processor.on_end(mock_span)

    def test_force_flush_method(self):
        """Test force_flush method."""
        # Arrange
        processor = ErrorDetectionProcessor()

        # Act
        result = processor.force_flush()

        # Assert
        assert result is True

    def test_force_flush_with_timeout(self):
        """Test force_flush method with timeout parameter."""
        # Arrange
        processor = ErrorDetectionProcessor()

        # Act
        result = processor.force_flush(timeout_millis=5000)

        # Assert
        assert result is True

    def test_shutdown_method(self):
        """Test shutdown method."""
        # Arrange
        processor = ErrorDetectionProcessor()

        # Act
        result = processor.shutdown()

        # Assert
        assert result is True

    @patch("netra.processors.error_detection_processor.httpx")
    @patch("netra.processors.error_detection_processor.Netra")
    def test_integration_error_detection_flow(self, mock_netra, mock_httpx):
        """Test complete error detection flow from on_start to error detection."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        original_set_attribute = Mock()
        mock_span.set_attribute = original_set_attribute

        mock_httpx.codes.is_error.return_value = True

        # Act
        processor.on_start(mock_span)

        # Simulate setting an error status code
        mock_span.set_attribute("http.status_code", 500)

        # Assert
        mock_httpx.codes.is_error.assert_called_once_with(500)
        mock_netra.set_custom_event.assert_called_once_with(
            event_name="error_detected", attributes={"has_error": True, "status_code": 500}
        )
        original_set_attribute.assert_called_once_with("http.status_code", 500)

    @patch("netra.processors.error_detection_processor.httpx")
    @patch("netra.processors.error_detection_processor.Netra")
    def test_multiple_error_status_codes(self, mock_netra, mock_httpx):
        """Test handling multiple error status codes."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 123456789
        mock_span_context.span_id = 987654321
        mock_span.get_span_context.return_value = mock_span_context

        original_set_attribute = Mock()
        mock_span.set_attribute = original_set_attribute

        mock_httpx.codes.is_error.return_value = True

        # Act
        processor.on_start(mock_span)

        # Simulate setting multiple error status codes
        mock_span.set_attribute("http.status_code", 404)
        mock_span.set_attribute("http.status_code", 500)

        # Assert
        assert mock_httpx.codes.is_error.call_count == 2
        mock_httpx.codes.is_error.assert_any_call(404)
        mock_httpx.codes.is_error.assert_any_call(500)

        assert mock_netra.set_custom_event.call_count == 2
        mock_netra.set_custom_event.assert_any_call(
            event_name="error_detected", attributes={"has_error": True, "status_code": 404}
        )
        mock_netra.set_custom_event.assert_any_call(
            event_name="error_detected", attributes={"has_error": True, "status_code": 500}
        )

    def test_edge_case_zero_trace_id(self):
        """Test _get_span_id with zero trace_id and span_id."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 0
        mock_span_context.span_id = 0
        mock_span.get_span_context.return_value = mock_span_context

        # Act
        result = processor._get_span_id(mock_span)

        # Assert
        expected_span_id = f"{0:032x}-{0:016x}"
        assert result == expected_span_id
        assert result == "00000000000000000000000000000000-0000000000000000"

    def test_edge_case_large_trace_id(self):
        """Test _get_span_id with maximum trace_id and span_id values."""
        # Arrange
        processor = ErrorDetectionProcessor()
        mock_span = Mock()
        mock_span_context = Mock()
        mock_span_context.trace_id = 2**128 - 1  # Max 128-bit value
        mock_span_context.span_id = 2**64 - 1  # Max 64-bit value
        mock_span.get_span_context.return_value = mock_span_context

        # Act
        result = processor._get_span_id(mock_span)

        # Assert
        expected_span_id = f"{2 ** 128 - 1:032x}-{2 ** 64 - 1:016x}"
        assert result == expected_span_id
        assert result == "ffffffffffffffffffffffffffffffff-ffffffffffffffff"
