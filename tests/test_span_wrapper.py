"""
Unit tests for the netra.span_wrapper module.

This module tests the SpanWrapper class and ATTRIBUTE constants used for
tracking observability data with OpenTelemetry integration.
"""

from unittest.mock import Mock, patch

from opentelemetry.trace import SpanKind, StatusCode

from netra.config import Config
from netra.span_wrapper import ATTRIBUTE, SpanWrapper


class TestUsageModel:
    """Test cases for the UsageModel class."""

    def test_usage_model_initialization_with_optional_fields(self):
        """Test UsageModel initialization with optional fields set to None."""
        from netra.span_wrapper import UsageModel

        # Test with all fields provided
        usage = UsageModel(model="gpt-4", usage_type="text", units_used=1000, cost_in_usd=0.02)
        assert usage.model == "gpt-4"
        assert usage.usage_type == "text"
        assert usage.units_used == 1000
        assert usage.cost_in_usd == 0.02

    def test_usage_model_without_optional_fields(self):
        """Test UsageModel initialization without optional fields."""
        from netra.span_wrapper import UsageModel

        # Test with optional fields not provided (should default to None)
        usage = UsageModel(model="gpt-4", usage_type="text")
        assert usage.model == "gpt-4"
        assert usage.usage_type == "text"
        assert usage.units_used is None
        assert usage.cost_in_usd is None


class TestATTRIBUTE:
    """Test cases for the ATTRIBUTE constants class."""

    def test_attribute_constants_exist(self):
        """Test that all expected attribute constants are defined."""
        expected_attributes = [
            "LLM_SYSTEM",
            "MODEL",
            "PROMPT",
            "NEGATIVE_PROMPT",
            "STATUS",
            "DURATION_MS",
            "ERROR_MESSAGE",
        ]

        for attr_name in expected_attributes:
            assert hasattr(ATTRIBUTE, attr_name), f"ATTRIBUTE.{attr_name} should be defined"
            assert isinstance(getattr(ATTRIBUTE, attr_name), str), f"ATTRIBUTE.{attr_name} should be a string"

    def test_attribute_values(self):
        """Test that attribute constants have expected string values."""
        assert ATTRIBUTE.LLM_SYSTEM == "llm_system"
        assert ATTRIBUTE.MODEL == "model"
        assert ATTRIBUTE.PROMPT == "prompt"
        assert ATTRIBUTE.NEGATIVE_PROMPT == "negative_prompt"

        assert ATTRIBUTE.STATUS == "status"
        assert ATTRIBUTE.DURATION_MS == "duration_ms"
        assert ATTRIBUTE.ERROR_MESSAGE == "error_message"


class TestSpanWrapperInitialization:
    """Test cases for SpanWrapper class initialization."""

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_span_wrapper_initialization_with_defaults(self, mock_get_tracer):
        """Test SpanWrapper initialization with default parameters."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        span_wrapper = SpanWrapper("test_span")

        assert span_wrapper.name == "test_span"
        assert span_wrapper.attributes == {}
        assert span_wrapper.start_time is None
        assert span_wrapper.end_time is None
        assert span_wrapper.status == "pending"
        assert span_wrapper.error_message is None
        assert span_wrapper.module_name == "combat_sdk"
        assert span_wrapper.tracer == mock_tracer
        assert span_wrapper.span is None
        assert span_wrapper.context_token is None

        mock_get_tracer.assert_called_once_with("combat_sdk")

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_span_wrapper_initialization_with_custom_parameters(self, mock_get_tracer):
        """Test SpanWrapper initialization with custom parameters."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        custom_attributes = {"key1": "value1", "key2": "value2"}
        span_wrapper = SpanWrapper("custom_span", attributes=custom_attributes, module_name="custom_module")

        assert span_wrapper.name == "custom_span"
        assert span_wrapper.attributes == custom_attributes
        assert span_wrapper.module_name == "custom_module"
        assert span_wrapper.tracer == mock_tracer

        mock_get_tracer.assert_called_once_with("custom_module")

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_span_wrapper_initialization_with_none_attributes(self, mock_get_tracer):
        """Test SpanWrapper initialization with None attributes defaults to empty dict."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        span_wrapper = SpanWrapper("test_span", attributes=None)

        assert span_wrapper.attributes == {}


class TestSpanWrapperAttributeSetters:
    """Test cases for SpanWrapper attribute setter methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.span_wrapper.trace.get_tracer"):
            self.span_wrapper = SpanWrapper("test_span")

    def test_set_attribute_basic(self):
        """Test basic set_attribute functionality."""
        result = self.span_wrapper.set_attribute("test_key", "test_value")

        assert self.span_wrapper.attributes["test_key"] == "test_value"
        assert result is self.span_wrapper  # Method chaining

    def test_set_attribute_with_span(self):
        """Test set_attribute when span exists."""
        mock_span = Mock()
        self.span_wrapper.span = mock_span

        result = self.span_wrapper.set_attribute("test_key", "test_value")

        assert self.span_wrapper.attributes["test_key"] == "test_value"
        mock_span.set_attribute.assert_called_once_with("test_key", "test_value")
        assert result is self.span_wrapper

    def test_set_prompt(self):
        """Test set_prompt method."""
        result = self.span_wrapper.set_prompt("Test prompt")

        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.PROMPT}"] == "Test prompt"
        assert result is self.span_wrapper

    def test_set_negative_prompt(self):
        """Test set_negative_prompt method."""
        result = self.span_wrapper.set_negative_prompt("Negative prompt")

        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.NEGATIVE_PROMPT}"] == "Negative prompt"
        assert result is self.span_wrapper

    def test_set_model(self):
        """Test set_model method."""
        result = self.span_wrapper.set_model("gpt-4")

        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.MODEL}"] == "gpt-4"
        assert result is self.span_wrapper

    def test_set_llm_system(self):
        """Test set_llm_system method."""
        result = self.span_wrapper.set_llm_system("openai")

        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.LLM_SYSTEM}"] == "openai"
        assert result is self.span_wrapper


class TestSpanWrapperStatusMethods:
    """Test cases for SpanWrapper status and error management methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.span_wrapper.trace.get_tracer"):
            self.span_wrapper = SpanWrapper("test_span")

    def test_set_error_without_span(self):
        """Test set_error method when no span exists."""
        result = self.span_wrapper.set_error("Test error message")

        assert self.span_wrapper.status == "error"
        assert self.span_wrapper.error_message == "Test error message"
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}"] == "Test error message"
        assert result is self.span_wrapper

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_set_success_with_span(self, mock_get_tracer):
        """Test set_success method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        span_wrapper = SpanWrapper("test")
        span_wrapper.span = mock_span

        # Act
        result = span_wrapper.set_success()

        # Assert
        assert result is span_wrapper
        assert span_wrapper.status == "success"
        # set_success only sets span status, not attributes
        mock_span.set_attribute.assert_not_called()

        # Verify span status - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK

    def test_set_success_without_span(self):
        """Test set_success method when no span exists."""
        result = self.span_wrapper.set_success()

        assert self.span_wrapper.status == "success"
        assert result is self.span_wrapper

    def test_get_current_span(self):
        """Test get_current_span method."""
        mock_span = Mock()
        self.span_wrapper.span = mock_span

        result = self.span_wrapper.get_current_span()

        assert result is mock_span

    def test_get_current_span_none(self):
        """Test get_current_span method when span is None."""
        result = self.span_wrapper.get_current_span()

        assert result is None


class TestSpanWrapperEventMethods:
    """Test cases for SpanWrapper event handling methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.span_wrapper.trace.get_tracer"):
            self.span_wrapper = SpanWrapper("test_span")

    def test_add_event_without_span(self):
        """Test add_event method when no span exists."""
        result = self.span_wrapper.add_event("test_event", {"key": "value"})

        # Should not raise an error and return self for chaining
        assert result is self.span_wrapper

    def test_add_event_with_span_and_attributes(self):
        """Test add_event method when span exists with attributes."""
        mock_span = Mock()
        self.span_wrapper.span = mock_span

        attributes = {"key1": "value1", "key2": "value2"}
        result = self.span_wrapper.add_event("test_event", attributes)

        mock_span.add_event.assert_called_once_with("test_event", attributes)
        assert result is self.span_wrapper

    def test_add_event_with_span_no_attributes(self):
        """Test add_event method when span exists without attributes."""
        mock_span = Mock()
        self.span_wrapper.span = mock_span

        result = self.span_wrapper.add_event("test_event")

        mock_span.add_event.assert_called_once_with("test_event", {})
        assert result is self.span_wrapper


class TestSpanWrapperContextManager:
    """Test cases for SpanWrapper context manager behavior (__enter__ and __exit__)."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_tracer = Mock()
        self.mock_span = Mock()
        self.mock_tracer.start_span.return_value = self.mock_span

        with patch("netra.span_wrapper.trace.get_tracer", return_value=self.mock_tracer):
            self.span_wrapper = SpanWrapper("test_span", {"initial_key": "initial_value"})

    @patch("netra.span_wrapper.time.time")
    @patch("netra.span_wrapper.set_span_in_context")
    @patch("netra.span_wrapper.context_api.attach")
    @patch("netra.span_wrapper.logger")
    def test_enter_method(self, mock_logger, mock_attach, mock_set_span_in_context, mock_time):
        """Test __enter__ method functionality."""
        mock_time.return_value = 1234567890.123
        mock_context = Mock()
        mock_set_span_in_context.return_value = mock_context
        mock_token = Mock()
        mock_attach.return_value = mock_token

        result = self.span_wrapper.__enter__()

        # Verify time tracking
        assert self.span_wrapper.start_time == 1234567890.123

        # Verify span creation
        self.mock_tracer.start_span.assert_called_once_with(
            name="test_span", kind=SpanKind.CLIENT, attributes={"initial_key": "initial_value"}
        )
        assert self.span_wrapper.span is self.mock_span

        # Verify context management
        mock_set_span_in_context.assert_called_once_with(self.mock_span)
        mock_attach.assert_called_once_with(mock_context)
        assert self.span_wrapper.context_token is mock_token

        # Verify logging
        mock_logger.info.assert_called_once_with("Started span wrapper: test_span")

        # Verify return value
        assert result is self.span_wrapper

    @patch("netra.span_wrapper.time.time")
    @patch("netra.span_wrapper.context_api.detach")
    @patch("netra.span_wrapper.logger")
    def test_exit_method_success_no_exception(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method with successful completion (no exception)."""
        # Setup span wrapper state
        self.span_wrapper.start_time = 1234567890.123
        mock_time.return_value = 1234567890.623  # 500ms later
        self.span_wrapper.span = self.mock_span
        mock_token = Mock()
        self.span_wrapper.context_token = mock_token

        result = self.span_wrapper.__exit__(None, None, None)

        # Verify time tracking
        assert self.span_wrapper.end_time == 1234567890.623
        duration_ms = (self.span_wrapper.end_time - self.span_wrapper.start_time) * 1000
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}"] == str(
            round(duration_ms, 2)
        )

        # Verify status
        assert self.span_wrapper.status == "success"
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}"] == "success"
        # Verify span status - check call arguments instead of Status object equality
        self.mock_span.set_status.assert_called_once()
        call_args = self.mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK

        self.mock_span.set_attribute.assert_any_call(
            f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}", str(round(duration_ms, 2))
        )
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}", "success")

        # Verify span and context cleanup
        self.mock_span.end.assert_called_once()
        mock_detach.assert_called_once_with(mock_token)

        # Verify logging
        mock_logger.info.assert_called_once_with("Ended span wrapper: test_span (Status: success, Duration: 500.00ms)")

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.span_wrapper.time.time")
    @patch("netra.span_wrapper.context_api.detach")
    @patch("netra.span_wrapper.logger")
    def test_exit_method_with_exception(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when an exception occurs."""
        # Setup span wrapper state
        self.span_wrapper.start_time = 1234567890.123
        mock_time.return_value = 1234567890.323  # 200ms later
        self.span_wrapper.span = self.mock_span
        mock_token = Mock()
        self.span_wrapper.context_token = mock_token

        # Create test exception
        test_exception = ValueError("Test error")

        result = self.span_wrapper.__exit__(ValueError, test_exception, None)

        # Verify time tracking
        assert self.span_wrapper.end_time == 1234567890.323
        duration_ms = (self.span_wrapper.end_time - self.span_wrapper.start_time) * 1000
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}"] == str(
            round(duration_ms, 2)
        )

        # Verify error handling
        assert self.span_wrapper.status == "error"
        assert self.span_wrapper.error_message == "Test error"
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}"] == "Test error"
        assert self.span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}"] == "error"

        # Verify span error handling - check call arguments instead of Status object equality
        self.mock_span.set_status.assert_called_once()
        call_args = self.mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Test error"

        self.mock_span.record_exception.assert_called_once_with(test_exception)
        self.mock_span.set_attribute.assert_any_call(
            f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}", str(round(duration_ms, 2))
        )
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}", "error")
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}", "Test error")

        # Verify logging
        mock_logger.error.assert_called_once_with("Span wrapper test_span failed: Test error")
        mock_logger.info.assert_called_once_with("Ended span wrapper: test_span (Status: error, Duration: 200.00ms)")

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.span_wrapper.time.time")
    @patch("netra.span_wrapper.context_api.detach")
    @patch("netra.span_wrapper.logger")
    def test_exit_method_with_manual_status_change(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when status was manually changed before exit."""
        # Setup span wrapper state with manual status change
        self.span_wrapper.start_time = 1234567890.123
        mock_time.return_value = 1234567890.223  # 100ms later
        self.span_wrapper.span = self.mock_span
        self.span_wrapper.status = "custom_status"  # Manually set status
        self.span_wrapper.error_message = "custom_error"
        mock_token = Mock()
        self.span_wrapper.context_token = mock_token

        result = self.span_wrapper.__exit__(None, None, None)

        # Verify that status is not overridden to "success"
        assert self.span_wrapper.status == "custom_status"
        assert self.span_wrapper.error_message == "custom_error"

        # Verify span operations - only duration_ms and status are set, not error_message since no exception occurred
        self.mock_span.set_attribute.assert_any_call(
            f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}", str(round((1234567890.223 - 1234567890.123) * 1000, 2))
        )
        self.mock_span.set_attribute.assert_any_call(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}", "custom_status")
        # error_message is NOT set by __exit__ when there's no exception, even if manually set
        # The span status is set based on the error_message if it exists
        self.mock_span.set_status.assert_not_called()  # Fix: span.set_status should not be called

        # Verify span and context cleanup
        self.mock_span.end.assert_called_once()
        mock_detach.assert_called_once_with(mock_token)

        # Verify logging
        mock_logger.info.assert_called_once_with(
            "Ended span wrapper: test_span (Status: custom_status, Duration: 100.00ms)"
        )

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.span_wrapper.time.time")
    @patch("netra.span_wrapper.context_api.detach")
    @patch("netra.span_wrapper.logger")
    def test_exit_method_no_start_time(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when start_time is None."""
        # Setup span wrapper state without start_time
        self.span_wrapper.start_time = None
        self.span_wrapper.span = self.mock_span

        result = self.span_wrapper.__exit__(None, None, None)

        # Verify duration is not calculated
        assert f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}" not in self.span_wrapper.attributes

        # Verify other functionality still works
        assert self.span_wrapper.status == "success"

        # Verify logging without duration
        mock_logger.info.assert_called_once_with("Ended span wrapper: test_span (Status: success)")

        # Verify return value (should not suppress exceptions)
        assert result is False


class TestSpanWrapperIntegration:
    """Integration tests for SpanWrapper as a context manager."""

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_set_error_with_span(self, mock_get_tracer):
        """Test set_error method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        span_wrapper = SpanWrapper("test")
        span_wrapper.span = mock_span

        # Act
        result = span_wrapper.set_error("Test error")

        # Assert
        assert result is span_wrapper
        assert span_wrapper.status == "error"
        assert span_wrapper.error_message == "Test error"
        assert span_wrapper.attributes[f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}"] == "Test error"

        # Verify span operations - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Test error"
        mock_span.set_attribute.assert_called_once_with(
            f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}", "Test error"
        )

    @patch("netra.span_wrapper.trace.get_tracer")
    def test_set_success_with_span(self, mock_get_tracer):
        """Test set_success method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        span_wrapper = SpanWrapper("test")
        span_wrapper.span = mock_span

        # Act
        result = span_wrapper.set_success()

        # Assert
        assert result is span_wrapper
        assert span_wrapper.status == "success"
        # set_success only sets span status, not attributes
        mock_span.set_attribute.assert_not_called()

        # Verify span status - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK
