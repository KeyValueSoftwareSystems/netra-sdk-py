"""
Unit tests for the netra.session module.

This module tests the Session class and ATTRIBUTE constants used for
tracking observability data with OpenTelemetry integration.
"""

from unittest.mock import Mock, patch

from opentelemetry.trace import SpanKind, StatusCode

from netra.session import ATTRIBUTE, Session


class TestATTRIBUTE:
    """Test cases for the ATTRIBUTE constants class."""

    def test_attribute_constants_exist(self):
        """Test that all expected attribute constants are defined."""
        expected_attributes = [
            "LLM_SYSTEM",
            "MODEL",
            "PROMPT",
            "NEGATIVE_PROMPT",
            "IMAGE_HEIGHT",
            "IMAGE_WIDTH",
            "TOKENS",
            "CREDITS",
            "COST",
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
        assert ATTRIBUTE.IMAGE_HEIGHT == "image_height"
        assert ATTRIBUTE.IMAGE_WIDTH == "image_width"
        assert ATTRIBUTE.TOKENS == "tokens"
        assert ATTRIBUTE.CREDITS == "credits"
        assert ATTRIBUTE.COST == "cost"
        assert ATTRIBUTE.STATUS == "status"
        assert ATTRIBUTE.DURATION_MS == "duration_ms"
        assert ATTRIBUTE.ERROR_MESSAGE == "error_message"


class TestSessionInitialization:
    """Test cases for Session class initialization."""

    @patch("netra.session.trace.get_tracer")
    def test_session_initialization_with_defaults(self, mock_get_tracer):
        """Test Session initialization with default parameters."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        session = Session("test_session")

        assert session.name == "test_session"
        assert session.attributes == {}
        assert session.start_time is None
        assert session.end_time is None
        assert session.status == "pending"
        assert session.error_message is None
        assert session.module_name == "combat_sdk"
        assert session.tracer == mock_tracer
        assert session.span is None
        assert session.context_token is None

        mock_get_tracer.assert_called_once_with("combat_sdk")

    @patch("netra.session.trace.get_tracer")
    def test_session_initialization_with_custom_parameters(self, mock_get_tracer):
        """Test Session initialization with custom parameters."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        custom_attributes = {"key1": "value1", "key2": "value2"}
        session = Session("custom_session", attributes=custom_attributes, module_name="custom_module")

        assert session.name == "custom_session"
        assert session.attributes == custom_attributes
        assert session.module_name == "custom_module"
        assert session.tracer == mock_tracer

        mock_get_tracer.assert_called_once_with("custom_module")

    @patch("netra.session.trace.get_tracer")
    def test_session_initialization_with_none_attributes(self, mock_get_tracer):
        """Test Session initialization with None attributes defaults to empty dict."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        session = Session("test_session", attributes=None)

        assert session.attributes == {}


class TestSessionAttributeSetters:
    """Test cases for Session attribute setter methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.session.trace.get_tracer"):
            self.session = Session("test_session")

    def test_set_attribute_basic(self):
        """Test basic set_attribute functionality."""
        result = self.session.set_attribute("test_key", "test_value")

        assert self.session.attributes["test_key"] == "test_value"
        assert result is self.session  # Method chaining

    def test_set_attribute_with_span(self):
        """Test set_attribute when span exists."""
        mock_span = Mock()
        self.session.span = mock_span

        result = self.session.set_attribute("test_key", "test_value")

        assert self.session.attributes["test_key"] == "test_value"
        mock_span.set_attribute.assert_called_once_with("test_key", "test_value")
        assert result is self.session

    def test_set_prompt(self):
        """Test set_prompt method."""
        result = self.session.set_prompt("Test prompt")

        assert self.session.attributes[ATTRIBUTE.PROMPT] == "Test prompt"
        assert result is self.session

    def test_set_negative_prompt(self):
        """Test set_negative_prompt method."""
        result = self.session.set_negative_prompt("Negative prompt")

        assert self.session.attributes[ATTRIBUTE.NEGATIVE_PROMPT] == "Negative prompt"
        assert result is self.session

    def test_set_image_height(self):
        """Test set_image_height method."""
        result = self.session.set_image_height("1024")

        assert self.session.attributes[ATTRIBUTE.IMAGE_HEIGHT] == "1024"
        assert result is self.session

    def test_set_image_width(self):
        """Test set_image_width method."""
        result = self.session.set_image_width("768")

        assert self.session.attributes[ATTRIBUTE.IMAGE_WIDTH] == "768"
        assert result is self.session

    def test_set_tokens(self):
        """Test set_tokens method."""
        result = self.session.set_tokens("100")

        assert self.session.attributes[ATTRIBUTE.TOKENS] == "100"
        assert result is self.session

    def test_set_credits(self):
        """Test set_credits method."""
        result = self.session.set_credits("50")

        assert self.session.attributes[ATTRIBUTE.CREDITS] == "50"
        assert result is self.session

    def test_set_cost(self):
        """Test set_cost method."""
        result = self.session.set_cost("0.05")

        assert self.session.attributes[ATTRIBUTE.COST] == "0.05"
        assert result is self.session

    def test_set_model(self):
        """Test set_model method."""
        result = self.session.set_model("gpt-4")

        assert self.session.attributes[ATTRIBUTE.MODEL] == "gpt-4"
        assert result is self.session

    def test_set_llm_system(self):
        """Test set_llm_system method."""
        result = self.session.set_llm_system("openai")

        assert self.session.attributes[ATTRIBUTE.LLM_SYSTEM] == "openai"
        assert result is self.session


class TestSessionStatusMethods:
    """Test cases for Session status and error management methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.session.trace.get_tracer"):
            self.session = Session("test_session")

    def test_set_error_without_span(self):
        """Test set_error method when no span exists."""
        result = self.session.set_error("Test error message")

        assert self.session.status == "error"
        assert self.session.error_message == "Test error message"
        assert self.session.attributes[ATTRIBUTE.ERROR_MESSAGE] == "Test error message"
        assert result is self.session

    @patch("netra.session.trace.get_tracer")
    def test_set_success_with_span(self, mock_get_tracer):
        """Test set_success method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        session = Session("test")
        session.span = mock_span

        # Act
        result = session.set_success()

        # Assert
        assert result is session
        assert session.status == "success"
        # set_success only sets span status, not attributes
        mock_span.set_attribute.assert_not_called()

        # Verify span status - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK

    def test_set_success_without_span(self):
        """Test set_success method when no span exists."""
        result = self.session.set_success()

        assert self.session.status == "success"
        assert result is self.session

    def test_get_current_span(self):
        """Test get_current_span method."""
        mock_span = Mock()
        self.session.span = mock_span

        result = self.session.get_current_span()

        assert result is mock_span

    def test_get_current_span_none(self):
        """Test get_current_span method when span is None."""
        result = self.session.get_current_span()

        assert result is None


class TestSessionEventMethods:
    """Test cases for Session event handling methods."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        with patch("netra.session.trace.get_tracer"):
            self.session = Session("test_session")

    def test_add_event_without_span(self):
        """Test add_event method when no span exists."""
        result = self.session.add_event("test_event", {"key": "value"})

        # Should not raise an error and return self for chaining
        assert result is self.session

    def test_add_event_with_span_and_attributes(self):
        """Test add_event method when span exists with attributes."""
        mock_span = Mock()
        self.session.span = mock_span

        attributes = {"key1": "value1", "key2": "value2"}
        result = self.session.add_event("test_event", attributes)

        mock_span.add_event.assert_called_once_with("test_event", attributes)
        assert result is self.session

    def test_add_event_with_span_no_attributes(self):
        """Test add_event method when span exists without attributes."""
        mock_span = Mock()
        self.session.span = mock_span

        result = self.session.add_event("test_event")

        mock_span.add_event.assert_called_once_with("test_event", {})
        assert result is self.session


class TestSessionContextManager:
    """Test cases for Session context manager behavior (__enter__ and __exit__)."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_tracer = Mock()
        self.mock_span = Mock()
        self.mock_tracer.start_span.return_value = self.mock_span

        with patch("netra.session.trace.get_tracer", return_value=self.mock_tracer):
            self.session = Session("test_session", {"initial_key": "initial_value"})

    @patch("netra.session.time.time")
    @patch("netra.session.set_span_in_context")
    @patch("netra.session.context_api.attach")
    @patch("netra.session.logger")
    def test_enter_method(self, mock_logger, mock_attach, mock_set_span_in_context, mock_time):
        """Test __enter__ method functionality."""
        mock_time.return_value = 1234567890.123
        mock_context = Mock()
        mock_set_span_in_context.return_value = mock_context
        mock_token = Mock()
        mock_attach.return_value = mock_token

        result = self.session.__enter__()

        # Verify time tracking
        assert self.session.start_time == 1234567890.123

        # Verify span creation
        self.mock_tracer.start_span.assert_called_once_with(
            name="test_session", kind=SpanKind.CLIENT, attributes={"initial_key": "initial_value"}
        )
        assert self.session.span is self.mock_span

        # Verify context management
        mock_set_span_in_context.assert_called_once_with(self.mock_span)
        mock_attach.assert_called_once_with(mock_context)
        assert self.session.context_token is mock_token

        # Verify logging
        mock_logger.info.assert_called_once_with("Started session: test_session")

        # Verify return value
        assert result is self.session

    @patch("netra.session.time.time")
    @patch("netra.session.context_api.detach")
    @patch("netra.session.logger")
    def test_exit_method_success_no_exception(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method with successful completion (no exception)."""
        # Setup session state
        self.session.start_time = 1234567890.123
        mock_time.return_value = 1234567890.623  # 500ms later
        self.session.span = self.mock_span
        mock_token = Mock()
        self.session.context_token = mock_token

        result = self.session.__exit__(None, None, None)

        # Verify time tracking
        assert self.session.end_time == 1234567890.623
        duration_ms = (self.session.end_time - self.session.start_time) * 1000
        assert self.session.attributes[ATTRIBUTE.DURATION_MS] == str(round(duration_ms, 2))

        # Verify status
        assert self.session.status == "success"
        assert self.session.attributes[ATTRIBUTE.STATUS] == "success"
        # Verify span status - check call arguments instead of Status object equality
        self.mock_span.set_status.assert_called_once()
        call_args = self.mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK

        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.DURATION_MS, str(round(duration_ms, 2)))
        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.STATUS, "success")

        # Verify span and context cleanup
        self.mock_span.end.assert_called_once()
        mock_detach.assert_called_once_with(mock_token)

        # Verify logging
        mock_logger.info.assert_called_once_with("Ended session: test_session (Status: success, Duration: 500.00ms)")

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.session.time.time")
    @patch("netra.session.context_api.detach")
    @patch("netra.session.logger")
    def test_exit_method_with_exception(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when an exception occurs."""
        # Setup session state
        self.session.start_time = 1234567890.123
        mock_time.return_value = 1234567890.323  # 200ms later
        self.session.span = self.mock_span
        mock_token = Mock()
        self.session.context_token = mock_token

        # Create test exception
        test_exception = ValueError("Test error")

        result = self.session.__exit__(ValueError, test_exception, None)

        # Verify time tracking
        assert self.session.end_time == 1234567890.323
        duration_ms = (self.session.end_time - self.session.start_time) * 1000
        assert self.session.attributes[ATTRIBUTE.DURATION_MS] == str(round(duration_ms, 2))

        # Verify error handling
        assert self.session.status == "error"
        assert self.session.error_message == "Test error"
        assert self.session.attributes[ATTRIBUTE.ERROR_MESSAGE] == "Test error"
        assert self.session.attributes[ATTRIBUTE.STATUS] == "error"

        # Verify span error handling - check call arguments instead of Status object equality
        self.mock_span.set_status.assert_called_once()
        call_args = self.mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Test error"

        self.mock_span.record_exception.assert_called_once_with(test_exception)
        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.DURATION_MS, str(round(duration_ms, 2)))
        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.STATUS, "error")
        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.ERROR_MESSAGE, "Test error")

        # Verify logging
        mock_logger.error.assert_called_once_with("Session test_session failed: Test error")
        mock_logger.info.assert_called_once_with("Ended session: test_session (Status: error, Duration: 200.00ms)")

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.session.time.time")
    @patch("netra.session.context_api.detach")
    @patch("netra.session.logger")
    def test_exit_method_with_manual_status_change(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when status was manually changed before exit."""
        # Setup session state with manual status change
        self.session.start_time = 1234567890.123
        mock_time.return_value = 1234567890.223  # 100ms later
        self.session.span = self.mock_span
        self.session.status = "custom_status"  # Manually set status
        self.session.error_message = "custom_error"
        mock_token = Mock()
        self.session.context_token = mock_token

        result = self.session.__exit__(None, None, None)

        # Verify that status is not overridden to "success"
        assert self.session.status == "custom_status"
        assert self.session.error_message == "custom_error"

        # Verify span operations - only duration_ms and status are set, not error_message since no exception occurred
        self.mock_span.set_attribute.assert_any_call(
            ATTRIBUTE.DURATION_MS, str(round((1234567890.223 - 1234567890.123) * 1000, 2))
        )
        self.mock_span.set_attribute.assert_any_call(ATTRIBUTE.STATUS, "custom_status")
        # error_message is NOT set by __exit__ when there's no exception, even if manually set
        # The span status is set based on the error_message if it exists
        self.mock_span.set_status.assert_not_called()  # Fix: span.set_status should not be called

        # Verify span and context cleanup
        self.mock_span.end.assert_called_once()
        mock_detach.assert_called_once_with(mock_token)

        # Verify logging
        mock_logger.info.assert_called_once_with(
            "Ended session: test_session (Status: custom_status, Duration: 100.00ms)"
        )

        # Verify return value (should not suppress exceptions)
        assert result is False

    @patch("netra.session.time.time")
    @patch("netra.session.context_api.detach")
    @patch("netra.session.logger")
    def test_exit_method_no_start_time(self, mock_logger, mock_detach, mock_time):
        """Test __exit__ method when start_time is None."""
        # Setup session state without start_time
        self.session.start_time = None
        self.session.span = self.mock_span

        result = self.session.__exit__(None, None, None)

        # Verify duration is not calculated
        assert ATTRIBUTE.DURATION_MS not in self.session.attributes

        # Verify other functionality still works
        assert self.session.status == "success"

        # Verify logging without duration
        mock_logger.info.assert_called_once_with("Ended session: test_session (Status: success)")

        # Verify return value (should not suppress exceptions)
        assert result is False


class TestSessionIntegration:
    """Integration tests for Session as a context manager."""

    @patch("netra.session.trace.get_tracer")
    def test_set_error_with_span(self, mock_get_tracer):
        """Test set_error method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        session = Session("test")
        session.span = mock_span

        # Act
        result = session.set_error("Test error")

        # Assert
        assert result is session
        assert session.status == "error"
        assert session.error_message == "Test error"
        assert session.attributes[ATTRIBUTE.ERROR_MESSAGE] == "Test error"

        # Verify span operations - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Test error"
        mock_span.set_attribute.assert_called_once_with(ATTRIBUTE.ERROR_MESSAGE, "Test error")

    @patch("netra.session.trace.get_tracer")
    def test_set_success_with_span(self, mock_get_tracer):
        """Test set_success method when span exists."""
        # Setup
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        session = Session("test")
        session.span = mock_span

        # Act
        result = session.set_success()

        # Assert
        assert result is session
        assert session.status == "success"
        # set_success only sets span status, not attributes
        mock_span.set_attribute.assert_not_called()

        # Verify span status - check call arguments instead of Status object equality
        mock_span.set_status.assert_called_once()
        call_args = mock_span.set_status.call_args[0][0]  # Get the Status object
        assert call_args.status_code == StatusCode.OK
