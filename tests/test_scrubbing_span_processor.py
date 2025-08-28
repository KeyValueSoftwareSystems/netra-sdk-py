"""
Unit tests for ScrubbingSpanProcessor class.
Tests scrubbing functionality for sensitive data patterns and edge cases.
"""

from unittest.mock import Mock, patch

from netra.processors.scrubbing_span_processor import ScrubbingSpanProcessor


class TestScrubbingSpanProcessor:
    """Test ScrubbingSpanProcessor core functionality."""

    def test_initialization(self):
        """Test ScrubbingSpanProcessor initialization."""
        # Act
        processor = ScrubbingSpanProcessor()

        # Assert
        assert processor is not None
        assert hasattr(processor, "on_start")
        assert hasattr(processor, "on_end")
        assert hasattr(processor, "force_flush")
        assert hasattr(processor, "shutdown")
        assert processor.scrub_replacement == "[SCRUBBED]"

    def test_on_start_method(self):
        """Test on_start method (no-op implementation)."""
        # Arrange
        processor = ScrubbingSpanProcessor()
        mock_span = Mock()
        mock_parent_context = Mock()

        # Act & Assert (should not raise any exception)
        processor.on_start(mock_span, mock_parent_context)

    def test_on_end_scrubs_sensitive_attributes(self):
        """Test on_end method scrubs sensitive data from span attributes."""
        # Arrange
        processor = ScrubbingSpanProcessor()
        mock_span = Mock()

        # Mock span attributes with sensitive data
        mock_span._attributes = {
            "user_email": "john.doe@example.com",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "password": "supersecret123",
            "normal_data": "This is normal data",
            "phone": "+1-555-123-4567",
            "credit_card": "4111111111111111",
        }

        # Act
        processor.on_end(mock_span)

        # Assert
        expected_scrubbed = {
            "user_email": "[SCRUBBED]",
            "api_key": "[SCRUBBED]",
            "password": "[SCRUBBED]",
            "normal_data": "This is normal data",
            "phone": "[SCRUBBED]",
            "credit_card": "[SCRUBBED]",
        }
        assert mock_span._attributes == expected_scrubbed

    def test_on_end_handles_missing_attributes(self):
        """Test on_end method handles spans without _attributes gracefully."""
        # Arrange
        processor = ScrubbingSpanProcessor()
        mock_span = Mock()
        mock_span._attributes = None

        # Act & Assert (should not raise any exception)
        processor.on_end(mock_span)

    @patch("netra.processors.scrubbing_span_processor.logger")
    def test_on_end_handles_exceptions(self, mock_logger):
        """Test on_end method handles exceptions gracefully."""
        # Arrange
        processor = ScrubbingSpanProcessor()
        mock_span = Mock()

        # Configure mock to raise exception when accessing _attributes
        mock_span._attributes = {"test": "value"}
        # Make the items() method raise an exception
        mock_span._attributes = Mock()
        mock_span._attributes.items.side_effect = Exception("Test exception")

        # Act
        processor.on_end(mock_span)

        # Assert
        mock_logger.exception.assert_called_once()
        assert "Error scrubbing span attributes:" in str(mock_logger.exception.call_args)

    def test_is_sensitive_key(self):
        """Test _is_sensitive_key method identifies sensitive keys correctly."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test sensitive keys
        sensitive_keys = [
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "auth",
            "authorization",
            "bearer",
            "credential",
            "private_key",
            "access_token",
            "refresh_token",
            "session_token",
            "x-api-key",
            "x-auth-token",
            "cookie",
            "set-cookie",
        ]

        for key in sensitive_keys:
            assert processor._is_sensitive_key(key) is True
            assert processor._is_sensitive_key(key.upper()) is True
            assert processor._is_sensitive_key(f"user_{key}") is True

        # Test non-sensitive keys
        non_sensitive_keys = ["username", "email", "name", "id", "data", "response"]
        for key in non_sensitive_keys:
            assert processor._is_sensitive_key(key) is False

    def test_scrub_string_value_email_pattern(self):
        """Test _scrub_string_value method scrubs email patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("Contact us at john.doe@example.com", "Contact us at [SCRUBBED]"),
            ("Email: user@domain.co.uk", "Email: [SCRUBBED]"),
            ("Multiple emails: a@b.com and c@d.org", "Multiple emails: [SCRUBBED] and [SCRUBBED]"),
            ("No email here", "No email here"),
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_string_value_phone_pattern(self):
        """Test _scrub_string_value method scrubs phone patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("Call me at +1-555-123-4567", "Call me at [SCRUBBED]"),
            ("Phone: (555) 123-4567", "Phone: [SCRUBBED]"),
            ("Contact: 555.123.4567", "Contact: [SCRUBBED]"),
            ("No phone number", "No phone number"),
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_string_value_credit_card_pattern(self):
        """Test _scrub_string_value method scrubs credit card patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("Card: 4111111111111111", "Card: [SCRUBBED]"),
            ("Visa: 4000000000000002", "Visa: [SCRUBBED]"),
            ("MasterCard: 5555555555554444", "MasterCard: [SCRUBBED]"),
            ("Not a card: 123", "Not a card: 123"),
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_string_value_api_key_pattern(self):
        """Test _scrub_string_value method scrubs API key patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("API Key: sk-1234567890abcdef1234567890abcdef", "API Key: [SCRUBBED]"),
            ("Token: abcdef1234567890abcdef1234567890abcdef12", "[SCRUBBED]"),
            ("Short key: abc123", "Short key: abc123"),  # Too short to be considered API key
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_string_value_password_pattern(self):
        """Test _scrub_string_value method scrubs password patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("password: mysecretpass", "[SCRUBBED]"),
            ("Password=supersecret123", "[SCRUBBED]"),
            ("token: abc123def", "[SCRUBBED]"),
            ("username: john", "username: john"),  # Not a password pattern
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_string_value_bearer_token_pattern(self):
        """Test _scrub_string_value method scrubs Bearer token patterns."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            ("Authorization: Bearer abc123def456", "[SCRUBBED]"),
            ("Bearer token123", "[SCRUBBED]"),
        ]

        for input_str, expected in test_cases:
            result = processor._scrub_string_value(input_str)
            assert result == expected

    def test_scrub_dict_value(self):
        """Test _scrub_dict_value method scrubs nested dictionaries."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test case
        input_dict = {
            "user_email": "john@example.com",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "normal_data": "safe data",
            "nested": {"password": "secret123", "username": "john_doe"},
        }

        expected = {
            "user_email": "[SCRUBBED]",
            "api_key": "[SCRUBBED]",
            "normal_data": "safe data",
            "nested": {"password": "[SCRUBBED]", "username": "john_doe"},
        }

        # Act
        result = processor._scrub_dict_value(input_dict)

        # Assert
        assert result == expected

    def test_scrub_list_value(self):
        """Test _scrub_list_value method scrubs lists and tuples."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test list
        input_list = ["john@example.com", "normal data", {"api_key": "secret123"}, ["nested@email.com", "safe"]]

        expected_list = ["[SCRUBBED]", "normal data", {"api_key": "[SCRUBBED]"}, ["[SCRUBBED]", "safe"]]

        result = processor._scrub_list_value(input_list)
        assert result == expected_list

        # Test tuple
        input_tuple = ("user@domain.com", "safe data")
        expected_tuple = ("[SCRUBBED]", "safe data")

        result = processor._scrub_list_value(input_tuple)
        assert result == expected_tuple
        assert isinstance(result, tuple)

    def test_scrub_key_value_comprehensive(self):
        """Test _scrub_key_value method with various data types."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Test cases
        test_cases = [
            # Sensitive key
            ("password", "any_value", ("password", "[SCRUBBED]")),
            # String value with email
            ("data", "Contact: user@example.com", ("data", "Contact: [SCRUBBED]")),
            # Dictionary value
            ("config", {"api_key": "secret"}, ("config", {"api_key": "[SCRUBBED]"})),
            # List value
            ("emails", ["user@domain.com", "safe"], ("emails", ["[SCRUBBED]", "safe"])),
            # Normal data
            ("username", "john_doe", ("username", "john_doe")),
            # Non-string value
            ("count", 42, ("count", 42)),
        ]

        for key, value, expected in test_cases:
            result = processor._scrub_key_value(key, value)
            assert result == expected

    def test_force_flush_method(self):
        """Test force_flush method (no-op implementation)."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Act & Assert (should not raise any exception)
        processor.force_flush()
        processor.force_flush(timeout_millis=5000)

    def test_shutdown_method(self):
        """Test shutdown method (no-op implementation)."""
        # Arrange
        processor = ScrubbingSpanProcessor()

        # Act & Assert (should not raise any exception)
        processor.shutdown()

    def test_complex_nested_data_scrubbing(self):
        """Test scrubbing of complex nested data structures."""
        # Arrange
        processor = ScrubbingSpanProcessor()
        mock_span = Mock()

        # Complex nested structure with sensitive data
        mock_span._attributes = {
            "request_data": {
                "user": {
                    "email": "user@example.com",
                    "profile": {"phone": "555-123-4567", "preferences": ["email_notifications", "sms_alerts"]},
                },
                "auth": {"api_key": "sk-abcdef1234567890", "tokens": ["token1", "token2"]},
            },
            "response_list": [
                {"email": "admin@company.com", "role": "admin"},
                {"email": "user@company.com", "role": "user"},
            ],
            "normal_field": "This should not be scrubbed",
        }

        # Act
        processor.on_end(mock_span)

        # Assert
        result = mock_span._attributes

        # Check nested email scrubbing
        assert result["request_data"]["user"]["email"] == "[SCRUBBED]"
        assert result["request_data"]["user"]["profile"]["phone"] == "[SCRUBBED]"

        # Check sensitive key scrubbing
        assert result["request_data"]["auth"]["api_key"] == "[SCRUBBED]"

        # Check list scrubbing
        assert result["response_list"][0]["email"] == "[SCRUBBED]"
        assert result["response_list"][1]["email"] == "[SCRUBBED]"

        # Check normal data is preserved
        assert result["normal_field"] == "This should not be scrubbed"
        assert result["request_data"]["user"]["profile"]["preferences"] == ["email_notifications", "sms_alerts"]
        assert result["response_list"][0]["role"] == "admin"
