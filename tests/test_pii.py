"""
Unit tests for the Netra SDK's PII detection module (netra/pii.py).

This module tests the core PII detection functionality including:
- PIIDetectionResult dataclass
- RegexPIIDetector basic functionality
- PresidioPIIDetector basic functionality
- Default detector creation
- Basic input processing
"""

import re
from typing import Dict, Pattern
from unittest.mock import MagicMock, patch

import pytest

from netra.exceptions import PIIBlockedException
from netra.pii import (
    DEFAULT_PII_PATTERNS,
    PIIDetectionResult,
    PresidioPIIDetector,
    RegexPIIDetector,
    get_default_detector,
)


class TestPIIDetectionResult:
    """Test the PIIDetectionResult dataclass functionality."""

    def test_pii_detection_result_creation_with_defaults(self):
        """Test PIIDetectionResult creation with default values."""
        result = PIIDetectionResult()

        assert result.has_pii is False
        assert result.pii_entities == {}
        assert result.masked_text is None
        assert result.original_text is None
        assert result.is_blocked is False
        assert result.is_masked is False
        assert result.pii_actions == {}
        assert result.hashed_entities == {}

    def test_pii_detection_result_creation_with_values(self):
        """Test PIIDetectionResult creation with custom values."""
        result = PIIDetectionResult(
            has_pii=True,
            pii_entities={"EMAIL": 2, "PHONE": 1},
            masked_text="Contact me at <EMAIL_HASH> or <PHONE_HASH>",
            original_text="Contact me at john@example.com or 555-1234",
            is_blocked=True,
            is_masked=True,
            pii_actions={"MASK": ["EMAIL", "PHONE"]},
            hashed_entities={"EMAIL_HASH": "john@example.com", "PHONE_HASH": "555-1234"},
        )

        assert result.has_pii is True
        assert result.pii_entities == {"EMAIL": 2, "PHONE": 1}
        assert result.masked_text == "Contact me at <EMAIL_HASH> or <PHONE_HASH>"
        assert result.original_text == "Contact me at john@example.com or 555-1234"
        assert result.is_blocked is True
        assert result.is_masked is True
        assert result.pii_actions == {"MASK": ["EMAIL", "PHONE"]}
        assert result.hashed_entities == {"EMAIL_HASH": "john@example.com", "PHONE_HASH": "555-1234"}


class TestRegexPIIDetector:
    """Test the RegexPIIDetector functionality."""

    def test_regex_detector_initialization_with_defaults(self):
        """Test RegexPIIDetector initialization with default parameters."""
        detector = RegexPIIDetector()

        assert detector._action_type == "MASK"
        assert detector.patterns == DEFAULT_PII_PATTERNS
        assert "EMAIL" in detector.patterns
        assert "PHONE" in detector.patterns
        assert "CREDIT_CARD" in detector.patterns
        assert "SSN" in detector.patterns

    def test_regex_detector_initialization_with_custom_patterns(self):
        """Test RegexPIIDetector initialization with custom patterns."""
        custom_patterns: Dict[str, Pattern[str]] = {
            "CUSTOM_EMAIL": re.compile(r"[a-z]+@[a-z]+\.com"),
            "CUSTOM_PHONE": re.compile(r"\d{10}"),
        }

        detector = RegexPIIDetector(patterns=custom_patterns, action_type="FLAG")

        assert detector._action_type == "FLAG"
        assert detector.patterns == custom_patterns
        assert len(detector.patterns) == 2
        assert "CUSTOM_EMAIL" in detector.patterns
        assert "CUSTOM_PHONE" in detector.patterns


class TestPresidioPIIDetector:
    """Test the PresidioPIIDetector functionality."""

    @patch("presidio_analyzer.AnalyzerEngine")
    @patch("netra.pii.Anonymizer")
    def test_presidio_detector_initialization_with_defaults(self, mock_anonymizer, mock_analyzer):
        """Test PresidioPIIDetector initialization with default parameters."""
        # Mock the analyzer engine
        mock_analyzer_instance = MagicMock()
        mock_analyzer.return_value = mock_analyzer_instance

        # Mock the anonymizer
        mock_anonymizer_instance = MagicMock()
        mock_anonymizer.return_value = mock_anonymizer_instance

        detector = PresidioPIIDetector()

        assert detector._action_type == "FLAG"
        assert detector.language == "en"
        assert detector.score_threshold == 0.6
        assert detector.entities is not None
        assert len(detector.entities) > 0
        assert "EMAIL_ADDRESS" in detector.entities
        assert "PHONE_NUMBER" in detector.entities

        # Verify analyzer was created
        mock_analyzer.assert_called_once()
        assert detector.analyzer == mock_analyzer_instance

        # Verify anonymizer was created with default parameters
        mock_anonymizer.assert_called_once_with(hash_function=None, cache_size=1000)
        assert detector.anonymizer == mock_anonymizer_instance

    @patch("builtins.__import__", side_effect=ImportError("No module named 'presidio_analyzer'"))
    def test_presidio_detector_import_error(self, mock_import):
        """Test PresidioPIIDetector raises ImportError when presidio is not available."""
        with pytest.raises(ImportError, match="Presidio-based PII detection requires: presidio-analyzer"):
            PresidioPIIDetector()


class TestGetDefaultDetector:
    """Test the get_default_detector function."""

    @patch("netra.pii.PresidioPIIDetector")
    def test_get_default_detector_returns_presidio_by_default(self, mock_presidio):
        """Test that get_default_detector returns PresidioPIIDetector by default."""
        mock_detector = MagicMock()
        mock_presidio.return_value = mock_detector

        detector = get_default_detector()

        assert detector == mock_detector
        mock_presidio.assert_called_once_with(
            action_type=None, entities=None, hash_function=None, nlp_configuration=None
        )

    @patch("netra.pii.PresidioPIIDetector")
    def test_get_default_detector_with_custom_parameters(self, mock_presidio):
        """Test get_default_detector with custom parameters."""
        mock_detector = MagicMock()
        mock_presidio.return_value = mock_detector

        custom_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER"]
        custom_hash_func = lambda x: "hash"

        detector = get_default_detector(action_type="MASK", entities=custom_entities, hash_function=custom_hash_func)

        assert detector == mock_detector
        mock_presidio.assert_called_once_with(
            action_type="MASK", entities=custom_entities, hash_function=custom_hash_func, nlp_configuration=None
        )


class TestPIIDetectorInputProcessing:
    """Test PIIDetector input processing and routing methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = RegexPIIDetector(action_type="FLAG")

    def test_process_input_data_with_string(self):
        """Test _process_input_data with string input."""
        test_text = "Contact me at john@example.com"

        with patch.object(self.detector, "_detect_single_message") as mock_detect:
            mock_result = PIIDetectionResult(has_pii=True, original_text=test_text)
            mock_detect.return_value = mock_result

            result = self.detector._process_input_data(test_text)

            assert result == mock_result
            mock_detect.assert_called_once_with(test_text)

    def test_process_input_data_with_list(self):
        """Test _process_input_data with list input."""
        test_list = ["john@example.com", "Call me at 555-1234"]

        with patch.object(self.detector, "_process_list_input") as mock_process:
            mock_result = PIIDetectionResult(has_pii=True)
            mock_process.return_value = mock_result

            result = self.detector._process_input_data(test_list)

            assert result == mock_result
            mock_process.assert_called_once_with(test_list)

    def test_process_input_data_with_unsupported_type(self):
        """Test _process_input_data with unsupported input type."""
        unsupported_input = 12345

        with pytest.raises(ValueError, match="Unsupported input type"):
            self.detector._process_input_data(unsupported_input)

    def test_process_list_input_with_strings(self):
        """Test _process_list_input with list of strings."""
        test_strings = ["john@example.com", "Call me at 555-1234"]

        with patch.object(self.detector, "_detect_string_list") as mock_detect:
            mock_result = PIIDetectionResult(has_pii=True)
            mock_detect.return_value = mock_result

            result = self.detector._process_list_input(test_strings)

            assert result == mock_result
            mock_detect.assert_called_once_with(test_strings)

    def test_process_list_input_with_dicts(self):
        """Test _process_list_input with list of dictionaries."""
        test_dicts = [
            {"role": "user", "content": "My email is john@example.com"},
            {"role": "assistant", "content": "I'll help you with that"},
        ]

        with patch.object(self.detector, "_detect_chat_messages") as mock_detect:
            mock_result = PIIDetectionResult(has_pii=True)
            mock_detect.return_value = mock_result

            result = self.detector._process_list_input(test_dicts)

            assert result == mock_result
            mock_detect.assert_called_once_with(test_dicts)

    def test_process_list_input_with_langchain_messages(self):
        """Test _process_list_input with LangChain-like message objects."""
        # Create mock LangChain-like objects
        mock_message1 = MagicMock()
        mock_message1.content = "My email is john@example.com"
        mock_message1.type = "human"

        mock_message2 = MagicMock()
        mock_message2.content = "I'll help you"
        mock_message2.type = "ai"

        test_messages = [mock_message1, mock_message2]

        with patch.object(self.detector, "_process_langchain_messages") as mock_process:
            mock_result = PIIDetectionResult(has_pii=True)
            mock_process.return_value = mock_result

            result = self.detector._process_list_input(test_messages)

            assert result == mock_result
            mock_process.assert_called_once_with(test_messages)

    def test_process_list_input_with_unsupported_items(self):
        """Test _process_list_input with unsupported item types."""
        unsupported_list = [123, 456, 789]

        with pytest.raises(ValueError, match="Unsupported input type in list"):
            self.detector._process_list_input(unsupported_list)

    def test_is_langchain_message_with_valid_message(self):
        """Test _is_langchain_message with valid LangChain-like object."""
        mock_message = MagicMock()
        mock_message.content = "Test content"
        mock_message.type = "human"

        result = self.detector._is_langchain_message(mock_message)

        assert result is True

    def test_is_langchain_message_with_invalid_message(self):
        """Test _is_langchain_message with invalid object."""
        invalid_objects = [
            "string",
            {"key": "value"},
            123,
            MagicMock(spec=[]),  # Mock without required attributes
        ]

        for obj in invalid_objects:
            result = self.detector._is_langchain_message(obj)
            assert result is False


class TestPIIDetectorExceptionHandling:
    """Test PIIDetector exception handling and result creation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = RegexPIIDetector(action_type="FLAG")

    def test_handle_pii_exception_with_block_action(self):
        """Test _handle_pii_exception with BLOCK action type."""
        self.detector._action_type = "BLOCK"

        exception = PIIBlockedException(
            message="PII detected",
            has_pii=True,
            pii_entities={"EMAIL": 1},
            masked_text="Contact <EMAIL_HASH>",
            hashed_entities={"EMAIL_HASH": "john@example.com"},
        )

        with pytest.raises(PIIBlockedException):
            self.detector._handle_pii_exception(exception)

    def test_handle_pii_exception_with_flag_action(self):
        """Test _handle_pii_exception with FLAG action type."""
        self.detector._action_type = "FLAG"

        exception = PIIBlockedException(
            message="PII detected",
            has_pii=True,
            pii_entities={"EMAIL": 1},
            masked_text="Contact <EMAIL_HASH>",
            original_text="Contact john@example.com",
            hashed_entities={"EMAIL_HASH": "john@example.com"},
        )

        with patch.object(self.detector, "_create_detection_result") as mock_create:
            mock_result = PIIDetectionResult(has_pii=True, is_blocked=False)
            mock_create.return_value = mock_result

            result = self.detector._handle_pii_exception(exception)

            assert result == mock_result
            mock_create.assert_called_once()

    def test_create_pii_actions_with_different_action_types(self):
        """Test _create_pii_actions with different action types."""
        exception = PIIBlockedException(pii_entities={"EMAIL": 1, "PHONE": 2})

        # Test FLAG action
        self.detector._action_type = "FLAG"
        result = self.detector._create_pii_actions(exception)
        expected = {"FLAG": ["EMAIL", "PHONE"]}
        assert result == expected

        # Test MASK action
        self.detector._action_type = "MASK"
        result = self.detector._create_pii_actions(exception)
        expected = {"MASK": ["EMAIL", "PHONE"]}
        assert result == expected

        # Test BLOCK action
        self.detector._action_type = "BLOCK"
        result = self.detector._create_pii_actions(exception)
        expected = {"BLOCK": ["EMAIL", "PHONE"]}
        assert result == expected

    def test_serialize_masked_text_with_different_types(self):
        """Test _serialize_masked_text with different input types."""
        # Test string
        result = self.detector._serialize_masked_text("Simple text")
        assert result == "Simple text"

        # Test list of dicts (chat messages)
        chat_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        result = self.detector._serialize_masked_text(chat_messages)
        expected = '[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]'
        assert result == expected

        # Test other types
        result = self.detector._serialize_masked_text(["item1", "item2"])
        assert result == '["item1", "item2"]'

    @patch("netra.pii.Netra.set_custom_event")
    def test_build_trace_attributes(self, mock_set_event):
        """Test _build_trace_attributes creates proper attributes dictionary."""
        exception = PIIBlockedException(
            has_pii=True,
            pii_entities={"EMAIL": 1, "PHONE": 1},
            masked_text="Contact <EMAIL_HASH> or <PHONE_HASH>",
            original_text="Contact john@example.com or 555-1234",
        )

        pii_actions = {"FLAG": ["EMAIL", "PHONE"]}

        with patch.object(self.detector, "_serialize_masked_text", return_value="serialized_text"):
            result = self.detector._build_trace_attributes(exception, pii_actions)

            expected_attributes = {
                "has_pii": True,
                "pii_entities": '{"EMAIL": 1, "PHONE": 1}',
                "is_blocked": False,  # FLAG action type
                "is_masked": False,  # FLAG action type
                "pii_actions": '{"FLAG": ["EMAIL", "PHONE"]}',
            }

            assert result == expected_attributes
