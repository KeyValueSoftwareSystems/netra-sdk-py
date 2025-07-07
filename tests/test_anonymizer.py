"""
Unit tests for Anonymizer class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from presidio_analyzer.recognizer_result import RecognizerResult

from netra.anonymizer.anonymizer import Anonymizer
from netra.anonymizer.base import AnonymizationResult


class TestAnonymizer:
    """Test Anonymizer core functionality."""

    def test_initialization_with_defaults(self):
        """Test anonymizer initialization with default settings."""
        # Act
        anonymizer = Anonymizer()

        # Assert
        assert anonymizer.base_anonymizer is not None
        assert anonymizer.email_anonymizer is not None
        assert anonymizer.base_anonymizer.cache_size == 1000

    def test_initialization_with_custom_settings(self):
        """Test anonymizer initialization with custom settings."""

        # Arrange
        def custom_hash(value: str) -> str:
            return f"custom_{hash(value)}"

        # Act
        anonymizer = Anonymizer(hash_function=custom_hash, cache_size=500)

        # Assert
        assert anonymizer.base_anonymizer.hash_function == custom_hash
        assert anonymizer.base_anonymizer.cache_size == 500

    def test_anonymize_text_with_email_entity(self):
        """Test anonymizing text containing email entity."""
        # Arrange
        anonymizer = Anonymizer()
        text = "Contact me at john@example.com"
        analyzer_results = [RecognizerResult(entity_type="EMAIL", start=14, end=30, score=0.9)]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "john@example.com" not in result.masked_text
        assert "@" in result.masked_text  # Email format preserved
        assert "Contact me at" in result.masked_text
        assert len(result.entities) == 1
        assert "john@example.com" in result.entities.values()

    def test_anonymize_text_with_non_email_entity(self):
        """Test anonymizing text containing non-email entity."""
        # Arrange
        anonymizer = Anonymizer()
        text = "My phone is 555-1234"
        analyzer_results = [RecognizerResult(entity_type="PHONE", start=12, end=20, score=0.8)]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "555-1234" not in result.masked_text
        assert "<PHONE_" in result.masked_text  # Hash format for non-email
        assert "My phone is" in result.masked_text
        assert len(result.entities) == 1
        assert "555-1234" in result.entities.values()

    def test_anonymize_text_with_mixed_entities(self):
        """Test anonymizing text with both email and non-email entities."""
        # Arrange
        anonymizer = Anonymizer()
        text = "Email john@example.com or call 555-1234"
        analyzer_results = [
            RecognizerResult(entity_type="EMAIL", start=6, end=22, score=0.9),
            RecognizerResult(entity_type="PHONE", start=31, end=39, score=0.8),
        ]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "john@example.com" not in result.masked_text
        assert "555-1234" not in result.masked_text
        assert "@" in result.masked_text  # Email format preserved
        assert "<PHONE_" in result.masked_text  # Hash format for phone
        assert len(result.entities) == 2
        assert "john@example.com" in result.entities.values()
        assert "555-1234" in result.entities.values()

    def test_anonymize_text_with_email_address_entity_type(self):
        """Test anonymizing text with EMAIL_ADDRESS entity type (alternative email type)."""
        # Arrange
        anonymizer = Anonymizer()
        text = "Send to admin@company.org"
        analyzer_results = [RecognizerResult(entity_type="EMAIL_ADDRESS", start=8, end=25, score=0.9)]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "admin@company.org" not in result.masked_text
        assert "@" in result.masked_text  # Email format preserved
        assert "Send to" in result.masked_text
        assert len(result.entities) == 1

    def test_anonymize_empty_text(self):
        """Test anonymizing empty text with no entities."""
        # Arrange
        anonymizer = Anonymizer()
        text = ""
        analyzer_results = []

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert result.masked_text == ""
        assert len(result.entities) == 0

    def test_anonymize_text_with_no_entities(self):
        """Test anonymizing text with no detected entities."""
        # Arrange
        anonymizer = Anonymizer()
        text = "This is just plain text with no PII"
        analyzer_results = []

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert result.masked_text == text  # Text unchanged
        assert len(result.entities) == 0
