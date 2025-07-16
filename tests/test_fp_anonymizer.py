"""
Unit tests for FormatPreservingEmailAnonymizer class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

from netra.anonymizer.fp_anonymizer import FormatPreservingEmailAnonymizer


class TestFormatPreservingEmailAnonymizer:
    """Test FormatPreservingEmailAnonymizer core functionality."""

    def test_initialization_with_defaults(self):
        """Test anonymizer initialization with default settings."""
        # Act
        anonymizer = FormatPreservingEmailAnonymizer()

        # Assert
        assert anonymizer.preserve_length is True
        assert anonymizer.preserve_structure is True
        assert len(anonymizer.email_cache) == 0
        assert len(anonymizer.part_cache) == 0

    def test_initialization_with_custom_settings(self):
        """Test anonymizer initialization with custom settings."""
        # Act
        anonymizer = FormatPreservingEmailAnonymizer(preserve_length=False, preserve_structure=False)

        # Assert
        assert anonymizer.preserve_length is False
        assert anonymizer.preserve_structure is False

    def test_single_email_anonymization_with_structure_preservation(self):
        """Test anonymizing a single email with structure preservation."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer(preserve_structure=True)
        email = "john.doe@example.com"

        # Act
        result = anonymizer._anonymize_email(email)

        # Assert
        assert "@" in result
        assert len(result) == len(email)  # Length preserved
        assert "." in result  # Structure preserved
        assert result != email  # Actually anonymized
        assert result in anonymizer.email_cache.values()

    def test_single_email_anonymization_without_structure_preservation(self):
        """Test anonymizing a single email without structure preservation."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer(preserve_structure=False, preserve_length=True)
        email = "john.doe@example.com"

        # Act
        result = anonymizer._anonymize_email(email)

        # Assert
        assert "@" in result
        assert len(result) == len(email)  # Length preserved
        assert result != email  # Actually anonymized

    def test_email_caching_consistency(self):
        """Test that same email produces same anonymized result (caching)."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer()
        email = "test@example.com"

        # Act
        result1 = anonymizer._anonymize_email(email)
        result2 = anonymizer._anonymize_email(email)

        # Assert
        assert result1 == result2
        assert len(anonymizer.email_cache) == 1

    def test_text_anonymization_single_email(self):
        """Test anonymizing text containing a single email."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer()
        text = "Contact me at john@example.com for more info"

        # Act
        result = anonymizer.anonymize_text(text)

        # Assert
        assert "john@example.com" not in result
        assert "@" in result
        assert "Contact me at" in result
        assert "for more info" in result

    def test_text_anonymization_multiple_emails(self):
        """Test anonymizing text containing multiple emails."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer()
        text = "Email john@example.com or jane@test.org"

        # Act
        result = anonymizer.anonymize_text(text)

        # Assert
        assert "john@example.com" not in result
        assert "jane@test.org" not in result
        assert result.count("@") == 2  # Both emails should have @ symbol
        assert "Email" in result and "or" in result  # Non-email text preserved

    def test_get_mapping_returns_cache(self):
        """Test that get_mapping returns the email mapping cache."""
        # Arrange
        anonymizer = FormatPreservingEmailAnonymizer()
        email = "test@example.com"

        # Act
        anonymizer._anonymize_email(email)
        mapping = anonymizer.get_mapping()

        # Assert
        assert len(mapping) == 1
        assert email in mapping
        assert mapping[email] != email  # Anonymized value is different
        assert isinstance(mapping, dict)

    def test_deterministic_anonymization(self):
        """Test that anonymization is deterministic across different instances."""
        # Arrange
        anonymizer1 = FormatPreservingEmailAnonymizer()
        anonymizer2 = FormatPreservingEmailAnonymizer()
        email = "test@example.com"

        # Act
        result1 = anonymizer1._anonymize_email(email)
        result2 = anonymizer2._anonymize_email(email)

        # Assert
        assert result1 == result2  # Should be deterministic
