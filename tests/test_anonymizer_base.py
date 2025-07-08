"""
Unit tests for BaseAnonymizer class.
Tests the happy path scenarios for PII anonymization functionality.
"""

from presidio_analyzer.recognizer_result import RecognizerResult

from netra.anonymizer.base import AnonymizationResult, BaseAnonymizer


class TestAnonymizationResult:
    """Test AnonymizationResult dataclass."""

    def test_anonymization_result_creation(self):
        """Test creating AnonymizationResult with valid data."""
        # Arrange
        masked_text = "Hello <EMAIL_abc123>, your phone is <PHONE_def456>"
        entities = {"EMAIL_abc123": "john@example.com", "PHONE_def456": "555-1234"}

        # Act
        result = AnonymizationResult(masked_text=masked_text, entities=entities)

        # Assert
        assert result.masked_text == masked_text
        assert result.entities == entities


class TestBaseAnonymizerInitialization:
    """Test BaseAnonymizer initialization."""

    def test_default_initialization(self):
        """Test BaseAnonymizer initialization with default parameters."""
        # Act
        anonymizer = BaseAnonymizer()

        # Assert
        assert anonymizer.cache_size == 1000
        assert anonymizer._entity_hash_cache is not None
        assert len(anonymizer._entity_hash_cache) == 0
        assert anonymizer.hash_function is not None

    def test_initialization_with_custom_cache_size(self):
        """Test BaseAnonymizer initialization with custom cache size."""
        # Act
        anonymizer = BaseAnonymizer(cache_size=500)

        # Assert
        assert anonymizer.cache_size == 500
        assert anonymizer._entity_hash_cache is not None

    def test_initialization_with_cache_disabled(self):
        """Test BaseAnonymizer initialization with cache disabled."""
        # Act
        anonymizer = BaseAnonymizer(cache_size=0)

        # Assert
        assert anonymizer.cache_size == 0
        assert anonymizer._entity_hash_cache is None

    def test_initialization_with_custom_hash_function(self):
        """Test BaseAnonymizer initialization with custom hash function."""

        # Arrange
        def custom_hash(value: str) -> str:
            return f"custom_{hash(value)}"

        # Act
        anonymizer = BaseAnonymizer(hash_function=custom_hash)

        # Assert
        assert anonymizer.hash_function == custom_hash


class TestBaseAnonymizerHashGeneration:
    """Test hash generation functionality."""

    def test_default_hash_function(self):
        """Test default hash function generates consistent hashes."""
        # Arrange
        anonymizer = BaseAnonymizer()
        test_value = "test@example.com"

        # Act
        hash1 = anonymizer._default_hash_function(test_value)
        hash2 = anonymizer._default_hash_function(test_value)

        # Assert
        assert hash1 == hash2
        assert len(hash1) == 8  # SHA-256 truncated to 8 characters
        assert isinstance(hash1, str)

    def test_entity_hash_generation_with_cache(self):
        """Test entity hash generation with caching enabled."""
        # Arrange
        anonymizer = BaseAnonymizer(cache_size=100)

        # Act
        hash1 = anonymizer._get_entity_hash("EMAIL", "john@example.com")
        hash2 = anonymizer._get_entity_hash("EMAIL", "john@example.com")

        # Assert
        assert hash1 == hash2
        assert hash1.startswith("EMAIL_")
        assert len(anonymizer._entity_hash_cache) == 1

    def test_entity_hash_generation_without_cache(self):
        """Test entity hash generation with caching disabled."""
        # Arrange
        anonymizer = BaseAnonymizer(cache_size=0)

        # Act
        hash1 = anonymizer._get_entity_hash("PHONE", "555-1234")
        hash2 = anonymizer._get_entity_hash("PHONE", "555-1234")

        # Assert
        assert hash1 == hash2
        assert hash1.startswith("PHONE_")
        assert anonymizer._entity_hash_cache is None


class TestBaseAnonymizerEntityAnonymization:
    """Test single entity anonymization."""

    def test_anonymize_single_entity(self):
        """Test anonymizing a single entity value."""
        # Arrange
        anonymizer = BaseAnonymizer()

        # Act
        result = anonymizer.anonymize_entity("EMAIL", "john@example.com")

        # Assert
        assert result.startswith("<EMAIL_")
        assert result.endswith(">")
        assert len(result) > 10  # Should have meaningful content

    def test_anonymize_different_entity_types(self):
        """Test anonymizing different entity types."""
        # Arrange
        anonymizer = BaseAnonymizer()

        # Act
        email_result = anonymizer.anonymize_entity("EMAIL", "john@example.com")
        phone_result = anonymizer.anonymize_entity("PHONE", "555-1234")

        # Assert
        assert email_result.startswith("<EMAIL_")
        assert phone_result.startswith("<PHONE_")
        assert email_result != phone_result


class TestBaseAnonymizerTextAnonymization:
    """Test full text anonymization functionality."""

    def test_anonymize_text_with_single_entity(self):
        """Test anonymizing text with a single entity."""
        # Arrange
        anonymizer = BaseAnonymizer()
        text = "Contact me at john@example.com"
        analyzer_results = [RecognizerResult(entity_type="EMAIL", start=14, end=30, score=0.9)]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "john@example.com" not in result.masked_text
        assert result.masked_text.startswith("Contact me at <EMAIL_")
        assert len(result.entities) == 1
        assert "john@example.com" in result.entities.values()

    def test_anonymize_text_with_multiple_entities(self):
        """Test anonymizing text with multiple entities."""
        # Arrange
        anonymizer = BaseAnonymizer()
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
        assert "<EMAIL_" in result.masked_text
        assert "<PHONE_" in result.masked_text
        assert len(result.entities) == 2
        assert "john@example.com" in result.entities.values()
        assert "555-1234" in result.entities.values()

    def test_anonymize_text_with_same_entity_multiple_times(self):
        """Test anonymizing text where the same entity appears multiple times."""
        # Arrange
        anonymizer = BaseAnonymizer()
        text = "Email john@example.com and also john@example.com"
        analyzer_results = [
            RecognizerResult(entity_type="EMAIL", start=6, end=22, score=0.9),
            RecognizerResult(entity_type="EMAIL", start=32, end=48, score=0.9),
        ]

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert "john@example.com" not in result.masked_text
        # Should have same hash for same entity value
        hash_placeholders = [part for part in result.masked_text.split() if part.startswith("<EMAIL_")]
        assert len(hash_placeholders) == 2
        assert hash_placeholders[0] == hash_placeholders[1]  # Same entity should have same hash
        assert len(result.entities) == 1  # Only one unique entity

    def test_anonymize_empty_text(self):
        """Test anonymizing empty text."""
        # Arrange
        anonymizer = BaseAnonymizer()
        text = ""
        analyzer_results = []

        # Act
        result = anonymizer.anonymize(text, analyzer_results)

        # Assert
        assert isinstance(result, AnonymizationResult)
        assert result.masked_text == ""
        assert len(result.entities) == 0
