"""
Custom anonymizer for PII data that provides consistent hashing of entities.

This module provides a custom anonymizer that can be used to replace PII entities
with consistent hash values, allowing for tracking the same entities across multiple
texts while maintaining privacy.
"""

from typing import Callable, List, Optional

try:
    from presidio_analyzer.recognizer_result import RecognizerResult
except Exception:
    raise ImportError(
        "PII Detetcion requires the 'presidio' packages: Install them explicitly as they are not available with the base SDK. Use pip install 'netra-sdk[presidio]' to install them."
    )


from .base import AnonymizationResult, BaseAnonymizer
from .fp_anonymizer import FormatPreservingEmailAnonymizer


class Anonymizer:
    """
    Main anonymizer that delegates to different anonymizer classes based on entity type.

    This anonymizer analyzes the entity types and uses appropriate anonymization
    strategies - format-preserving for email addresses and hash-based for other types.
    """

    def __init__(self, hash_function: Optional[Callable[[str], str]] = None, cache_size: int = 1000):
        """
        Initialize the Anonymizer.

        Args:
            hash_function: Optional custom hash function that takes a string and returns a hash.
                           If not provided, a default hash function will be used.
            cache_size: Maximum number of entities to cache. Uses LRU eviction policy.
                       Default is 1000. Set to 0 to disable caching.
        """
        # Initialize different anonymizer instances
        self.base_anonymizer = BaseAnonymizer(hash_function=hash_function, cache_size=cache_size)
        self.email_anonymizer = FormatPreservingEmailAnonymizer()

    def anonymize(self, text: str, analyzer_results: List[RecognizerResult]) -> AnonymizationResult:
        """
        Anonymize text by replacing detected entities using appropriate anonymization strategies.

        Args:
            text: The original text containing PII.
            analyzer_results: List of RecognizerResult objects from the Presidio analyzer.

        Returns:
            AnonymizationResult containing the masked text and a mapping of entity hashes to original values.
        """
        # Sort results by start index in descending order to avoid offset issues when replacing
        sorted_results = sorted(analyzer_results, key=lambda x: x.start, reverse=True)

        # Make a copy of the original text that we'll modify
        masked_text = text

        # Dictionary to store mapping of anonymized values to original entity values
        entities_map = {}

        # Replace each entity with its anonymized value
        for result in sorted_results:
            entity_type = result.entity_type
            entity_value = text[result.start : result.end]

            # Use appropriate anonymizer based on entity type
            if entity_type.upper() in ["EMAIL", "EMAIL_ADDRESS"]:
                # Use format-preserving email anonymization
                anonymized_value = self.email_anonymizer._anonymize_email(entity_value)
                placeholder = anonymized_value
                entities_map[anonymized_value] = entity_value
            else:
                # Use base anonymizer for other entity types
                entity_hash = self.base_anonymizer._get_entity_hash(entity_type, entity_value)
                placeholder = f"<{entity_hash}>"
                entities_map[entity_hash] = entity_value

            # Replace the entity in the text with the placeholder
            masked_text = masked_text[: result.start] + placeholder + masked_text[result.end :]

        return AnonymizationResult(masked_text=masked_text, entities=entities_map)
