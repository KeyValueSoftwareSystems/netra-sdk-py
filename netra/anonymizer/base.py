"""
Base anonymizer class for PII data anonymization.

This module provides the base anonymizer class that contains the core anonymization
logic that can be extended by specific anonymizer implementations.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from presidio_analyzer.recognizer_result import RecognizerResult


@dataclass
class AnonymizationResult:
    """
    Result of anonymization containing the masked text and entity mappings.

    Attributes:
        masked_text: The text with PII entities replaced by hash placeholders.
        entities: Dictionary mapping entity hashes to their original values.
    """

    masked_text: str
    entities: Dict[str, str]


class BaseAnonymizer:
    """
    Base anonymizer that replaces entities with consistent hash values.

    This base anonymizer provides the core anonymization logic that can be
    extended by specific anonymizer implementations for different entity types.
    """

    def __init__(self, hash_function: Optional[Callable[[str], str]] = None, cache_size: int = 1000):
        """
        Initialize the BaseAnonymizer.

        Args:
            hash_function: Optional custom hash function that takes a string and returns a hash.
                           If not provided, a default hash function will be used.
            cache_size: Maximum number of entities to cache. Uses LRU eviction policy.
                       Default is 1000. Set to 0 to disable caching.
        """
        self.hash_function = hash_function or self._default_hash_function
        self.cache_size = cache_size

        # Initialize LRU cache for entity hashes
        if cache_size > 0:
            self._entity_hash_cache: Optional[OrderedDict[str, str]] = OrderedDict()
        else:
            self._entity_hash_cache = None

    def _default_hash_function(self, value: str) -> str:
        """
        Default hash function using SHA-256.

        Args:
            value: The string to hash.

        Returns:
            A hexadecimal hash string.
        """
        return hashlib.sha256(value.encode()).hexdigest()[:8]

    def _get_entity_hash(self, entity_type: str, entity_value: str) -> str:
        """
        Get a consistent hash for an entity value, creating one if it doesn't exist.
        Uses LRU cache with configurable size to balance performance and memory usage.

        Args:
            entity_type: The type of entity (e.g., 'EMAIL', 'PHONE', etc.)
            entity_value: The original value of the entity.

        Returns:
            A hash string for the entity.
        """
        # Skip caching if cache_size is 0
        if self.cache_size == 0:
            entity_hash = f"{entity_type}_{self.hash_function(entity_value)}"
            return entity_hash

        # Create a composite key for the entity cache
        cache_key = f"{entity_type}:{entity_value}"

        # Check if entity exists in cache and move to end (mark as recently used)
        if self._entity_hash_cache is not None and cache_key in self._entity_hash_cache:
            # Move to end to mark as recently used
            self._entity_hash_cache.move_to_end(cache_key)
            return self._entity_hash_cache[cache_key]

        # Generate a new hash for this entity
        entity_hash = f"{entity_type}_{self.hash_function(entity_value)}"

        # Add to cache if cache is enabled
        if self._entity_hash_cache is not None:
            self._entity_hash_cache[cache_key] = entity_hash

            # Evict oldest entry if cache exceeds size limit
            if len(self._entity_hash_cache) > self.cache_size:
                # Remove the least recently used item (first item)
                self._entity_hash_cache.popitem(last=False)

        return entity_hash

    def anonymize_entity(self, entity_type: str, entity_value: str) -> str:
        """
        Anonymize a single entity value.

        Args:
            entity_type: The type of entity (e.g., 'EMAIL', 'PHONE', etc.)
            entity_value: The original value of the entity.

        Returns:
            The anonymized entity value.
        """
        # Get or create hash for this entity
        entity_hash = self._get_entity_hash(entity_type, entity_value)
        return f"<{entity_hash}>"

    def anonymize(self, text: str, analyzer_results: List[RecognizerResult]) -> AnonymizationResult:
        """
        Anonymize text by replacing detected entities with hash values.

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

        # Dictionary to store mapping of hash values to original entity values
        entities_map: Dict[str, str] = {}

        # Replace each entity with its hash
        for result in sorted_results:
            entity_type = result.entity_type
            entity_value = text[result.start : result.end]

            # Get or create hash for this entity
            entity_hash = self._get_entity_hash(entity_type, entity_value)

            # Replace the entity in the text with the hash placeholder
            placeholder = f"<{entity_hash}>"
            masked_text = masked_text[: result.start] + placeholder + masked_text[result.end :]

            # Store the mapping of hash to original value
            entities_map[entity_hash] = entity_value

        return AnonymizationResult(masked_text=masked_text, entities=entities_map)
