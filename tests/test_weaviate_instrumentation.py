"""
Unit tests for WeaviateInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

import logging
from typing import Collection
from unittest.mock import Mock

from netra.instrumentation.weaviate import WeaviateInstrumentor


class TestWeaviateInstrumentor:
    """Test WeaviateInstrumentor core functionality."""

    def test_initialization(self):
        """Test WeaviateInstrumentor initialization."""
        # Act
        instrumentor = WeaviateInstrumentor()

        # Assert
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_initialization_with_exception_logger(self):
        """Test WeaviateInstrumentor initialization with custom exception logger."""
        # Arrange
        mock_logger = Mock(spec=logging.Logger)

        # Act
        instrumentor = WeaviateInstrumentor(exception_logger=mock_logger)

        # Assert
        assert instrumentor is not None

    def test_instrumentation_dependencies(self):
        """Test instrumentation_dependencies returns correct packages."""
        # Arrange
        instrumentor = WeaviateInstrumentor()

        # Act
        dependencies = instrumentor.instrumentation_dependencies()

        # Assert
        assert isinstance(dependencies, Collection)
        assert "weaviate-client >= 3.26.0, <5" in dependencies
