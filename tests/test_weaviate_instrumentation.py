"""
Unit tests for WeaviateInstrumentor class.
Minimal tests focusing on core functionality and happy path scenarios.
"""

import logging
from typing import Collection
from unittest.mock import Mock, patch

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

    @patch("netra.instrumentation.weaviate.get_tracer")
    @patch("netra.instrumentation.weaviate.wrap_function_wrapper")
    @patch("netra.instrumentation.weaviate.importlib.import_module")
    def test_instrument_with_default_parameters(self, mock_import_module, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with default parameters."""
        # Arrange
        instrumentor = WeaviateInstrumentor()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Create mock modules with the required attributes
        mock_collections_module = Mock()
        mock_collections_module._Collections = Mock()

        mock_data_module = Mock()
        mock_data_module._DataCollection = Mock()

        mock_batch_module = Mock()
        mock_batch_module._BatchCollection = Mock()

        mock_query_module = Mock()
        mock_query_module._QueryGRPC = Mock()

        mock_client_module = Mock()
        mock_client_module.WeaviateClient = Mock()

        # Map module names to mock modules
        module_map = {
            "weaviate.collections.collections": mock_collections_module,
            "weaviate.collections.data": mock_data_module,
            "weaviate.collections.batch.collection": mock_batch_module,
            "weaviate.collections.grpc.query": mock_query_module,
            "weaviate.client": mock_client_module,
        }
        mock_import_module.side_effect = lambda name: module_map.get(name, Mock())

        # Act
        instrumentor._instrument()

        # Assert
        mock_get_tracer.assert_called_once()
        # Should wrap all methods defined in WRAPPED_METHODS (11 methods)
        assert mock_wrap_function.call_count == 11

    @patch("netra.instrumentation.weaviate.get_tracer")
    @patch("netra.instrumentation.weaviate.wrap_function_wrapper")
    @patch("netra.instrumentation.weaviate.importlib.import_module")
    def test_instrument_with_custom_tracer_provider(self, mock_import_module, mock_wrap_function, mock_get_tracer):
        """Test _instrument method with custom tracer provider."""
        # Arrange
        instrumentor = WeaviateInstrumentor()
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        # Create mock modules with the required attributes
        mock_collections_module = Mock()
        mock_collections_module._Collections = Mock()

        mock_data_module = Mock()
        mock_data_module._DataCollection = Mock()

        mock_batch_module = Mock()
        mock_batch_module._BatchCollection = Mock()

        mock_query_module = Mock()
        mock_query_module._QueryGRPC = Mock()

        mock_client_module = Mock()
        mock_client_module.WeaviateClient = Mock()

        # Map module names to mock modules
        module_map = {
            "weaviate.collections.collections": mock_collections_module,
            "weaviate.collections.data": mock_data_module,
            "weaviate.collections.batch.collection": mock_batch_module,
            "weaviate.collections.grpc.query": mock_query_module,
            "weaviate.client": mock_client_module,
        }
        mock_import_module.side_effect = lambda name: module_map.get(name, Mock())

        # Act
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # Assert
        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.weaviate", mock_get_tracer.call_args[0][1], mock_tracer_provider  # version
        )
        assert mock_wrap_function.call_count == 11

    @patch("netra.instrumentation.weaviate.unwrap")
    @patch("netra.instrumentation.weaviate.importlib.import_module")
    def test_uninstrument(self, mock_import_module, mock_unwrap):
        """Test _uninstrument method unwraps all wrapped methods."""
        # Arrange
        instrumentor = WeaviateInstrumentor()

        # Create mock modules with the required attributes
        mock_collections_module = Mock()
        mock_collections_module._Collections = Mock()

        mock_data_module = Mock()
        mock_data_module._DataCollection = Mock()

        mock_batch_module = Mock()
        mock_batch_module._BatchCollection = Mock()

        mock_query_module = Mock()
        mock_query_module._QueryGRPC = Mock()

        mock_client_module = Mock()
        mock_client_module.WeaviateClient = Mock()

        # Map module names to mock modules
        module_map = {
            "weaviate.collections.collections": mock_collections_module,
            "weaviate.collections.data": mock_data_module,
            "weaviate.collections.batch.collection": mock_batch_module,
            "weaviate.collections.grpc.query": mock_query_module,
            "weaviate.client": mock_client_module,
        }
        mock_import_module.side_effect = lambda name: module_map.get(name, Mock())

        # Act
        instrumentor._uninstrument()

        # Assert
        # Should unwrap all methods defined in WRAPPED_METHODS (11 methods)
        assert mock_unwrap.call_count == 11
