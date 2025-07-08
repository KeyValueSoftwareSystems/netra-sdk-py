"""
Unit tests for SpanAggregationProcessor and SpanAggregationData classes.
Minimal tests focusing on core functionality and happy path scenarios.
"""

import json
from unittest.mock import Mock, patch

from netra.processors.span_aggregation_processor import SpanAggregationData, SpanAggregationProcessor


class TestSpanAggregationData:
    """Test SpanAggregationData core functionality."""

    def test_initialization(self):
        """Test SpanAggregationData initialization with default values."""
        # Act
        data = SpanAggregationData()

        # Assert
        assert data.tokens == {}
        assert data.models == set()
        assert data.has_pii is False
        assert data.pii_entities == set()
        assert data.pii_actions == {}
        assert data.has_violation is False
        assert data.violations == set()
        assert data.violation_actions == {}
        assert data.has_error is False
        assert data.status_codes == set()

    def test_merge_from_other(self):
        """Test merging data from another SpanAggregationData instance."""
        # Arrange
        data1 = SpanAggregationData()
        data1.has_pii = True
        data1.pii_entities.add("EMAIL")
        data1.models.add("gpt-4")
        data1.tokens["gpt-4"]["input"] = 100

        data2 = SpanAggregationData()
        data2.has_error = True
        data2.status_codes.add(500)
        data2.models.add("claude")
        data2.tokens["gpt-4"]["input"] = 50  # Lower value, should not override
        data2.tokens["gpt-4"]["output"] = 75

        # Act
        data1.merge_from_other(data2)

        # Assert
        assert data1.has_pii is True
        assert data1.has_error is True
        assert data1.status_codes == {500}
        assert data1.models == {"gpt-4", "claude"}
        assert data1.tokens["gpt-4"]["input"] == 100  # Max value preserved
        assert data1.tokens["gpt-4"]["output"] == 75

    def test_to_attributes(self):
        """Test converting aggregated data to span attributes."""
        # Arrange
        data = SpanAggregationData()
        data.has_pii = True
        data.pii_entities.add("EMAIL")
        data.models.add("gpt-4")
        data.tokens["gpt-4"] = {"input": 100, "output": 50}

        # Act
        attributes = data.to_attributes()

        # Assert
        assert attributes["has_pii"] == "true"
        assert attributes["has_error"] == "false"
        assert attributes["has_violation"] == "false"
        assert json.loads(attributes["pii_entities"]) == ["EMAIL"]
        assert json.loads(attributes["models"]) == ["gpt-4"]
        assert json.loads(attributes["tokens"]) == {"gpt-4": {"input": 100, "output": 50}}


class TestSpanAggregationProcessor:
    """Test SpanAggregationProcessor core functionality."""

    def test_initialization(self):
        """Test SpanAggregationProcessor initialization."""
        # Act
        processor = SpanAggregationProcessor()

        # Assert
        assert processor._span_data == {}
        assert processor._span_hierarchy == {}
        assert processor._root_spans == set()
        assert processor._captured_data == {}
        assert processor._active_spans == {}

    @patch("netra.processors.span_aggregation_processor.SpanAggregationProcessor._wrap_span_methods")
    def test_on_start_root_span(self, mock_wrap_methods):
        """Test on_start method with root span (no parent)."""
        # Arrange
        processor = SpanAggregationProcessor()
        mock_span = Mock()
        mock_span.parent = None

        with patch.object(processor, "_get_span_id", return_value="span-123"):
            # Act
            processor.on_start(mock_span)

            # Assert
            assert "span-123" in processor._span_data
            assert "span-123" in processor._captured_data
            assert "span-123" in processor._active_spans
            assert "span-123" in processor._root_spans
            assert processor._active_spans["span-123"] == mock_span
            mock_wrap_methods.assert_called_once_with(mock_span, "span-123")

    @patch("netra.processors.span_aggregation_processor.SpanAggregationProcessor._wrap_span_methods")
    def test_on_start_child_span(self, mock_wrap_methods):
        """Test on_start method with child span (has parent)."""
        # Arrange
        processor = SpanAggregationProcessor()
        mock_span = Mock()
        mock_parent_context = Mock()
        mock_parent_context.span_id = 456
        mock_parent_context.trace_id = 789
        mock_span.parent = mock_parent_context

        with patch.object(processor, "_get_span_id", return_value="span-123"):
            # Act
            processor.on_start(mock_span)

            # Assert
            assert "span-123" in processor._span_data
            assert "span-123" not in processor._root_spans
            expected_parent_id = f"{789:032x}-{456:016x}"
            assert processor._span_hierarchy["span-123"] == expected_parent_id
            mock_wrap_methods.assert_called_once_with(mock_span, "span-123")

    def test_on_end_span(self):
        """Test on_end method processing span data."""
        # Arrange
        processor = SpanAggregationProcessor()
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        # Setup processor state
        span_id = "span-123"
        processor._span_data[span_id] = SpanAggregationData()
        processor._captured_data[span_id] = {"attributes": {"key": "value"}, "events": []}
        processor._active_spans[span_id] = mock_span

        with patch.object(processor, "_get_span_id", return_value=span_id):
            # Act
            processor.on_end(mock_span)

            # Assert - verify span was processed and cleaned up
            assert span_id not in processor._span_data
            assert span_id not in processor._captured_data
            assert span_id not in processor._active_spans

    def test_force_flush(self):
        """Test force_flush method."""
        # Arrange
        processor = SpanAggregationProcessor()

        # Act & Assert
        result = processor.force_flush()
        assert result is True

        result = processor.force_flush(timeout_millis=5000)
        assert result is True

    def test_shutdown(self):
        """Test shutdown method clears all data structures."""
        # Arrange
        processor = SpanAggregationProcessor()
        processor._span_data["test"] = SpanAggregationData()
        processor._span_hierarchy["child"] = "parent"
        processor._root_spans.add("root")
        processor._captured_data["test"] = {}
        processor._active_spans["test"] = Mock()

        # Act
        result = processor.shutdown()

        # Assert
        assert result is True
        assert processor._span_data == {}
        assert processor._span_hierarchy == {}
        assert processor._root_spans == set()
        assert processor._captured_data == {}
        assert processor._active_spans == {}
