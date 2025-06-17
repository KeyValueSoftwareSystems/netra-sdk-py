"""
Span aggregation utilities for Combat SDK.
Handles aggregation of child span data into parent spans.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Context, Span

from .config import Config

logger = logging.getLogger(__name__)


class SpanAggregationData:
    """Holds aggregated data for a span."""

    def __init__(self) -> None:
        self.tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.models: Set[str] = set()
        self.has_pii: bool = False
        self.pii_entities: Set[str] = set()
        self.pii_actions: Dict[str, Set[str]] = defaultdict(set)
        self.has_violation: bool = False
        self.violations: Set[str] = set()
        self.violation_actions: Dict[str, Set[str]] = defaultdict(set)

    def merge_from_other(self, other: "SpanAggregationData") -> None:
        """Merge data from another SpanAggregationData instance."""
        # Merge tokens - take the maximum values for each model
        for model, token_data in other.tokens.items():
            if model not in self.tokens:
                self.tokens[model] = {}
            for token_type, count in token_data.items():
                self.tokens[model][token_type] = max(self.tokens[model].get(token_type, 0), count)

        # Merge models
        self.models.update(other.models)

        # Merge PII data
        if other.has_pii:
            self.has_pii = True
        self.pii_entities.update(other.pii_entities)
        for action, entities in other.pii_actions.items():
            self.pii_actions[action].update(entities)

        # Merge violation data
        if other.has_violation:
            self.has_violation = True
        self.violations.update(other.violations)
        for action, violations in other.violation_actions.items():
            self.violation_actions[action].update(violations)

    def to_attributes(self) -> Dict[str, str]:
        """Convert aggregated data to span attributes."""
        attributes = {}

        # Token usage by model
        if self.tokens:
            tokens_dict = {}
            for model, usage in self.tokens.items():
                tokens_dict[model] = dict(usage)
            attributes["tokens"] = json.dumps(tokens_dict)

        # Models used
        if self.models:
            attributes["models"] = json.dumps(sorted(list(self.models)))

        # PII information
        attributes["has_pii"] = str(self.has_pii).lower()
        if self.pii_entities:
            attributes["pii_entities"] = json.dumps(sorted(list(self.pii_entities)))
        if self.pii_actions:
            pii_actions_dict = {}
            for action, entities in self.pii_actions.items():
                pii_actions_dict[action] = sorted(list(entities))
            attributes["pii_actions"] = json.dumps(pii_actions_dict)

        # Violation information
        attributes["has_violation"] = str(self.has_violation).lower()
        if self.violations:
            attributes["violations"] = json.dumps(sorted(list(self.violations)))
        if self.violation_actions:
            violation_actions_dict = {}
            for action, violations in self.violation_actions.items():
                violation_actions_dict[action] = sorted(list(violations))
            attributes["violation_actions"] = json.dumps(violation_actions_dict)

        return attributes


class SpanAggregationProcessor(SpanProcessor):  # type: ignore[misc]
    """
    OpenTelemetry span processor that aggregates data from child spans into parent spans.
    """

    def __init__(self) -> None:
        self._span_data: Dict[str, SpanAggregationData] = {}
        self._span_hierarchy: Dict[str, Optional[str]] = {}  # child_id -> parent_id
        self._root_spans: Set[str] = set()
        self._captured_data: Dict[str, Dict[str, Any]] = {}  # span_id -> {attributes, events}
        self._active_spans: Dict[str, Span] = {}  # span_id -> original span reference

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span starts."""
        span_id = self._get_span_id(span)
        if not span_id:
            return

        # Store the original span for later use
        self._active_spans[span_id] = span

        # Initialize aggregation data
        self._span_data[span_id] = SpanAggregationData()
        self._captured_data[span_id] = {"attributes": {}, "events": []}

        # Check if this is a root span (no parent)
        if span.parent is None:
            self._root_spans.add(span_id)
        else:
            # Track parent-child relationship - span.parent is a SpanContext, not a Span
            try:
                parent_span_context = span.parent
                if parent_span_context and parent_span_context.span_id:
                    parent_span_id = f"{parent_span_context.trace_id:032x}-{parent_span_context.span_id:016x}"
                    self._span_hierarchy[span_id] = parent_span_id
                else:
                    logger.warning(f"DEBUG: Parent span context is invalid for child {span_id}")
            except Exception as e:
                logger.warning(f"DEBUG: Could not get parent span ID for child {span_id}: {e}")

        # Wrap span methods to capture data
        self._wrap_span_methods(span, span_id)

    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        span_id = self._get_span_id(span)
        if not span_id or span_id not in self._span_data:
            return

        try:
            # Process this span's captured data
            captured = self._captured_data.get(span_id, {})
            self._process_attributes(self._span_data[span_id], captured.get("attributes", {}))

            # Set aggregated attributes on this span
            original_span = self._active_spans.get(span_id)
            if original_span and original_span.is_recording():
                self._set_span_attributes(original_span, self._span_data[span_id])

            # Handle parent-child aggregation for any remaining data
            self._aggregate_to_all_parents(span_id)

        except Exception as e:
            logger.error(f"Error during span aggregation for span {span_id}: {e}")
            # Even if there's an error, try to do basic aggregation
            try:
                original_span = self._active_spans.get(span_id)
                if original_span and original_span.is_recording():
                    self._set_span_attributes(original_span, self._span_data[span_id])
            except Exception as inner_e:
                logger.error(f"Failed to set basic aggregation attributes: {inner_e}")

        # Clean up
        self._span_data.pop(span_id, None)
        self._captured_data.pop(span_id, None)
        self._active_spans.pop(span_id, None)
        self._root_spans.discard(span_id)
        self._span_hierarchy.pop(span_id, None)

    def _wrap_span_methods(self, span: Span, span_id: str) -> Any:
        """Wrap span methods to capture attributes and events."""
        # Wrap set_attribute
        original_set_attribute = span.set_attribute

        def wrapped_set_attribute(key: str, value: Any) -> Any:
            self._captured_data[span_id]["attributes"][key] = value
            return original_set_attribute(key, value)

        span.set_attribute = wrapped_set_attribute

        # Wrap add_event
        original_add_event = span.add_event

        def wrapped_add_event(name: str, attributes: Dict[str, Any] = {}, timestamp: int = 0) -> Any:
            # Only process PII and violation events
            if name == "pii_detected" and attributes:
                self._process_pii_event(self._span_data[span_id], attributes)
                if span.is_recording():
                    self._set_span_attributes(span, self._span_data[span_id])
                    # Immediately aggregate to parent spans
                    self._aggregate_to_all_parents(span_id)
            elif name == "violation_detected" and attributes:
                self._process_violation_event(self._span_data[span_id], attributes)
                if span.is_recording():
                    self._set_span_attributes(span, self._span_data[span_id])
                    # Immediately aggregate to parent spans
                    self._aggregate_to_all_parents(span_id)

            return original_add_event(name, attributes, timestamp)

        span.add_event = wrapped_add_event

    def _process_attributes(self, data: SpanAggregationData, attributes: Dict[str, Any]) -> None:
        """Process span attributes for aggregation."""
        # Extract model information
        model = attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model")
        if model:
            data.models.add(model)
            # Extract token usage
            token_fields = {
                "prompt_tokens": attributes.get("gen_ai.usage.prompt_tokens", 0),
                "completion_tokens": attributes.get("gen_ai.usage.completion_tokens", 0),
                "total_tokens": attributes.get("llm.usage.total_tokens", 0),
                "cache_read_input_tokens": attributes.get("gen_ai.usage.cache_read_input_tokens", 0),
            }

            # Initialize token fields if they don't exist
            if model not in data.tokens:
                data.tokens[model] = {}

            # Add token values
            for field, value in token_fields.items():
                if isinstance(value, (int, str)):
                    current_value = data.tokens[model].get(field, 0)
                    data.tokens[model][field] = current_value + int(value)

    def _process_pii_event(self, data: SpanAggregationData, attrs: Dict[str, Any]) -> None:
        """Process pii_detected event."""
        if attrs.get("has_pii"):
            data.has_pii = True

            # Extract entities from pii_entities field
            entity_counts_str = attrs.get("pii_entities")
            if entity_counts_str:
                try:
                    entity_counts = (
                        json.loads(entity_counts_str) if isinstance(entity_counts_str, str) else entity_counts_str
                    )
                    if isinstance(entity_counts, dict):
                        entities = set(entity_counts.keys())
                        data.pii_entities.update(entities)

                        # Determine action
                        if attrs.get("is_blocked"):
                            data.pii_actions["BLOCK"].update(entities)
                        elif attrs.get("is_masked"):
                            data.pii_actions["MASK"].update(entities)
                        else:
                            data.pii_actions["FLAG"].update(entities)
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"Failed to parse pii_entities: {entity_counts_str}")

    def _process_violation_event(self, data: SpanAggregationData, attrs: Dict[str, Any]) -> None:
        """Process violation_detected event."""
        if attrs.get("has_violation"):
            data.has_violation = True
            violations = attrs.get("violations", [])
            if violations:
                data.violations.update(violations)
                # Set action based on is_blocked flag
                action = "BLOCK" if attrs.get("is_blocked") else "FLAG"
                data.violation_actions[action].update(violations)

    def _aggregate_to_all_parents(self, child_span_id: str) -> None:
        """Aggregate data from child span to all its parent spans in the hierarchy."""
        if child_span_id not in self._span_data:
            return

        child_data = self._span_data[child_span_id]
        current_span_id = child_span_id

        # Traverse up the parent hierarchy
        while True:
            parent_id = self._span_hierarchy.get(current_span_id)
            if not parent_id or parent_id not in self._span_data:
                break

            # Merge child data into parent
            self._span_data[parent_id].merge_from_other(child_data)

            # Update parent span attributes if it's still active and recording
            parent_span = self._active_spans.get(parent_id)
            if parent_span and parent_span.is_recording():
                self._set_span_attributes(parent_span, self._span_data[parent_id])

            # Move up to the next parent
            current_span_id = parent_id

    def _set_span_attributes(self, span: Span, data: SpanAggregationData) -> None:
        """Set aggregated attributes on the given span."""
        try:
            aggregated_attrs = data.to_attributes()
            # Set all aggregated attributes under a single 'aggregator' key as a JSON object
            span.set_attribute(f"{Config.LIBRARY_NAME}.aggregated_attributes", json.dumps(aggregated_attrs))
        except Exception as e:
            logger.error(f"Failed to set aggregated attributes: {e}")

    def _get_span_id(self, span: Span) -> Optional[str]:
        """Get a unique identifier for the span."""
        try:
            span_context = span.get_span_context()
            return f"{span_context.trace_id:032x}-{span_context.span_id:016x}"
        except Exception:
            return None

    def _get_span_id_from_context(self, context: Context) -> Optional[str]:
        """Extract span ID from context."""
        if context:
            span_context = trace.get_current_span(context).get_span_context()
            if span_context and span_context.span_id:
                return f"{span_context.trace_id:032x}-{span_context.span_id:016x}"
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending data."""
        return True

    def shutdown(self) -> bool:
        """Shutdown the processor."""
        self._span_data.clear()
        self._span_hierarchy.clear()
        self._root_spans.clear()
        self._captured_data.clear()
        self._active_spans.clear()
        return True
