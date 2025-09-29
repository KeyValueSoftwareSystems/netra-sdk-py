"""Netra OpenTelemetry tracer configuration module.

This module handles the initialization and configuration of OpenTelemetry tracing,
including exporter setup and span processor configuration.
"""

import logging
from typing import Any, Dict, List, Sequence

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from netra.config import Config

logger = logging.getLogger(__name__)


class FilteringSpanExporter(SpanExporter):  # type: ignore[misc]
    """
    SpanExporter wrapper that filters out spans by name.

    Matching rules:
    - Exact match: pattern "Foo" blocks span.name == "Foo".
    - Prefix match: pattern ending with '*' (e.g., "CloudSpanner.*") blocks spans whose
      names start with the prefix before '*', e.g., "CloudSpanner.", "CloudSpanner.Query".
    - Suffix match: pattern starting with '*' (e.g., "*.Query") blocks spans whose
      names end with the suffix after '*', e.g., "DB.Query", "Search.Query".
    """

    def __init__(self, exporter: SpanExporter, patterns: Sequence[str]) -> None:
        self._exporter = exporter
        # Normalize once for efficient checks
        exact: List[str] = []
        prefixes: List[str] = []
        suffixes: List[str] = []
        for p in patterns:
            if not p:
                continue
            if p.endswith("*") and not p.startswith("*"):
                prefixes.append(p[:-1])
            elif p.startswith("*") and not p.endswith("*"):
                suffixes.append(p[1:])
            else:
                exact.append(p)
        self._exact = set(exact)
        self._prefixes = prefixes
        self._suffixes = suffixes

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        filtered: List[ReadableSpan] = []
        blocked_parent_map: Dict[Any, Any] = {}
        for span in spans:
            name = getattr(span, "name", None)
            if name is None or not self._is_blocked(name):
                filtered.append(span)
                continue

            span_context = getattr(span, "context", None)
            span_id = getattr(span_context, "span_id", None) if span_context else None
            if span_id is not None:
                blocked_parent_map[span_id] = getattr(span, "parent", None)
        if blocked_parent_map:
            self._reparent_blocked_children(filtered, blocked_parent_map)
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._exporter.export(filtered)

    def _is_blocked(self, name: str) -> bool:
        if name in self._exact:
            return True
        for pref in self._prefixes:
            if name.startswith(pref):
                return True
        for suf in self._suffixes:
            if name.endswith(suf):
                return True
        return False

    def _reparent_blocked_children(
        self,
        spans: Sequence[ReadableSpan],
        blocked_parent_map: Dict[Any, Any],
    ) -> None:
        if not blocked_parent_map:
            return

        for span in spans:
            parent_context = getattr(span, "parent", None)
            if parent_context is None:
                continue

            updated_parent = parent_context
            visited: set[Any] = set()
            changed = False

            while updated_parent is not None:
                parent_span_id = getattr(updated_parent, "span_id", None)
                if parent_span_id not in blocked_parent_map or parent_span_id in visited:
                    break
                visited.add(parent_span_id)
                updated_parent = blocked_parent_map[parent_span_id]
                changed = True

            if changed:
                self._set_span_parent(span, updated_parent)

    def _set_span_parent(self, span: ReadableSpan, parent: Any) -> None:
        if hasattr(span, "_parent"):
            try:
                span._parent = parent
                return
            except Exception:
                pass
        try:
            setattr(span, "parent", parent)
        except Exception:
            logger.debug("Failed to reparent span %s", getattr(span, "name", "<unknown>"), exc_info=True)

    def shutdown(self) -> None:
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            return True


class Tracer:
    """
    Configures Netra's OpenTelemetry tracer with OTLP exporter (or Console exporter as fallback)
    and appropriate span processor.
    """

    def __init__(self, cfg: Config) -> None:
        """Initialize the Netra tracer with the provided configuration.

        Args:
            cfg: Configuration object with tracer settings
        """
        self.cfg = cfg
        self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Set up the OpenTelemetry tracer with appropriate exporters and processors.

        Creates a resource with service name and custom attributes,
        configures the appropriate exporter (OTLP or Console fallback),
        and sets up either a batch or simple span processor based on configuration.
        """
        # Create Resource with service.name + custom attributes
        resource_attrs: Dict[str, Any] = {
            SERVICE_NAME: self.cfg.app_name,
            DEPLOYMENT_ENVIRONMENT: self.cfg.environment,
        }
        if self.cfg.resource_attributes:
            resource_attrs.update(self.cfg.resource_attributes)
        resource = Resource(attributes=resource_attrs)

        # Build TracerProvider
        provider = TracerProvider(resource=resource)

        # Configure exporter based on configuration
        if not self.cfg.otlp_endpoint:
            logger.warning("OTLP endpoint not provided, falling back to console exporter")
            exporter = ConsoleSpanExporter()
        else:
            exporter = OTLPSpanExporter(
                endpoint=self._format_endpoint(self.cfg.otlp_endpoint),
                headers=self.cfg.headers,
            )
        # Wrap exporter with filtering if blocked span patterns are provided
        try:
            patterns = getattr(self.cfg, "blocked_spans", None)
            if patterns:
                exporter = FilteringSpanExporter(exporter, patterns)
                logger.info("Enabled FilteringSpanExporter with %d pattern(s)", len(patterns))
        except Exception as e:
            logger.warning("Failed to enable FilteringSpanExporter: %s", e)
        # Add span processors: first instrumentation wrapper, then session processor
        from netra.processors import InstrumentationSpanProcessor, ScrubbingSpanProcessor, SessionSpanProcessor

        provider.add_span_processor(InstrumentationSpanProcessor())
        provider.add_span_processor(SessionSpanProcessor())

        # Add scrubbing processor if enabled
        if self.cfg.enable_scrubbing:
            provider.add_span_processor(ScrubbingSpanProcessor())  # type: ignore[no-untyped-call]

        # Install appropriate span processor
        if self.cfg.disable_batch:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        else:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        logger.info(
            "Netra TracerProvider initialized: endpoint=%s, disable_batch=%s",
            self.cfg.otlp_endpoint,
            self.cfg.disable_batch,
        )

    def _format_endpoint(self, endpoint: str) -> str:
        """Format the OTLP endpoint URL to ensure it ends with '/v1/traces'.

        Args:
            endpoint: Base OTLP endpoint URL

        Returns:
            Properly formatted endpoint URL
        """
        if not endpoint.endswith("/v1/traces"):
            return endpoint.rstrip("/") + "/v1/traces"
        return endpoint
