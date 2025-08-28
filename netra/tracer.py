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
        for s in spans:
            name = getattr(s, "name", None)
            if name is None:
                filtered.append(s)
                continue
            if name in self._exact:
                continue
            blocked = False
            for pref in self._prefixes:
                if name.startswith(pref):
                    blocked = True
                    break
            if not blocked and self._suffixes:
                for suf in self._suffixes:
                    if name.endswith(suf):
                        blocked = True
                        break
            if not blocked:
                filtered.append(s)
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._exporter.export(filtered)

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
