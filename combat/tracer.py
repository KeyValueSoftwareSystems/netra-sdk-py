"""Combat OpenTelemetry tracer configuration module.

This module handles the initialization and configuration of OpenTelemetry tracing,
including exporter setup and span processor configuration.
"""

import logging
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.sdk.resources import (DEPLOYMENT_ENVIRONMENT, SERVICE_NAME,
                                         Resource)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor,
                                            ConsoleSpanExporter,
                                            SimpleSpanProcessor)

from .config import Config
from .session import SessionSpanProcessor

logger = logging.getLogger(__name__)


class Tracer:
    """
    Configures Combat's OpenTelemetry tracer with OTLP exporter (or Console exporter as fallback)
    and appropriate span processor.
    """

    def __init__(self, cfg: Config) -> None:
        """Initialize the Combat tracer with the provided configuration.

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
            logger.warning(
                "OTLP endpoint not provided, falling back to console exporter"
            )
            exporter = ConsoleSpanExporter()
        else:
            exporter = OTLPSpanExporter(
                endpoint=self._format_endpoint(self.cfg.otlp_endpoint),
                headers=self._format_headers(),
            )
        # Add session span processor
        provider.add_span_processor(SessionSpanProcessor())

        # Install appropriate span processor
        if self.cfg.disable_batch:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        else:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set global tracer provider
        trace.set_tracer_provider(provider)
        logger.info(
            "Combat TracerProvider initialized: endpoint=%s, disable_batch=%s",
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

    def _format_headers(self) -> Optional[Dict[str, str]]:
        """Generate request headers for the OTLP exporter.

        Constructs headers from either:
        1. Explicit header string in format "key1=value1,key2=value2"
        2. API key in Authorization bearer format

        Returns:
            Dictionary of header key-value pairs or None if no headers needed
        """
        if self.cfg.headers:
            # Parse headers from string format: "key1=value1,key2=value2"
            header_pairs = {}
            try:
                if not isinstance(self.cfg.headers, str):
                    logger.warning("Headers configuration is not a string. Using empty headers.")
                    return {}

                for pair in self.cfg.headers.split(","):
                    pair = pair.strip()
                    if not pair:  # Skip empty pairs
                        continue

                    if "=" in pair:
                        try:
                            key, value = pair.split("=", 1)
                            header_pairs[key.strip()] = value.strip()
                        except Exception as e:
                            logger.warning(f"Failed to parse header pair '{pair}': {str(e)}")
                    else:
                        logger.warning(f"Ignoring malformed header without '=' delimiter: '{pair}'")

                return header_pairs
            except Exception as e:
                logger.warning(f"Error parsing headers: {str(e)}. Using empty headers.")
                return {}
        elif self.cfg.api_key:
            # Use API key as bearer token
            return {"Authorization": f"Bearer {self.cfg.api_key}"}

        return None
