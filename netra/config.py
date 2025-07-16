import json
import os
from typing import Any, Dict, Optional

from opentelemetry.util.re import parse_env_headers

from netra.version import __version__


class Config:
    """
    Holds configuration options for the tracer:
      - app_name:                Logical name for this service
      - otlp_endpoint:           URL for OTLP collector
      - api_key:                 API key for the collector (sent as Bearer token)
      - headers:                 Additional headers (W3C Correlation-Context format)
      - disable_batch:           Whether to disable batch span processor (bool)
      - trace_content:           Whether to capture prompt/completion content (bool)
      - resource_attributes:     Custom resource attributes dict (e.g., {'env': 'prod', 'version': '1.0.0'})
    """

    # SDK Constants
    SDK_NAME = "netra"
    LIBRARY_NAME = "netra"
    LIBRARY_VERSION = __version__

    def __init__(
        self,
        app_name: Optional[str] = None,
        headers: Optional[str] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
    ):
        # Application name: from param, else env
        self.app_name = (
            app_name or os.getenv("NETRA_APP_NAME") or os.getenv("OTEL_SERVICE_NAME") or "llm_tracing_service"
        )

        # OTLP endpoint: if explicit param, else OTEL_EXPORTER_OTLP_ENDPOINT
        self.otlp_endpoint = os.getenv("NETRA_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

        # API key: if explicit param, else env NETRA_API_KEY
        self.api_key = os.getenv("NETRA_API_KEY")
        self.headers = {}

        # Custom headers: comma-separated W3C format (if provided, overrides API key)
        headers = headers or os.getenv("NETRA_HEADERS")

        if isinstance(headers, str):
            self.headers = parse_env_headers(headers)

        if self.otlp_endpoint in ["https://api.dev.getcombat.ai", "https://api.eu.getnetra.ai"] and not self.api_key:
            print("Error: Missing Netra API key, go to https://app.dev.getcombat.ai/api-key to create one")
            print("Set the NETRA_API_KEY environment variable to the key")
            return

        # Handle API key authentication based on OTLP endpoint
        if self.api_key and self.otlp_endpoint:
            # For Netra endpoints, use x-api-key header
            if "getcombat" in self.otlp_endpoint.lower() or "getnetra" in self.otlp_endpoint.lower():
                if not self.headers:
                    self.headers = {"x-api-key": self.api_key}
                elif "x-api-key" not in self.headers:
                    self.headers = {**self.headers, "x-api-key": self.api_key}
            # For other endpoints, set up basic auth
            else:
                if not self.headers:
                    self.headers = {"Authorization": f"Bearer {self.api_key}"}
                elif "Authorization" not in self.headers:
                    self.headers = {**self.headers, "Authorization": f"Bearer {self.api_key}"}

        # Disable batch span processor?
        if disable_batch is not None:
            self.disable_batch = disable_batch
        else:
            # Environment var can be "true"/"false"
            env_db = os.getenv("NETRA_DISABLE_BATCH")
            self.disable_batch = True if (env_db is not None and env_db.lower() in ("1", "true")) else False

        # Trace content (prompts/completions)? Default true unless env says false
        if trace_content is not None:
            self.trace_content = trace_content
        else:
            env_tc = os.getenv("NETRA_TRACE_CONTENT")
            self.trace_content = False if (env_tc is not None and env_tc.lower() in ("0", "false")) else True

        if not self.trace_content:
            os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
        else:
            os.environ["TRACELOOP_TRACE_CONTENT"] = "true"

        # 7. Environment: param override, else env
        if environment is not None:
            self.environment = environment
        else:
            self.environment = os.getenv("NETRA_ENV", "local")

        # Resource attributes: param override, else parse JSON from env, else empty dict
        if resource_attributes is not None:
            self.resource_attributes = resource_attributes
        else:
            # Expecting something like: {"env":"prod","version":"1.0.0"}
            env_ra = os.getenv("NETRA_RESOURCE_ATTRS")
            if env_ra:
                try:
                    self.resource_attributes = json.loads(env_ra)
                except (json.JSONDecodeError, ValueError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to parse NETRA_RESOURCE_ATTRS: {e}")
                    self.resource_attributes = {}
            else:
                self.resource_attributes = {}
