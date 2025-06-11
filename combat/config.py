import json
import os
from typing import Optional, Dict, Any
from .version import __version__


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
    SDK_NAME = "combat"
    LIBRARY_NAME = "combat"
    LIBRARY_VERSION = __version__

    def __init__(
        self,
        app_name: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None
    ):
        # Application name: from param, else env
        self.app_name = (
            app_name
            or os.getenv("OTEL_SERVICE_NAME")
            or os.getenv("COMBAT_APP_NAME")
            or "llm_tracing_service"
        )

        # OTLP endpoint: if explicit param, else OTEL_EXPORTER_OTLP_ENDPOINT
        self.otlp_endpoint = (
            otlp_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        )

        # API key: if explicit param, else env COMBAT_API_KEY
        self.api_key = api_key or os.getenv("COMBAT_API_KEY")

        # Custom headers: comma-separated W3C format (if provided, overrides API key)
        self.headers = headers or os.getenv("COMBAT_HEADERS")

        # Disable batch span processor?
        if disable_batch is not None:
            self.disable_batch = disable_batch
        else:
            # Environment var can be "true"/"false"
            env_db = os.getenv("COMBAT_DISABLE_BATCH")
            self.disable_batch = (
                True
                if (env_db is not None and env_db.lower() in ("1", "true"))
                else False
            )

        # Trace content (prompts/completions)? Default true unless env says false
        if trace_content is not None:
            self.trace_content = trace_content
        else:
            env_tc = os.getenv("COMBAT_TRACE_CONTENT")
            self.trace_content = (
                False
                if (env_tc is not None and env_tc.lower() in ("0", "false"))
                else True
            )

        # 7. Environment: param override, else env
        if environment is not None:
            self.environment = environment
        else:
            self.environment = os.getenv("COMBAT_ENV", "local")

        # Resource attributes: param override, else parse JSON from env, else empty dict
        if resource_attributes is not None:
            self.resource_attributes = resource_attributes
        else:
            # Expecting something like: {"env":"prod","version":"1.0.0"}
            env_ra = os.getenv("COMBAT_RESOURCE_ATTRS")
            if env_ra:
                try:
                    self.resource_attributes = json.loads(env_ra)
                except (json.JSONDecodeError, ValueError) as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to parse COMBAT_RESOURCE_ATTRS: {e}")
                    self.resource_attributes = {}
            else:
                self.resource_attributes = {}
