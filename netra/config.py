import json
import os
from typing import Any, Dict, List, Optional

from opentelemetry.util.re import parse_env_headers

from netra.version import __version__


class Config:
    """
    Holds configuration options for the tracer.

    Attributes:
        app_name: Logical name for this service
        otlp_endpoint: URL for OTLP collector
        api_key: API key for the collector (sent as Bearer token)
        headers: Additional headers (W3C Correlation-Context format)
        disable_batch: Whether to disable batch span processor
        trace_content: Whether to capture prompt/completion content
        debug_mode: Whether to enable SDK logging (default: False)
        enable_root_span: Whether to create a process root span (default: False)
        resource_attributes: Custom resource attributes dict (e.g., {'env': 'prod', 'version': '1.0.0'})
        enable_scrubbing: Whether to enable pydantic logfire scrubbing (default: False)
        blocked_spans: List of span names (prefix/suffix patterns) to block from export
    """

    # SDK Constants
    SDK_NAME = "netra"
    LIBRARY_NAME = "netra"
    LIBRARY_VERSION = __version__
    ATTRIBUTE_MAX_LEN = int(os.getenv("NETRA_ATTRIBUTE_MAX_LEN", 50000))
    CONVERSATION_MAX_LEN = int(os.getenv("NETRA_CONVERSATION_CONTENT_MAX_LEN", 50000))

    def __init__(
        self,
        app_name: Optional[str] = None,
        headers: Optional[str] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        enable_root_span: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        enable_scrubbing: Optional[bool] = None,
        blocked_spans: Optional[List[str]] = None,
    ):
        self.app_name = self._get_app_name(app_name)
        self.otlp_endpoint = self._get_otlp_endpoint()
        self.api_key = os.getenv("NETRA_API_KEY")
        self.headers = self._parse_headers(headers)

        self._validate_api_key()
        self._setup_authentication()

        self.disable_batch = self._get_bool_config(disable_batch, "NETRA_DISABLE_BATCH", default=False)
        self.trace_content = self._get_bool_config(trace_content, "NETRA_TRACE_CONTENT", default=True)
        self.debug_mode = self._get_bool_config(debug_mode, "NETRA_DEBUG", default=False)
        self.enable_root_span = self._get_bool_config(enable_root_span, "NETRA_ENABLE_ROOT_SPAN", default=False)
        self.enable_scrubbing = self._get_bool_config(enable_scrubbing, "NETRA_ENABLE_SCRUBBING", default=False)

        self.environment = environment or os.getenv("NETRA_ENV", "local")
        self.resource_attributes = self._get_resource_attributes(resource_attributes)
        self.blocked_spans = blocked_spans

        self._set_trace_content_env()

    def _get_app_name(self, app_name: Optional[str]) -> str:
        """Get application name from param or environment variables."""
        return app_name or os.getenv("NETRA_APP_NAME") or os.getenv("OTEL_SERVICE_NAME") or "llm_tracing_service"

    def _get_otlp_endpoint(self) -> str | None:
        """Get OTLP endpoint from environment variables."""
        return os.getenv("NETRA_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    def _parse_headers(self, headers: Optional[str]) -> Dict[str, str] | Any:
        """Parse headers from parameter or environment variable."""
        headers = headers or os.getenv("NETRA_HEADERS")
        if isinstance(headers, str):
            return parse_env_headers(headers)
        return {}

    def _validate_api_key(self) -> None:
        """Validate that API key exists for Netra endpoints."""
        if self.otlp_endpoint and "getnetra" in self.otlp_endpoint.lower() and not self.api_key:
            print("Error: Missing Netra API key, go to netra dashboard to create one")
            print("Set the NETRA_API_KEY environment variable to the key")

    def _setup_authentication(self) -> None:
        """Setup authentication headers based on endpoint and API key."""
        if not self.api_key or not self.otlp_endpoint:
            return

        is_netra = "getnetra" in self.otlp_endpoint.lower()
        auth_key = "x-api-key" if is_netra else "Authorization"
        auth_value = self.api_key if is_netra else f"Bearer {self.api_key}"

        if not self.headers:
            self.headers = {auth_key: auth_value}
        elif auth_key not in self.headers:
            self.headers[auth_key] = auth_value

    def _get_bool_config(self, param: Optional[bool], env_var: str, default: bool) -> bool:
        """Get boolean configuration from parameter or environment variable."""
        if param is not None:
            return param

        env_value = os.getenv(env_var)
        if env_value is None:
            return default

        return env_value.lower() in ("1", "true")

    def _get_resource_attributes(self, resource_attributes: Optional[Dict[str, Any]]) -> Dict[str, Any] | Any:
        """Get resource attributes from parameter or environment variable."""
        if resource_attributes is not None:
            return resource_attributes

        env_ra = os.getenv("NETRA_RESOURCE_ATTRS")
        if not env_ra:
            return {}

        try:
            return json.loads(env_ra)
        except (json.JSONDecodeError, ValueError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse NETRA_RESOURCE_ATTRS: {e}")
            return {}

    def _set_trace_content_env(self) -> None:
        """Set TRACELOOP_TRACE_CONTENT environment variable based on trace_content."""
        os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if self.trace_content else "false"
