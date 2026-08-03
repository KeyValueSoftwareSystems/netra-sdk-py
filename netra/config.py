import json
import logging
import os
from typing import Any, Dict, FrozenSet, List, Optional

from opentelemetry.util.re import parse_env_headers

from netra.version import __version__

logger = logging.getLogger(__name__)

# Fallback limits used when no Config has been activated yet (e.g. code paths that
# run before ``Netra.init()``, or tests that never call it). Once ``init()`` runs,
# the active Config instance's values (resolved from env at init time) take over.
_DEFAULT_ATTRIBUTE_MAX_LEN = 50000
_DEFAULT_CONVERSATION_CONTENT_MAX_LEN = 50000
_DEFAULT_TRIAL_BLOCK_DURATION_SECONDS = 15 * 60

# --- Voice-agent audio capture (env-only; no Netra.init() parameter) ------------
# The path segment appended to the OTLP endpoint when no explicit audio endpoint
# is given. Deliberately not derived via UsageHttpClient._resolve_base_url, which
# strips a "/telemetry" suffix — correct for the REST APIs, wrong here.
_AUDIO_CHUNK_PATH = "/v1/audio/chunk"

# Header names that count as an audio-ingest credential. An unauthenticated PCM
# POST is never attempted.
_AUDIO_AUTH_HEADERS = ("x-api-key", "Authorization")

# The only recognised speaker roles.
AUDIO_ROLES: FrozenSet[str] = frozenset({"user", "agent"})

_DEFAULT_AUDIO_BATCH_BYTES = 32768
_DEFAULT_AUDIO_BATCH_INTERVAL_MS = 1000
_DEFAULT_AUDIO_BUFFER_BYTES = 2097152
_DEFAULT_AUDIO_MAX_REQUEST_BYTES = 262144

_MIN_AUDIO_BATCH_BYTES = 1024
_MIN_AUDIO_BATCH_INTERVAL_MS = 100
_MAX_AUDIO_BATCH_INTERVAL_MS = 30000


class Config:
    """
    Holds configuration options for the tracer.
    """

    # SDK Constants
    SDK_NAME = "netra"
    LIBRARY_NAME = "netra"
    LIBRARY_VERSION = __version__

    # Root-span attribute marking traces produced by evaluation/simulation runs
    # so the FE/BE can distinguish them from normal workflow invocations.
    TRACE_ORIGIN_KEY = "netra.trace.origin"
    TRACE_ORIGIN_EVALUATION = "evaluation"

    def __init__(
        self,
        app_name: Optional[str] = None,
        headers: Optional[str | Dict[str, str]] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        enable_scrubbing: Optional[bool] = None,
        blocked_spans: Optional[List[str]] = None,
        enable_metrics: Optional[bool] = None,
        metrics_export_interval_ms: Optional[int] = None,
        export_auto_metrics: Optional[bool] = None,
    ):
        """
        Initialize the configuration.

        Args:
            app_name: Logical name for this service
            headers: Additional headers (W3C Correlation-Context format)
            disable_batch: Whether to disable batch span processor
            trace_content: Whether to capture prompt/completion content
            debug_mode: Whether to enable SDK logging (default: False)
            resource_attributes: Custom resource attributes dict (e.g., {'env': 'prod', 'version': '1.0.0'})
            enable_scrubbing: Whether to enable pydantic logfire scrubbing (default: False)
            blocked_spans: List of span names (prefix/suffix patterns) to block from export
            enable_metrics: Whether to enable custom metrics export via OTLP (default: False)
            metrics_export_interval_ms: How often to push metrics to the collector in ms (default: 60000)
            export_auto_metrics: Whether to export OTel auto-instrumented system metrics (default: False)
        """
        self.app_name = self._get_app_name(app_name)
        self.otlp_endpoint = self._get_otlp_endpoint()
        self.api_key = os.getenv("NETRA_API_KEY")
        self.headers = self._parse_headers(headers)

        self._validate_api_key()
        self._setup_authentication()

        self.disable_batch = self._get_bool_config(disable_batch, "NETRA_DISABLE_BATCH", default=False)
        self.trace_content = self._get_bool_config(trace_content, "NETRA_TRACE_CONTENT", default=True)
        self.debug_mode = self._get_bool_config(debug_mode, "NETRA_DEBUG", default=False)
        self.enable_scrubbing = self._get_bool_config(enable_scrubbing, "NETRA_ENABLE_SCRUBBING", default=False)
        self.enable_metrics = self._get_bool_config(enable_metrics, "NETRA_ENABLE_METRICS", default=False)
        self.export_auto_metrics = self._get_bool_config(
            export_auto_metrics, "NETRA_EXPORT_AUTO_METRICS", default=False
        )

        self.environment = environment or os.getenv("NETRA_ENV", "default")
        self.resource_attributes = self._get_resource_attributes(resource_attributes)
        self.blocked_spans = blocked_spans
        self.metrics_export_interval_ms = self._get_int_config(
            metrics_export_interval_ms, "NETRA_METRICS_EXPORT_INTERVAL", default=60000
        )

        # Resolved at init time (env-only) so overrides applied before ``Netra.init()``
        # — including a late ``load_dotenv()`` — are honored. Previously these were
        # class attributes read at import time, which ignored any post-import env change.
        self.attribute_max_len = self._get_int_config(
            None, "NETRA_ATTRIBUTE_MAX_LEN", default=_DEFAULT_ATTRIBUTE_MAX_LEN
        )
        self.conversation_max_len = self._get_int_config(
            None, "NETRA_CONVERSATION_CONTENT_MAX_LEN", default=_DEFAULT_CONVERSATION_CONTENT_MAX_LEN
        )
        self.trial_block_duration_seconds = self._get_int_config(
            None, "TRIAL_BLOCK_DURATION_SECONDS", default=_DEFAULT_TRIAL_BLOCK_DURATION_SECONDS
        )

        self._resolve_audio_settings()

        self._set_trace_content_env()

    def _resolve_audio_settings(self) -> None:
        """Resolve and validate the voice-agent audio-capture settings.

        Env-only by design: there is no ``capture_audio`` parameter on
        ``Netra.init()``.  Whether audio is captured at all is decided by
        :attr:`audio_capture_enabled`, not by a flag.

        Every validation failure logs a ``WARNING`` naming the setting and the
        value actually used, then falls back to a safe value.  A bad number MUST
        NOT raise out of ``Netra.init()``.
        """
        self.audio_endpoint_override = os.getenv("NETRA_AUDIO_ENDPOINT")
        self.audio_batch_bytes = self._get_int_config(
            None, "NETRA_AUDIO_BATCH_BYTES", default=_DEFAULT_AUDIO_BATCH_BYTES
        )
        self.audio_batch_interval_ms = self._get_int_config(
            None, "NETRA_AUDIO_BATCH_INTERVAL_MS", default=_DEFAULT_AUDIO_BATCH_INTERVAL_MS
        )
        self.audio_buffer_bytes = self._get_int_config(
            None, "NETRA_AUDIO_BUFFER_BYTES", default=_DEFAULT_AUDIO_BUFFER_BYTES
        )
        self.audio_max_request_bytes = self._get_int_config(
            None, "NETRA_AUDIO_MAX_REQUEST_BYTES", default=_DEFAULT_AUDIO_MAX_REQUEST_BYTES
        )
        self.audio_roles = self._get_role_set("NETRA_AUDIO_ROLES")

        # Order matters: audio_batch_bytes is clamped against the resolved
        # max-request size first, then the two ceilings are raised to whatever
        # batch size survived. Doing it the other way round lets a tiny
        # max_request_bytes silently shrink the batch below its floor.
        if self.audio_max_request_bytes < _MIN_AUDIO_BATCH_BYTES:
            logger.warning(
                "netra.audio: NETRA_AUDIO_MAX_REQUEST_BYTES=%d is below the minimum batch size; using %d",
                self.audio_max_request_bytes,
                _MIN_AUDIO_BATCH_BYTES,
            )
            self.audio_max_request_bytes = _MIN_AUDIO_BATCH_BYTES

        clamped_batch = min(max(self.audio_batch_bytes, _MIN_AUDIO_BATCH_BYTES), self.audio_max_request_bytes)
        if clamped_batch != self.audio_batch_bytes:
            logger.warning(
                "netra.audio: NETRA_AUDIO_BATCH_BYTES=%d out of range [%d, %d]; using %d",
                self.audio_batch_bytes,
                _MIN_AUDIO_BATCH_BYTES,
                self.audio_max_request_bytes,
                clamped_batch,
            )
            self.audio_batch_bytes = clamped_batch

        clamped_interval = min(
            max(self.audio_batch_interval_ms, _MIN_AUDIO_BATCH_INTERVAL_MS),
            _MAX_AUDIO_BATCH_INTERVAL_MS,
        )
        if clamped_interval != self.audio_batch_interval_ms:
            logger.warning(
                "netra.audio: NETRA_AUDIO_BATCH_INTERVAL_MS=%d out of range [%d, %d]; using %d",
                self.audio_batch_interval_ms,
                _MIN_AUDIO_BATCH_INTERVAL_MS,
                _MAX_AUDIO_BATCH_INTERVAL_MS,
                clamped_interval,
            )
            self.audio_batch_interval_ms = clamped_interval

        if self.audio_buffer_bytes < self.audio_batch_bytes:
            logger.warning(
                "netra.audio: NETRA_AUDIO_BUFFER_BYTES=%d is below the batch size; using %d",
                self.audio_buffer_bytes,
                self.audio_batch_bytes,
            )
            self.audio_buffer_bytes = self.audio_batch_bytes

        if self.audio_max_request_bytes < self.audio_batch_bytes:
            logger.warning(
                "netra.audio: NETRA_AUDIO_MAX_REQUEST_BYTES=%d is below the batch size; using %d",
                self.audio_max_request_bytes,
                self.audio_batch_bytes,
            )
            self.audio_max_request_bytes = self.audio_batch_bytes

        if not self.audio_roles:
            logger.warning(
                "netra.audio: NETRA_AUDIO_ROLES resolved empty; no call audio will be captured. "
                "Traces are unaffected."
            )

    def _get_role_set(self, env_var: str) -> FrozenSet[str]:
        """Parse a comma-separated speaker-role list, dropping unknown roles.

        An explicitly empty value (``NETRA_AUDIO_ROLES=``) is the documented way
        to disable audio capture without affecting traces, so it resolves to an
        empty set rather than the default.

        Args:
            env_var: Name of the environment variable holding the role list.

        Returns:
            The recognised roles, or the full default set when *env_var* is unset.
        """
        raw = os.getenv(env_var)
        if raw is None:
            return AUDIO_ROLES

        requested = {part.strip().lower() for part in raw.split(",") if part.strip()}
        unknown = requested - AUDIO_ROLES
        if unknown:
            logger.warning(
                "netra.audio: %s contains unknown role(s) %s; recognised roles are %s",
                env_var,
                sorted(unknown),
                sorted(AUDIO_ROLES),
            )
        return frozenset(requested & AUDIO_ROLES)

    def audio_endpoint(self) -> Optional[str]:
        """Resolve the audio ingest URL, or None if audio must not be sent.

        This is the ONLY gate on audio capture: there is no ``capture_audio``
        flag.  A non-None return means audio WILL be captured and streamed once a
        LiveKit session starts.  Returns None unless a concrete endpoint resolves
        AND an auth header is present.

        Callers treat None as "disable capture entirely", not "retry later" — the
        result is resolved from init-time state and does not change during the
        process.

        Returns:
            The absolute audio ingest URL, or ``None`` when audio must not be sent.
        """
        if self.audio_endpoint_override:
            url = self.audio_endpoint_override
        elif self.otlp_endpoint:
            url = self.otlp_endpoint.rstrip("/") + _AUDIO_CHUNK_PATH
        else:
            return None

        if not any(header in self.headers for header in _AUDIO_AUTH_HEADERS):
            logger.warning(
                "netra.audio: an audio endpoint resolved but no credential is configured; "
                "audio capture is disabled. Set NETRA_API_KEY or pass an auth header."
            )
            return None

        return url

    @property
    def audio_capture_enabled(self) -> bool:
        """Whether call audio will be captured and streamed.

        The single derived predicate behind audio capture, so the instrumentor,
        the session hooks and the startup log line cannot disagree about it.

        Returns:
            True when an audio endpoint resolves and at least one speaker role is
            enabled; False otherwise, meaning no audio is captured or streamed.
        """
        return self.audio_endpoint() is not None and bool(self.audio_roles)

    def _get_app_name(self, app_name: Optional[str]) -> str:
        """Get application name from param or environment variables."""
        return app_name or os.getenv("NETRA_APP_NAME") or os.getenv("OTEL_SERVICE_NAME") or "llm_tracing_service"

    def _get_otlp_endpoint(self) -> str | None:
        """Get OTLP endpoint from environment variables."""
        return os.getenv("NETRA_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    def _parse_headers(self, headers: Optional[str] | Dict[str, str]) -> Dict[str, str] | Any:
        """Parse headers from parameter or environment variable."""
        headers = headers or os.getenv("NETRA_HEADERS")
        if isinstance(headers, str):
            return parse_env_headers(headers)
        elif isinstance(headers, dict):
            return headers
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

    def _get_int_config(self, param: Optional[int], env_var: str, default: int) -> int:
        """Get integer configuration from parameter or environment variable."""
        if param is not None:
            return param

        env_value = os.getenv(env_var)
        if env_value is None:
            return default

        try:
            return int(env_value)
        except ValueError:
            return default

    def _set_trace_content_env(self) -> None:
        """Set TRACELOOP_TRACE_CONTENT environment variable based on trace_content."""
        os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if self.trace_content else "false"


# The process-active Config instance, set by ``Netra.init()``. Global/static consumers
# (span processors, the SessionManager classmethods, module-level exporter helpers) read
# their limits off this instance rather than import-time class attributes, so limits
# resolve at init time. ``None`` until ``init()`` runs; getters fall back to the defaults.
_active_config: Optional["Config"] = None


def set_active_config(config: "Config") -> None:
    """Register *config* as the process-active configuration.

    Called once by ``Netra.init()``. Subsequent calls replace the reference,
    matching the single-init singleton semantics of ``Netra``.
    """
    global _active_config
    _active_config = config


def get_active_config() -> Optional["Config"]:
    """Return the process-active Config, or ``None`` if ``Netra.init()`` has not run."""
    return _active_config


def get_attribute_max_len() -> int:
    """Max length for a single span attribute value, from the active config or default."""
    cfg = get_active_config()
    return cfg.attribute_max_len if cfg is not None else _DEFAULT_ATTRIBUTE_MAX_LEN


def get_conversation_max_len() -> int:
    """Max length for a single conversation entry's content, from the active config or default."""
    cfg = get_active_config()
    return cfg.conversation_max_len if cfg is not None else _DEFAULT_CONVERSATION_CONTENT_MAX_LEN


def get_trial_block_duration_seconds() -> int:
    """Trial/quota export-block duration in seconds, from the active config or default."""
    cfg = get_active_config()
    return cfg.trial_block_duration_seconds if cfg is not None else _DEFAULT_TRIAL_BLOCK_DURATION_SECONDS
