import logging
import threading
from typing import Any, Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)

from netra.exporters.utils import set_trial_blocked
from netra.netra_logger import get_logger

logger = get_logger(__name__)

HTTP_STATUS_PAYLOAD_TOO_LARGE = 413
# Longest excerpt of a non-JSON error body quoted in the export-failure debug message.
RESPONSE_EXCERPT_MAX_CHARS = 200
PAYLOAD_TOO_LARGE_HINT = (
    "Lower OTEL_BSP_MAX_EXPORT_BATCH_SIZE or reduce large span attributes such as prompts and completions."
)
# Module-level logger of opentelemetry-exporter-otlp-proto-http's OTLPSpanExporter.
OTEL_TRACE_EXPORTER_LOGGER_NAME = "opentelemetry.exporter.otlp.proto.http.trace_exporter"

# Set while the current thread is inside TrialAwareOTLPExporter.export(). OTLPSpanExporter exports
# synchronously on the calling thread, so its log records for our batches are emitted under this flag.
_netra_export_scope = threading.local()


class _SuppressOtelErrorsDuringNetraExport(logging.Filter):
    """Drop OTLPSpanExporter's ERROR records for batches Netra exports; Netra's own notice reports them.

    Without this, each failed batch prints twice: OpenTelemetry's "Failed to export span batch ..." and
    Netra's notice. Only records emitted inside a Netra export are dropped, so failures of an exporter the
    application runs itself stay visible. Retry WARNINGs are kept.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for an ERROR (or above) record emitted inside a Netra export."""
        return record.levelno < logging.ERROR or not getattr(_netra_export_scope, "active", False)


_SUPPRESS_OTEL_ERRORS_FILTER = _SuppressOtelErrorsDuringNetraExport()


def _install_otel_error_filter() -> None:
    """Attach the filter to the OpenTelemetry exporter logger once per process."""
    otel_logger = logging.getLogger(OTEL_TRACE_EXPORTER_LOGGER_NAME)
    if _SUPPRESS_OTEL_ERRORS_FILTER not in otel_logger.filters:
        otel_logger.addFilter(_SUPPRESS_OTEL_ERRORS_FILTER)


def _describe_server_message(response: Any) -> str:
    """Return what the backend said, quoted.

    Uses ``error.message`` from the backend's error envelope (joined when it is a list of validation
    messages), else an excerpt of the raw body.
    """
    try:
        body = response.json()
    except ValueError:  # requests' JSONDecodeError subclasses ValueError
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str):
            return f'"{message}"'
        if isinstance(message, list) and all(isinstance(item, str) for item in message):
            return '"' + "; ".join(message) + '"'
    excerpt = (getattr(response, "text", "") or "").strip()[:RESPONSE_EXCERPT_MAX_CHARS]
    return f'"{excerpt}"' if excerpt else "(empty response body)"


def _format_unanswered_export(error: BaseException, span_count: int) -> str:
    """Message for a batch that got no HTTP response (connection refused, DNS, TLS, timeout)."""
    return f"netra: Span export failed: {error}. Dropped a batch of {span_count} spans."


def _format_export_failure(response: Any, span_count: int) -> str:
    """Debug message for a batch the backend rejected.

    netra-sdk-js's TrialAwareOTLPExporter uses the same format, so both SDKs report export failures
    identically.
    """
    status_code = getattr(response, "status_code", None)
    status = " ".join(str(part) for part in (status_code or "unknown", getattr(response, "reason", None)) if part)
    request_body = getattr(getattr(response, "request", None), "body", None)
    request_bytes = len(request_body) if isinstance(request_body, (bytes, str)) else "unknown"
    hint = f" {PAYLOAD_TOO_LARGE_HINT}" if status_code == HTTP_STATUS_PAYLOAD_TOO_LARGE else ""
    return (
        f"Span export failed with HTTP {status}: {_describe_server_message(response)}. "
        f"Dropped a batch of {span_count} spans ({request_bytes} bytes).{hint}"
    )


class ResponseInterceptor:
    """Wraps requests.Session to intercept and monitor HTTP responses."""

    def __init__(
        self,
        session: Any,
        callback: Callable[[Any], None],
        on_request_error: Callable[[Exception], None],
    ) -> None:
        """Initialize the interceptor.

        Args:
            session: The requests.Session to wrap
            callback: Callable invoked with each response
            on_request_error: Callable invoked with the exception when a request raises instead of
                returning a response (connection error, timeout), before the exception propagates
        """
        self._session = session
        self._callback = callback
        self._on_request_error = on_request_error
        self._original_request = session.request

        # Replace the request method with our wrapped version
        session.request = self._wrapped_request

    def _wrapped_request(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper around session.request that intercepts responses."""
        try:
            response = self._original_request(*args, **kwargs)
        except Exception as error:
            self._on_request_error(error)
            raise
        try:
            self._callback(response)
        except Exception as e:
            logger.debug("Error in response callback: %s", e)
        return response


class TrialAwareOTLPExporter(SpanExporter):  # type: ignore[misc]
    """Wrapper around OTLPSpanExporter that detects trial/quota blocks from backend."""

    def __init__(self, wrapped_exporter: SpanExporter) -> None:
        """Initialize with the exporter to wrap.

        Args:
            wrapped_exporter: The actual OTLPSpanExporter instance to wrap
        """
        self._exporter = wrapped_exporter
        self._interceptor: Optional[ResponseInterceptor] = None
        # OTLPSpanExporter sends synchronously inside export(), on the calling thread, so a
        # thread-local carries the last HTTP response back to export() without racing other exports.
        self._export_state = threading.local()
        self._setup_response_interception()
        _install_otel_error_filter()

    def _setup_response_interception(self) -> None:
        """Setup interception of HTTP responses from the OTLP exporter.

        This wraps the internal requests.Session to monitor responses.
        """
        try:
            # Access the internal requests.Session from OTLPSpanExporter
            if hasattr(self._exporter, "_session") and self._exporter._session is not None:
                session = self._exporter._session
                self._interceptor = ResponseInterceptor(session, self._on_http_response, self._on_http_request_error)
                logger.debug("Successfully setup HTTP response interception")
            else:
                logger.debug("Could not find _session on OTLPSpanExporter")
        except Exception as e:
            logger.debug("Failed to setup response interception: %s", e)

    def _on_http_response(self, response: Any) -> None:
        """Callback invoked when HTTP response is received.

        Args:
            response: The requests.Response object
        """
        try:
            self._export_state.last_response = response
            self._export_state.last_request_error = None
            self._check_response_for_trial_blocking(response)
        except Exception as e:
            logger.debug("Error processing response in callback: %s", e)

    def _on_http_request_error(self, error: Exception) -> None:
        """Remember why an attempt got no response, and forget the previous attempt's response.

        Otherwise a 503 followed by a timeout would be reported as the 503.

        Args:
            error: The exception the HTTP request raised
        """
        self._export_state.last_response = None
        self._export_state.last_request_error = error

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans and monitor response for trial status.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult from the wrapped exporter
        """
        self._export_state.last_response = None
        self._export_state.last_request_error = None
        _netra_export_scope.active = True
        try:
            try:
                result = self._exporter.export(spans)
            except Exception as e:
                # No backend answer (connection refused, DNS, TLS, timeout). Older OTLPSpanExporter
                # versions (e.g. 1.38) let the request error propagate instead of logging it.
                logger.notice(_format_unanswered_export(e, len(spans)))
                logger.debug("Error during OTLP export: %s", e, exc_info=True)
                return SpanExportResult.FAILURE
            if result == SpanExportResult.FAILURE:
                self._log_failed_batch(len(spans))
            return result
        finally:
            _netra_export_scope.active = False
            # The response references the serialized batch (response.request.body); don't keep it
            # alive on this thread until its next export.
            self._export_state.last_response = None
            self._export_state.last_request_error = None

    def _log_failed_batch(self, span_count: int) -> None:
        """Report why OTLPSpanExporter gave up on a batch: the backend's final answer, or the request error.

        Logged once per batch, after OTLPSpanExporter's own retries (408, 5xx and, on newer versions,
        connection errors), so only the last attempt is shown. Always visible, whether or not debug_mode
        is on. OTLPSpanExporter's own ERROR for the batch is filtered out, so this is the only line.
        """
        response = getattr(self._export_state, "last_response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and not 200 <= status_code < 300:
            logger.notice("netra: %s", _format_export_failure(response, span_count))
            return
        request_error = getattr(self._export_state, "last_request_error", None)
        if request_error is not None:
            # Newer OTLPSpanExporter versions (e.g. 1.41) catch request errors and return FAILURE.
            logger.notice(_format_unanswered_export(request_error, span_count))
        # Neither: no request was sent (exporter already shut down); OTLPSpanExporter's WARNING says so.

    def _check_response_for_trial_blocking(self, response: Any) -> None:
        """Check HTTP response for trial blocking indicators.

        Args:
            response: The HTTP response object (requests.Response)
        """
        try:
            try:
                body = response.json()
            except Exception:
                body = None
            if (error := body.get("error")) and (inner := error.get("error")):
                error_code = inner.get("code")
                blocked = True if error_code == "QUOTA_EXCEEDED" else False
                if blocked:
                    logger.warning("Quota Exceeded: %s", error_code)
                    set_trial_blocked(True)
                    return
        except Exception as e:
            logger.debug("Error checking response for trial blocking: %s", e)

    def shutdown(self) -> None:
        """Shutdown the wrapped exporter."""
        try:
            self._exporter.shutdown()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        """Force flush the wrapped exporter."""
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            return True
