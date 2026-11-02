import logging
import threading
from typing import Any, Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)

from netra.exporters.utils import set_trial_blocked

logger = logging.getLogger(__name__)

HTTP_STATUS_PAYLOAD_TOO_LARGE = 413
# Longest excerpt of a non-JSON error body quoted in the export-failure debug message.
RESPONSE_EXCERPT_MAX_CHARS = 200
PAYLOAD_TOO_LARGE_HINT = (
    "Lower OTEL_BSP_MAX_EXPORT_BATCH_SIZE or reduce large span attributes such as prompts and completions."
)


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
        on_request_error: Callable[[], None],
    ) -> None:
        """Initialize the interceptor.

        Args:
            session: The requests.Session to wrap
            callback: Callable invoked with each response
            on_request_error: Callable invoked when a request raises instead of returning a response
                (connection error, timeout), before the exception propagates
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
        except Exception:
            self._on_request_error()
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
            self._check_response_for_trial_blocking(response)
        except Exception as e:
            logger.debug("Error processing response in callback: %s", e)

    def _on_http_request_error(self) -> None:
        """Forget the previous attempt's response when a retry gets no response at all.

        Otherwise a 503 followed by a timeout would be reported as the 503.
        """
        self._export_state.last_response = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans and monitor response for trial status.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult from the wrapped exporter
        """
        self._export_state.last_response = None
        try:
            try:
                result = self._exporter.export(spans)
            except Exception as e:
                logger.debug("Error during OTLP export: %s", e, exc_info=True)
                return SpanExportResult.FAILURE
            if result == SpanExportResult.FAILURE:
                self._log_rejected_batch(len(spans))
            return result
        finally:
            # The response references the serialized batch (response.request.body); don't keep it
            # alive on this thread until its next export.
            self._export_state.last_response = None

    def _log_rejected_batch(self, span_count: int) -> None:
        """Report the backend's final answer for a batch OTLPSpanExporter gave up on.

        Visible only in debug mode: outside it the ``netra`` logger is silenced. Logged once per batch,
        after OTLPSpanExporter's own retries (408 and 5xx), so only the last response is shown.
        """
        response = getattr(self._export_state, "last_response", None)
        status_code = getattr(response, "status_code", None)
        if not isinstance(status_code, int) or 200 <= status_code < 300:
            return  # no backend answer (connection error, timeout): OTLPSpanExporter logs those itself
        if not logger.isEnabledFor(logging.WARNING):
            return
        logger.warning("%s", _format_export_failure(response, span_count))

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
