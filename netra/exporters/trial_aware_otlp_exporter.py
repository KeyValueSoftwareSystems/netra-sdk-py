import logging
from typing import Any, Callable, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)

from netra.exporters.utils import set_trial_blocked

logger = logging.getLogger(__name__)


class ResponseInterceptor:
    """Wraps requests.Session to intercept and monitor HTTP responses."""

    def __init__(self, session: Any, callback: Callable[[Any], None]) -> None:
        """Initialize the interceptor.

        Args:
            session: The requests.Session to wrap
            callback: Callable invoked with each response
        """
        self._session = session
        self._callback = callback
        self._original_request = session.request

        # Replace the request method with our wrapped version
        session.request = self._wrapped_request

    def _wrapped_request(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper around session.request that intercepts responses."""
        try:
            response = self._original_request(*args, **kwargs)
            try:
                self._callback(response)
            except Exception as e:
                logger.debug("Error in response callback: %s", e)
            return response
        except Exception:
            raise


class TrialAwareOTLPExporter(SpanExporter):  # type: ignore[misc]
    """Wrapper around OTLPSpanExporter that detects trial/quota blocks from backend."""

    def __init__(self, wrapped_exporter: SpanExporter) -> None:
        """Initialize with the exporter to wrap.

        Args:
            wrapped_exporter: The actual OTLPSpanExporter instance to wrap
        """
        self._exporter = wrapped_exporter
        self._interceptor: Optional[ResponseInterceptor] = None
        self._setup_response_interception()

    def _setup_response_interception(self) -> None:
        """Setup interception of HTTP responses from the OTLP exporter.

        This wraps the internal requests.Session to monitor responses.
        """
        try:
            # Access the internal requests.Session from OTLPSpanExporter
            if hasattr(self._exporter, "_session") and self._exporter._session is not None:
                session = self._exporter._session
                self._interceptor = ResponseInterceptor(session, self._on_http_response)
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
            self._check_response_for_trial_blocking(response)
        except Exception as e:
            logger.debug("Error processing response in callback: %s", e)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans and monitor response for trial status.

        Args:
            spans: Sequence of spans to export

        Returns:
            SpanExportResult from the wrapped exporter
        """
        try:
            result = self._exporter.export(spans)
            return result
        except Exception as e:
            logger.debug("Error during OTLP export: %s", e, exc_info=True)
            return SpanExportResult.FAILURE

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
