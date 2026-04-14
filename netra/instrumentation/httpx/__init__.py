import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.httpx.utils import get_default_span_name
from netra.instrumentation.httpx.version import __version__
from netra.instrumentation.httpx.wrappers import async_send_wrapper, send_wrapper

logger = logging.getLogger(__name__)

_instruments = ("httpx >= 0.18.0",)


class HTTPXInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom HTTPX instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the list of required instrumentation dependencies."""
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument httpx.Client.send and httpx.AsyncClient.send.

        Args:
            **kwargs: Keyword arguments passed by the instrumentation framework.
                tracer_provider: Optional TracerProvider to use for creating spans.
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_function_wrapper("httpx", "Client.send", send_wrapper(tracer))
            wrap_function_wrapper("httpx", "AsyncClient.send", async_send_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument httpx: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Uninstrument httpx.Client.send and httpx.AsyncClient.send.

        Args:
            **kwargs: Keyword arguments passed by the instrumentation framework.
        """
        try:
            import httpx

            unwrap(httpx.Client, "send")
            unwrap(httpx.AsyncClient, "send")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument httpx")
