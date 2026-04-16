import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.requests.version import __version__
from netra.instrumentation.requests.wrappers import send_wrapper

logger = logging.getLogger(__name__)

_instruments = ("requests >= 2.0.0",)


class RequestsInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom requests instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the list of required instrumentation dependencies.

        Returns:
            A collection of package requirement strings that must be satisfied
            for this instrumentor to function.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument requests.Session.send.

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
            wrap_function_wrapper("requests", "Session.send", send_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument requests: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Uninstrument requests.Session.send.

        Args:
            **kwargs: Keyword arguments passed by the instrumentation framework
                (unused but required by the base class interface).
        """
        try:
            unwrap("requests.Session", "send")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument requests")
