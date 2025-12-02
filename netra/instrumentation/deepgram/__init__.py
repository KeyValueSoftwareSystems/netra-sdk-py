import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.deepgram.version import __version__
from netra.instrumentation.deepgram.wrappers import (
    transcribe_file_wrapper,
    transcribe_url_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("deepgram-sdk >= 5.0.0",)


class NetraDeepgramInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Deepgram instrumentor for Netra SDK:
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Return the instrument dependencies for this instrumentor.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument the Deepgram client methods.
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize Deepgram tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "MediaClient.transcribe_url",
                transcribe_url_wrapper(tracer),
            )
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "MediaClient.transcribe_file",
                transcribe_file_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram transcribe utility: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Deepgram client methods.
        """
        try:
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_url")
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_file")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Deepgram transcribe utility")
