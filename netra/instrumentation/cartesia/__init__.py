import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.cartesia.version import __version__
from netra.instrumentation.cartesia.wrappers import (
    stt_wrapper,
    stt_ws_wrapper,
    tts_bytes_wrapper,
    tts_ws_wrapper,
    voice_changer_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("cartesia >= 2.0.17",)


class NetraCartesiaInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Cartesia instrumentor for Netra SDK:
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Return the instrument dependencies for this instrumentor.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument the Cartesia client methods.
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize Cartesia tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "cartesia.tts.client",
                "TtsClient.bytes",
                tts_bytes_wrapper(tracer),
            )
            wrap_function_wrapper(
                "cartesia.tts.client",
                "TtsClient.sse",
                tts_bytes_wrapper(tracer),
            )
            wrap_function_wrapper(
                "cartesia.tts.socket_client", "TtsClientWithWebsocket.websocket", tts_ws_wrapper(tracer)
            )
        except Exception as e:
            logger.error(f"Failed to instrument Cartesia tts utility: {e}")

        try:
            wrap_function_wrapper(
                "cartesia.stt.client",
                "SttClient.transcribe",
                stt_wrapper(tracer),
            )
            wrap_function_wrapper(
                "cartesia.stt.socket_client", "SttClientWithWebsocket.websocket", stt_ws_wrapper(tracer)
            )
        except Exception as e:
            logger.error(f"Failed to instrument Cartesia sst utility: {e}")

        try:
            wrap_function_wrapper(
                "cartesia.voice_changer.client",
                "VoiceChangerClient.bytes",
                voice_changer_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Cartesia voice changer utility: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Cartesia client methods.
        """
        try:
            unwrap("cartesia.tts.client", "TtsClient.bytes")
            unwrap("cartesia.tts.client", "TtsClient.sse")
            unwrap("cartesia.tts.client", "TtsClientWithWebsocket.websocket")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Cartesia tts utility")

        try:
            unwrap("cartesia.stt.client", "SttClient.transcribe")
            unwrap("cartesia.stt.client", "SttClientWithWebsocket.websocket")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Cartesia stt utility")

        try:
            unwrap("cartesia.voice_changer.client", "VoiceChangerClient.bytes")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Cartesia voice changer utility")
