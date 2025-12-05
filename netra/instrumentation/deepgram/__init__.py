import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.deepgram.version import __version__
from netra.instrumentation.deepgram.wrappers import (
    agent_v1_aconnect_wrapper,
    agent_v1_connect_wrapper,
    analyze_wrapper,
    generate_wrapper,
    listen_v1_aconnect_wrapper,
    listen_v1_connect_wrapper,
    listen_v2_aconnect_wrapper,
    listen_v2_connect_wrapper,
    speak_v1_aconnect_wrapper,
    speak_v1_connect_wrapper,
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

        try:
            wrap_function_wrapper(
                "deepgram.read.v1.text.client",
                "TextClient.analyze",
                analyze_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram analyze utility: {e}")

        try:
            wrap_function_wrapper(
                "deepgram.speak.v1.audio.client",
                "AudioClient.generate",
                generate_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram generate utility: {e}")

        try:
            wrap_function_wrapper(
                "deepgram.listen.v1.client",
                "V1Client.connect",
                listen_v1_connect_wrapper(tracer),
            )
            wrap_function_wrapper(
                "deepgram.listen.v1.client",
                "AsyncV1Client.connect",
                listen_v1_aconnect_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Listen v1 websocket: {e}")

        # WebSocket Listen v2
        try:
            wrap_function_wrapper(
                "deepgram.listen.v2.client",
                "V2Client.connect",
                listen_v2_connect_wrapper(tracer),
            )
            wrap_function_wrapper(
                "deepgram.listen.v2.client",
                "AsyncV2Client.connect",
                listen_v2_aconnect_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Listen v2 websocket: {e}")

        # WebSocket Speak v1
        try:
            wrap_function_wrapper(
                "deepgram.speak.v1.client",
                "V1Client.connect",
                speak_v1_connect_wrapper(tracer),
            )
            wrap_function_wrapper(
                "deepgram.speak.v1.client",
                "AsyncV1Client.connect",
                speak_v1_aconnect_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Speak v1 websocket: {e}")

        # WebSocket Agent v1
        try:
            wrap_function_wrapper(
                "deepgram.agent.v1.client",
                "V1Client.connect",
                agent_v1_connect_wrapper(tracer),
            )
            wrap_function_wrapper(
                "deepgram.agent.v1.client",
                "AsyncV1Client.connect",
                agent_v1_aconnect_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Agent v1 websocket: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Deepgram client methods.
        """
        try:
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_url")
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_file")
            unwrap("deepgram.read.v1.text.client", "TextClient.analyze")
            unwrap("deepgram.speak.v1.audio.client", "AudioClient.generate")
            unwrap("deepgram.listen.v1.client", "V1Client.connect")
            unwrap("deepgram.listen.v1.client", "AsyncV1Client.connect")
            unwrap("deepgram.listen.v2.client", "V2Client.connect")
            unwrap("deepgram.listen.v2.client", "AsyncV2Client.connect")
            unwrap("deepgram.speak.v1.client", "V1Client.connect")
            unwrap("deepgram.speak.v1.client", "AsyncV1Client.connect")
            unwrap("deepgram.agent.v1.client", "V1Client.connect")
            unwrap("deepgram.agent.v1.client", "AsyncV1Client.connect")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Deepgram transcribe utility")
