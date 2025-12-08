import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.deepgram.version import __version__
from netra.instrumentation.deepgram.wrappers import (
    AGENT_V1_CONNECT_SPAN_NAME,
    ANALYZE_SPAN_NAME,
    GENERATE_SPAN_NAME,
    LISTEN_V1_CONNECT_SPAN_NAME,
    LISTEN_V2_CONNECT_SPAN_NAME,
    SPEAK_V1_CONNECT_SPAN_NAME,
    TRANSCRIBE_FILE_SPAN_NAME,
    TRANSCRIBE_URL_SPAN_NAME,
    wrap_async,
    wrap_async_generator,
    wrap_connect,
    wrap_connect_async,
    wrap_sync,
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

        # Sync MediaClient
        try:
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "MediaClient.transcribe_url",
                wrap_sync(tracer, TRANSCRIBE_URL_SPAN_NAME, "url"),
            )
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "MediaClient.transcribe_file",
                wrap_sync(tracer, TRANSCRIBE_FILE_SPAN_NAME, "file"),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram transcribe utility: {e}")

        # Async MediaClient
        try:
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "AsyncMediaClient.transcribe_url",
                wrap_async(tracer, TRANSCRIBE_URL_SPAN_NAME, "url"),
            )
            wrap_function_wrapper(
                "deepgram.listen.v1.media.client",
                "AsyncMediaClient.transcribe_file",
                wrap_async(tracer, TRANSCRIBE_FILE_SPAN_NAME, "file"),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram async transcribe utility: {e}")

        # Sync TextClient
        try:
            wrap_function_wrapper(
                "deepgram.read.v1.text.client",
                "TextClient.analyze",
                wrap_sync(tracer, ANALYZE_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram analyze utility: {e}")

        # Async TextClient
        try:
            wrap_function_wrapper(
                "deepgram.read.v1.text.client",
                "AsyncTextClient.analyze",
                wrap_async(tracer, ANALYZE_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram async analyze utility: {e}")

        # Sync AudioClient
        try:
            wrap_function_wrapper(
                "deepgram.speak.v1.audio.client",
                "AudioClient.generate",
                wrap_sync(tracer, GENERATE_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram generate utility: {e}")

        # Async AudioClient
        try:
            wrap_function_wrapper(
                "deepgram.speak.v1.audio.client",
                "AsyncAudioClient.generate",
                wrap_async_generator(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram async generate utility: {e}")

        # Websocket Listen v1
        try:
            wrap_function_wrapper(
                "deepgram.listen.v1.client",
                "V1Client.connect",
                wrap_connect(tracer, LISTEN_V1_CONNECT_SPAN_NAME),
            )
            wrap_function_wrapper(
                "deepgram.listen.v1.client",
                "AsyncV1Client.connect",
                wrap_connect_async(tracer, LISTEN_V1_CONNECT_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Listen v1 websocket: {e}")

        # WebSocket Listen v2
        try:
            wrap_function_wrapper(
                "deepgram.listen.v2.client",
                "V2Client.connect",
                wrap_connect(tracer, LISTEN_V2_CONNECT_SPAN_NAME),
            )
            wrap_function_wrapper(
                "deepgram.listen.v2.client",
                "AsyncV2Client.connect",
                wrap_connect_async(tracer, LISTEN_V2_CONNECT_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Listen v2 websocket: {e}")

        # WebSocket Speak v1
        try:
            wrap_function_wrapper(
                "deepgram.speak.v1.client",
                "V1Client.connect",
                wrap_connect(tracer, SPEAK_V1_CONNECT_SPAN_NAME),
            )
            wrap_function_wrapper(
                "deepgram.speak.v1.client",
                "AsyncV1Client.connect",
                wrap_connect_async(tracer, SPEAK_V1_CONNECT_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Speak v1 websocket: {e}")

        # WebSocket Agent v1
        try:
            wrap_function_wrapper(
                "deepgram.agent.v1.client",
                "V1Client.connect",
                wrap_connect(tracer, AGENT_V1_CONNECT_SPAN_NAME),
            )
            wrap_function_wrapper(
                "deepgram.agent.v1.client",
                "AsyncV1Client.connect",
                wrap_connect_async(tracer, AGENT_V1_CONNECT_SPAN_NAME),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Deepgram Agent v1 websocket: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Deepgram client methods.
        """
        try:
            # Sync clients
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_url")
            unwrap("deepgram.listen.v1.media.client", "MediaClient.transcribe_file")
            unwrap("deepgram.read.v1.text.client", "TextClient.analyze")
            unwrap("deepgram.speak.v1.audio.client", "AudioClient.generate")
            # Async clients
            unwrap("deepgram.listen.v1.media.async_client", "AsyncMediaClient.transcribe_url")
            unwrap("deepgram.listen.v1.media.async_client", "AsyncMediaClient.transcribe_file")
            unwrap("deepgram.read.v1.text.async_client", "AsyncTextClient.analyze")
            unwrap("deepgram.speak.v1.audio.async_client", "AsyncAudioClient.generate")
            # WebSocket clients
            unwrap("deepgram.listen.v1.client", "V1Client.connect")
            unwrap("deepgram.listen.v1.client", "AsyncV1Client.connect")
            unwrap("deepgram.listen.v2.client", "V2Client.connect")
            unwrap("deepgram.listen.v2.client", "AsyncV2Client.connect")
            unwrap("deepgram.speak.v1.client", "V1Client.connect")
            unwrap("deepgram.speak.v1.client", "AsyncV1Client.connect")
            unwrap("deepgram.agent.v1.client", "V1Client.connect")
            unwrap("deepgram.agent.v1.client", "AsyncV1Client.connect")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Deepgram utility")
