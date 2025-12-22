import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.elevenlabs.version import __version__
from netra.instrumentation.elevenlabs.wrappers import (
    create_dialogue_async_wrapper,
    create_dialogue_stream_async_wrapper,
    create_dialogue_stream_with_timestamps_async_wrapper,
    create_dialogue_stream_with_timestamps_wrapper,
    create_dialogue_stream_wrapper,
    create_dialogue_with_timestamps_async_wrapper,
    create_dialogue_with_timestamps_wrapper,
    create_dialogue_wrapper,
    create_sound_effect_async_wrapper,
    create_sound_effect_wrapper,
    create_speech_async_wrapper,
    create_speech_stream_async_wrapper,
    create_speech_stream_with_timestamp_async_wrapper,
    create_speech_stream_with_timestamp_wrapper,
    create_speech_stream_wrapper,
    create_speech_with_timestamp_async_wrapper,
    create_speech_with_timestamp_wrapper,
    create_speech_wrapper,
    create_transcript_async_wrapper,
    create_transcript_wrapper,
    voice_changer_async_wrapper,
    voice_changer_stream_async_wrapper,
    voice_changer_stream_wrapper,
    voice_changer_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("elevenlabs >= 2.15.0",)


class NetraElevenlabsInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Elevenlabs instrumentor for Netra SDK
    """

    @staticmethod
    def has_method(module: str, cls: str, method: str) -> bool:
        try:
            mod = __import__(module, fromlist=[cls])
            klass = getattr(mod, cls)
            return hasattr(klass, method)
        except Exception:
            return False

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Return the instrument dependencies for this instrumentor.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument the Elevenlabs client methods.
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize Elevenlabs tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "TextToSpeechClient.convert",
                create_speech_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "TextToSpeechClient.convert_with_timestamps",
                create_speech_with_timestamp_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "TextToSpeechClient.stream",
                create_speech_stream_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "TextToSpeechClient.stream_with_timestamps",
                create_speech_stream_with_timestamp_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs create speech utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "AsyncTextToSpeechClient.convert",
                create_speech_async_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "AsyncTextToSpeechClient.convert_with_timestamps",
                create_speech_with_timestamp_async_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "AsyncTextToSpeechClient.stream",
                create_speech_stream_async_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.text_to_speech.client",
                "AsyncTextToSpeechClient.stream_with_timestamps",
                create_speech_stream_with_timestamp_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs async create speech utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.speech_to_text.client",
                "SpeechToTextClient.convert",
                create_transcript_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs create text utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.speech_to_text.client",
                "AsyncSpeechToTextClient.convert",
                create_transcript_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs async create text utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_dialogue.client", "TextToDialogueClient.convert", create_dialogue_wrapper(tracer)
            )

            if self.has_method(
                "elevenlabs.text_to_dialogue.client",
                "TextToDialogueClient",
                "convert_with_timestamps",
            ):
                wrap_function_wrapper(
                    "elevenlabs.text_to_dialogue.client",
                    "TextToDialogueClient.convert_with_timestamps",
                    create_dialogue_with_timestamps_wrapper(tracer),
                )
            else:
                logger.warning(
                    f"Elevenlabs TextToDialogueClient.convert_with_timestamps method not found, skipping instrumentation."
                )

            wrap_function_wrapper(
                "elevenlabs.text_to_dialogue.client",
                "TextToDialogueClient.stream",
                create_dialogue_stream_wrapper(tracer),
            )

            if self.has_method(
                "elevenlabs.text_to_dialogue.client",
                "TextToDialogueClient",
                "stream_with_timestamps",
            ):
                wrap_function_wrapper(
                    "elevenlabs.text_to_dialogue.client",
                    "TextToDialogueClient.stream_with_timestamps",
                    create_dialogue_stream_with_timestamps_wrapper(tracer),
                )
            else:
                logger.warning(
                    f"Elevenlabs TextToDialogueClient.stream_with_timestamps method not found, skipping instrumentation."
                )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs create dialogue utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_dialogue.client",
                "AsyncTextToDialogueClient.convert",
                create_dialogue_async_wrapper(tracer),
            )
            if self.has_method(
                "elevenlabs.text_to_dialogue.client",
                "AsyncTextToDialogueClient",
                "convert_with_timestamps",
            ):
                wrap_function_wrapper(
                    "elevenlabs.text_to_dialogue.client",
                    "AsyncTextToDialogueClient.convert_with_timestamps",
                    create_dialogue_with_timestamps_async_wrapper(tracer),
                )
            else:
                logger.warning(
                    f"Elevenlabs TextToDialogueClient.convert_with_timestamps async method not found, skipping instrumentation."
                )

            wrap_function_wrapper(
                "elevenlabs.text_to_dialogue.client",
                "AsyncTextToDialogueClient.stream",
                create_dialogue_stream_async_wrapper(tracer),
            )

            if self.has_method(
                "elevenlabs.text_to_dialogue.client",
                "AsyncTextToDialogueClient",
                "stream_with_timestamps",
            ):
                wrap_function_wrapper(
                    "elevenlabs.text_to_dialogue.client",
                    "AsyncTextToDialogueClient.stream_with_timestamps",
                    create_dialogue_stream_with_timestamps_async_wrapper(tracer),
                )
            else:
                logger.warning(
                    f"Elevenlabs TextToDialogueClient.stream_with_timestamps method not found, skipping instrumentation."
                )

        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs async create dialogue utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.speech_to_speech.client", "SpeechToSpeechClient.convert", voice_changer_wrapper(tracer)
            )
            wrap_function_wrapper(
                "elevenlabs.speech_to_speech.client",
                "SpeechToSpeechClient.stream",
                voice_changer_stream_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs voice changer utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.speech_to_speech.client",
                "AsyncSpeechToSpeechClient.convert",
                voice_changer_async_wrapper(tracer),
            )
            wrap_function_wrapper(
                "elevenlabs.speech_to_speech.client",
                "AsyncSpeechToSpeechClient.stream",
                voice_changer_stream_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs async voice changer utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_sound_effects.client",
                "TextToSoundEffectsClient.convert",
                create_sound_effect_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs create sound effects utility, {e}")

        try:
            wrap_function_wrapper(
                "elevenlabs.text_to_sound_effects.client",
                "AsyncTextToSoundEffectsClient.convert",
                create_sound_effect_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Elevenlabs async create sound effects utility, {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Elevenlabs client methods.
        """
        try:
            unwrap("elevenlabs.text_to_speech.client", "TextToSpeechClient.convert")
            unwrap("elevenlabs.text_to_speech.client", "TextToSpeechClient.convert_with_timestamps")
            unwrap("elevenlabs.text_to_speech.client", "TextToSpeechClient.stream")
            unwrap("elevenlabs.text_to_speech.client", "TextToSpeechClient.stream_with_timestamps")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs create speech utility")

        try:
            unwrap("elevenlabs.text_to_speech.client", "AsyncTextToSpeechClient.convert")
            unwrap("elevenlabs.text_to_speech.client", "AsyncTextToSpeechClient.convert_with_timestamps")
            unwrap("elevenlabs.text_to_speech.client", "AsyncTextToSpeechClient.stream")
            unwrap("elevenlabs.text_to_speech.client", "AsyncTextToSpeechClient.stream_with_timestamps")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async create speech utility")

        try:
            unwrap("elevenlabs.speech_to_text.client", "SpeechToTextClient.convert")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs transcribe utility")

        try:
            unwrap("elevenlabs.speech_to_text.client", "AsyncSpeechToTextClient.convert")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async transcribe utility")

        try:
            unwrap("elevenlabs.text_to_dialogue.client", "TextToDialogueClient.convert")
            unwrap("elevenlabs.text_to_dialogue.client", "TextToDialogueClient.stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs create dialogue utility")

        try:
            unwrap("elevenlabs.text_to_dialogue.client", "TextToDialogueClient.convert_with_timestamps")
            unwrap("elevenlabs.text_to_dialogue.client", "TextToDialogueClient.stream_with_timestamps")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs create dialogue with timestamps utility")

        try:
            unwrap("elevenlabs.text_to_dialogue.client", "AsyncTextToDialogueClient.convert")
            unwrap("elevenlabs.text_to_dialogue.client", "AsyncTextToDialogueClient.stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async create dialogue utility")

        try:
            unwrap("elevenlabs.text_to_dialogue.client", "AsyncTextToDialogueClient.convert_with_timestamps")
            unwrap("elevenlabs.text_to_dialogue.client", "AsyncTextToDialogueClient.stream_with_timestamps")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async create dialogue with timestamps utility")

        try:
            unwrap("elevenlabs.speech_to_speech.client", "SpeechToSpeechClient.convert")
            unwrap("elevenlabs.speech_to_speech.client", "SpeechToSpeechClient.stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs voice changer utility")

        try:
            unwrap("elevenlabs.speech_to_speech.client", "AsyncSpeechToSpeechClient.convert")
            unwrap("elevenlabs.speech_to_speech.client", "AsyncSpeechToSpeechClient.stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async voice changer utility")

        try:
            unwrap("elevenlabs.text_to_sound_effects.client", "TextToSoundEffectsClient.convert")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs create sound effect utility")

        try:
            unwrap("elevenlabs.text_to_sound_effects.client", "AsyncTextToSoundEffectsClient.convert")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Elevenlabs async create sound effect utility")
