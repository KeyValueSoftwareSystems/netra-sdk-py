"""OpenTelemetry Google GenAI API instrumentation"""

import logging
import os
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.google_genai.version import __version__
from netra.instrumentation.google_genai.wrappers import (
    acontent_stream_wrapper,
    acontent_wrapper,
    aimages_wrapper,
    avideos_wrapper,
    content_stream_wrapper,
    content_wrapper,
    images_wrapper,
    videos_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("google-genai >= 0.1.0",)


class NetraGoogleGenAiInstrumentor(BaseInstrumentor):  # type: ignore
    """Custom Google GenAI instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument Google GenAI client methods"""

        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_content",
                content_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_content",
                acontent_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_content: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_content_stream",
                content_stream_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_content_stream",
                acontent_stream_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_content_stream: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_images",
                images_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_images",
                aimages_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_images: {e}")

        try:
            wrap_function_wrapper(
                "google.genai.models",
                "Models.generate_videos",
                videos_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.genai.models",
                "AsyncModels.generate_videos",
                avideos_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument generate_videos: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("google.genai.models", "Models.generate_content")
            unwrap("google.genai.models", "AsyncModels.generate_content")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_content")

        try:
            unwrap("google.genai.models", "Models.generate_content_stream")
            unwrap("google.genai.models", "AsyncModels.generate_content_stream")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_content_stream")

        try:
            unwrap("google.genai.models", "Models.generate_images")
            unwrap("google.genai.models", "AsyncModels.generate_images")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_images")

        try:
            unwrap("google.genai.models", "Models.generate_videos")
            unwrap("google.genai.models", "AsyncModels.generate_videos")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument generate_videos")
