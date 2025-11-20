import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.litellm.utils import should_suppress_instrumentation
from netra.instrumentation.litellm.version import __version__
from netra.instrumentation.litellm.wrappers import (
    acompletion_wrapper,
    aembedding_wrapper,
    aimage_generation_wrapper,
    aresponses_wrapper,
    completion_wrapper,
    embedding_wrapper,
    image_generation_wrapper,
    responses_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("litellm >= 1.0.0",)


class LiteLLMInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom LiteLLM instrumentor for Netra SDK with enhanced support for:
    - completion() and acompletion() methods
    - embedding() and aembedding() methods
    - image_generation() and aimage_generation() methods
    - Proper streaming/non-streaming span handling
    - Integration with Netra tracing
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Instrument LiteLLM methods"""
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:  # Mirror OpenAI instrumentor error handling
            logger.error(f"Failed to initialize tracer: {e}")
            return

        # Chat completions
        try:
            wrap_function_wrapper(
                "litellm",
                "completion",
                completion_wrapper(tracer),
            )
            wrap_function_wrapper(
                "litellm",
                "acompletion",
                acompletion_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument LiteLLM completions: {e}")

        # Response
        try:
            wrap_function_wrapper(
                "litellm",
                "responses",
                responses_wrapper(tracer),
            )
            wrap_function_wrapper(
                "litellm",
                "aresponses",
                aresponses_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument LiteLLM completions: {e}")

        # Embeddings
        try:
            wrap_function_wrapper(
                "litellm",
                "embedding",
                embedding_wrapper(tracer),
            )
            wrap_function_wrapper(
                "litellm",
                "aembedding",
                aembedding_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument LiteLLM embeddings: {e}")

        # Image generation
        try:
            wrap_function_wrapper(
                "litellm",
                "image_generation",
                image_generation_wrapper(tracer),
            )
            wrap_function_wrapper(
                "litellm",
                "aimage_generation",
                aimage_generation_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument LiteLLM image generation: {e}")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument LiteLLM methods"""

        try:
            unwrap("litellm", "completion")
            unwrap("litellm", "acompletion")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument LiteLLM completions")

        try:
            unwrap("litellm", "embedding")
            unwrap("litellm", "aembedding")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument LiteLLM embeddings")

        try:
            unwrap("litellm", "image_generation")
            unwrap("litellm", "aimage_generation")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument LiteLLM image generation")
