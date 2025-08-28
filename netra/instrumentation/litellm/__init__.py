import logging
import time
from typing import Any, Collection, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from netra.instrumentation.litellm.version import __version__
from netra.instrumentation.litellm.wrappers import (
    acompletion_wrapper,
    aembedding_wrapper,
    aimage_generation_wrapper,
    completion_wrapper,
    embedding_wrapper,
    image_generation_wrapper,
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
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        logger.debug("Starting LiteLLM instrumentation...")

        # Force import litellm to ensure it's available for wrapping
        try:
            import litellm
        except ImportError as e:
            logger.error(f"Failed to import litellm: {e}")
            return

        # Store original functions for uninstrumentation
        self._original_completion = getattr(litellm, "completion", None)
        self._original_acompletion = getattr(litellm, "acompletion", None)
        self._original_embedding = getattr(litellm, "embedding", None)
        self._original_aembedding = getattr(litellm, "aembedding", None)
        self._original_image_generation = getattr(litellm, "image_generation", None)
        self._original_aimage_generation = getattr(litellm, "aimage_generation", None)

        # Chat completions - use direct monkey patching with proper function wrapping
        if self._original_completion:
            try:

                def instrumented_completion(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = completion_wrapper(tracer)
                    return wrapper(self._original_completion, None, args, kwargs)

                litellm.completion = instrumented_completion
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.completion: {e}")

        if self._original_acompletion:
            try:

                async def instrumented_acompletion(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = acompletion_wrapper(tracer)
                    return await wrapper(self._original_acompletion, None, args, kwargs)

                litellm.acompletion = instrumented_acompletion
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.acompletion: {e}")

        # Embeddings
        if self._original_embedding:
            try:

                def instrumented_embedding(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = embedding_wrapper(tracer)
                    return wrapper(self._original_embedding, None, args, kwargs)

                litellm.embedding = instrumented_embedding
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.embedding: {e}")

        if self._original_aembedding:
            try:

                async def instrumented_aembedding(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = aembedding_wrapper(tracer)
                    return await wrapper(self._original_aembedding, None, args, kwargs)

                litellm.aembedding = instrumented_aembedding
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.aembedding: {e}")

        # Image generation
        if self._original_image_generation:
            try:

                def instrumented_image_generation(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = image_generation_wrapper(tracer)
                    return wrapper(self._original_image_generation, None, args, kwargs)

                litellm.image_generation = instrumented_image_generation
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.image_generation: {e}")

        if self._original_aimage_generation:
            try:

                async def instrumented_aimage_generation(*args, **kwargs):  # type: ignore[no-untyped-def]
                    wrapper = aimage_generation_wrapper(tracer)
                    return await wrapper(self._original_aimage_generation, None, args, kwargs)

                litellm.aimage_generation = instrumented_aimage_generation
            except Exception as e:
                logger.error(f"Failed to monkey-patch litellm.aimage_generation: {e}")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument LiteLLM methods"""
        try:
            import litellm

            # Restore original functions
            if hasattr(self, "_original_completion") and self._original_completion:
                litellm.completion = self._original_completion

            if hasattr(self, "_original_acompletion") and self._original_acompletion:
                litellm.acompletion = self._original_acompletion

            if hasattr(self, "_original_embedding") and self._original_embedding:
                litellm.embedding = self._original_embedding

            if hasattr(self, "_original_aembedding") and self._original_aembedding:
                litellm.aembedding = self._original_aembedding

            if hasattr(self, "_original_image_generation") and self._original_image_generation:
                litellm.image_generation = self._original_image_generation

            if hasattr(self, "_original_aimage_generation") and self._original_aimage_generation:
                litellm.aimage_generation = self._original_aimage_generation

        except ImportError:
            pass


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
