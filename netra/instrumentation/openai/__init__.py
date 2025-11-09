import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.openai.version import __version__
from netra.instrumentation.openai.wrappers import (
    achat_wrapper,
    aembeddings_wrapper,
    aresponses_wrapper,
    chat_wrapper,
    embeddings_wrapper,
    responses_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("openai >= 1.0.0",)


class NetraOpenAIInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom OpenAI instrumentor for Netra SDK:
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Instrument OpenAI client methods"""

        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "openai.resources.chat.completions",
                "Completions.create",
                chat_wrapper(tracer),
            )
            wrap_function_wrapper(
                "openai.resources.chat.completions",
                "AsyncCompletions.create",
                achat_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument chat completions: {e}")

        try:
            wrap_function_wrapper(
                "openai.resources.embeddings",
                "Embeddings.create",
                embeddings_wrapper(tracer),
            )
            wrap_function_wrapper(
                "openai.resources.embeddings",
                "AsyncEmbeddings.create",
                aembeddings_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument embeddings: {e}")

        try:
            wrap_function_wrapper(
                "openai.resources.responses",
                "Responses.create",
                responses_wrapper(tracer),
            )
            wrap_function_wrapper(
                "openai.resources.responses",
                "AsyncResponses.create",
                aresponses_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument responses: {e}")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument OpenAI client methods"""

        try:
            unwrap("openai.resources.chat.completions", "Completions.create")
            unwrap("openai.resources.chat.completions", "AsyncCompletions.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument chat completions")

        try:
            unwrap("openai.resources.completions", "Completions.create")
            unwrap("openai.resources.completions", "AsyncCompletions.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument completions")

        try:
            unwrap("openai.resources.embeddings", "Embeddings.create")
            unwrap("openai.resources.embeddings", "AsyncEmbeddings.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument embeddings")

        try:
            unwrap("openai.resources.responses", "Responses.create")
            unwrap("openai.resources.responses", "AsyncResponses.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument responses")
