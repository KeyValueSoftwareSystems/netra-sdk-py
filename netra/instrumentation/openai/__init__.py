import json
import logging
import time
from typing import Any, Collection, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.trace import SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from netra.instrumentation.openai.version import __version__
from netra.instrumentation.openai.wrappers import (
    achat_wrapper,
    acompletion_wrapper,
    aembeddings_wrapper,
    aresponses_wrapper,
    chat_wrapper,
    completion_wrapper,
    embeddings_wrapper,
    responses_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("openai >= 1.0.0",)


class NetraOpenAIInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom OpenAI instrumentor for Netra SDK with enhanced support for:
    - responses.create method
    - Proper streaming/non-streaming span handling
    - Integration with Netra tracing
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Instrument OpenAI client methods"""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Chat completions
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

        # Traditional completions
        wrap_function_wrapper(
            "openai.resources.completions",
            "Completions.create",
            completion_wrapper(tracer),
        )

        wrap_function_wrapper(
            "openai.resources.completions",
            "AsyncCompletions.create",
            acompletion_wrapper(tracer),
        )

        # Embeddings
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

        # New responses.create method
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
        except (AttributeError, ModuleNotFoundError):
            logger.debug("responses.create method not available in this OpenAI version")

        # Beta APIs
        try:
            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "Completions.parse",
                chat_wrapper(tracer),
            )

            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "AsyncCompletions.parse",
                achat_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Beta chat completions not available in this OpenAI version")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument OpenAI client methods"""
        # Chat completions
        unwrap("openai.resources.chat.completions", "Completions.create")
        unwrap("openai.resources.chat.completions", "AsyncCompletions.create")

        # Traditional completions
        unwrap("openai.resources.completions", "Completions.create")
        unwrap("openai.resources.completions", "AsyncCompletions.create")

        # Embeddings
        unwrap("openai.resources.embeddings", "Embeddings.create")
        unwrap("openai.resources.embeddings", "AsyncEmbeddings.create")

        # New responses.create method
        try:
            unwrap("openai.resources.responses", "Responses.create")
            unwrap("openai.resources.responses", "AsyncResponses.create")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Beta APIs
        try:
            unwrap("openai.resources.beta.chat.completions", "Completions.parse")
            unwrap("openai.resources.beta.chat.completions", "AsyncCompletions.parse")
        except (AttributeError, ModuleNotFoundError):
            pass


def is_streaming_response(response: Any) -> bool:
    """Check if response is a streaming response"""
    return hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict))


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
