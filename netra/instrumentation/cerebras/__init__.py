import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.cerebras.version import __version__
from netra.instrumentation.cerebras.wrappers import (
    achat_wrapper,
    acompletions_wrapper,
    chat_wrapper,
    completions_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("cerebras-cloud-sdk >= 1.0.0",)


class NetraCerebrasInstrumentor(BaseInstrumentor):  # type:ignore[misc]
    """Custom Cerebras instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer = get_tracer(__name__, __version__, kwargs.get("tracer_provider"))

        try:
            wrap_function_wrapper(
                "cerebras.cloud.sdk.resources.chat.completions",
                "CompletionsResource.create",
                chat_wrapper(tracer),
            )
            wrap_function_wrapper(
                "cerebras.cloud.sdk.resources.chat.completions",
                "AsyncCompletionsResource.create",
                achat_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Cerebras chat completions: {e}")

        try:
            wrap_function_wrapper(
                "cerebras.cloud.sdk.resources.completions",
                "CompletionsResource.create",
                completions_wrapper(tracer),
            )
            wrap_function_wrapper(
                "cerebras.cloud.sdk.resources.completions",
                "AsyncCompletionsResource.create",
                acompletions_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Cerebras completions: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("cerebras.cloud.sdk.resources.chat.completions", "CompletionsResource.create")
            unwrap("cerebras.cloud.sdk.resources.chat.completions", "AsyncCompletionsResource.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Cerebras chat completions")

        try:
            unwrap("cerebras.cloud.sdk.resources.completions", "CompletionsResource.create")
            unwrap("cerebras.cloud.sdk.resources.completions", "AsyncCompletionsResource.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument Cerebras completions")
