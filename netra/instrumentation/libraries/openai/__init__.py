from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.libraries.openai.version import __version__
from netra.instrumentation.libraries.openai.wrappers import (
    achat_wrapper,
    adecisions_wrapper,
    aembeddings_wrapper,
    aresponses_wrapper,
    chat_wrapper,
    decisions_wrapper,
    embeddings_wrapper,
    responses_wrapper,
)
from netra.netra_logger import get_logger

logger = get_logger(__name__)

_instruments = ("openai >= 1.0.0",)


# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.openai"

# The Decisions API only exists in newer openai releases, so its absence is expected
# on older installs and must not be reported as an instrumentation failure.
_DECISIONS_MODULE = "openai.resources.decisions"


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
            tracer = get_tracer(_TRACER_NAME, __version__, tracer_provider)
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

        self._instrument_decisions(tracer)

    @staticmethod
    def _instrument_decisions(tracer: Any) -> None:
        """Wrap ``decisions.create`` (sync and async) when the installed openai has it.

        Args:
            tracer (Any): The tracer the decision spans are started from.

        Returns:
            None
        """
        try:
            wrap_function_wrapper(_DECISIONS_MODULE, "Decisions.create", decisions_wrapper(tracer))
            wrap_function_wrapper(_DECISIONS_MODULE, "AsyncDecisions.create", adecisions_wrapper(tracer))
        except (AttributeError, ImportError):
            logger.debug("openai Decisions API not available in the installed openai version; skipping")
        except Exception as e:
            logger.error(f"Failed to instrument decisions: {e}")

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

        try:
            # unwrap() takes the dotted path of the class that owns the method; given
            # ``(module, "Class.method")`` it silently does nothing.
            unwrap(f"{_DECISIONS_MODULE}.Decisions", "create")
            unwrap(f"{_DECISIONS_MODULE}.AsyncDecisions", "create")
        except (AttributeError, ImportError):
            logger.debug("openai Decisions API not available in the installed openai version; nothing to unwrap")
