import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.dspy.version import __version__
from netra.instrumentation.dspy.wrappers import acall_wrapper, aforward_wrapper

logger = logging.getLogger(__name__)

_instruments = ("dspy-ai >= 2.0.0",)


class NetraDSPyInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom DSPy instrumentor for Netra SDK with enhanced support for:
    - LM.acall method (async call with callback support)
    - LM.aforward method (async forward for actual LLM calls)
    - OpenTelemetry semantic conventions for Generative AI
    - Integration with Netra tracing and monitoring
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Instrument DSPy LM methods"""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Instrument LM.acall method
        try:
            wrap_function_wrapper(
                "dspy.clients.base_lm",
                "BaseLM.acall",
                acall_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("BaseLM.acall method not available in this dspy-ai version")

        # Instrument LM.aforward method
        try:
            wrap_function_wrapper(
                "dspy.clients.lm",
                "LM.aforward",
                aforward_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("LM.aforward method not available in this dspy-ai version")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument DSPy LM methods"""
        # Uninstrument LM.acall
        try:
            unwrap("dspy.clients.base_lm", "BaseLM.acall")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument LM.aforward
        try:
            unwrap("dspy.clients.lm", "LM.aforward")
        except (AttributeError, ModuleNotFoundError):
            pass


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    from opentelemetry import context as context_api
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True

