import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper, wrap_object

from netra.instrumentation.dspy.version import __version__
from netra.instrumentation.dspy.wrappers import (
    CopyableFunctionWrapper,
    EmbedderCallWrapper,
    LMAsyncCallWrapper,
    LMCallWrapper,
    ModuleAsyncCallWrapper,
    RetrieverForwardWrapper,
    ToolAsyncCallWrapper,
    ToolCallWrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("dspy >= 2.0.0",)


class NetraDSPyInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom DSPy instrumentor for Netra SDK"""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Instrument DSPy components"""
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_object(
                module="dspy",
                name="LM.__call__",
                factory=CopyableFunctionWrapper,
                args=(LMCallWrapper(tracer),),
            )
        except Exception as e:
            logger.error(f"Failed to instrument LM.__call__: {e}", exc_info=True)

        try:
            wrap_object(
                module="dspy.clients.base_lm",
                name="BaseLM.acall",
                factory=CopyableFunctionWrapper,
                args=(LMAsyncCallWrapper(tracer),),
            )
        except Exception as e:
            logger.error(f"Failed to instrument BaseLM.acall: {e}", exc_info=True)

        try:
            wrap_object(
                module="dspy",
                name="Module.acall",
                factory=CopyableFunctionWrapper,
                args=(ModuleAsyncCallWrapper(tracer),),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Module.acall: {e}", exc_info=True)

        try:
            wrap_function_wrapper(
                "dspy.adapters.types.tool",
                "Tool.__call__",
                ToolCallWrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Tool.__call__: {e}", exc_info=True)

        try:
            wrap_function_wrapper(
                "dspy.adapters.types.tool",
                "Tool.acall",
                ToolAsyncCallWrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Tool.acall: {e}", exc_info=True)

        try:
            wrap_function_wrapper(
                "dspy",
                "Retrieve.forward",
                RetrieverForwardWrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Retrieve.forward: {e}", exc_info=True)

        try:
            wrap_function_wrapper(
                "dspy",
                "Embedder.__call__",
                EmbedderCallWrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument Embedder.__call__: {e}", exc_info=True)

        try:
            wrap_function_wrapper(
                "dspy",
                "ColBERTv2.__call__",
                RetrieverForwardWrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument ColBERTv2.__call__: {e}", exc_info=True)

    def _uninstrument(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Uninstrument DSPy components"""
        try:
            unwrap("dspy", "LM.__call__")
            unwrap("dspy.clients.base_lm", "BaseLM.acall")
        except Exception as e:
            logger.error(f"Failed to uninstrument LM.__call__: {e}", exc_info=True)

        try:
            try:
                from dspy import Predict

                unwrap("dspy", "Predict.forward")
                predict_subclasses = Predict.__subclasses__()
                for predict_subclass in predict_subclasses:
                    try:
                        unwrap("dspy", f"{predict_subclass.__name__}.forward")
                    except Exception as e:
                        logger.error(
                            f"Failed to uninstrument Predict subclass forward method: {e}",
                            exc_info=True,
                        )
            except ImportError:
                # If dspy is not installed or Predict is unavailable, skip Predict-specific uninstrumentation
                pass
        except Exception as e:
            logger.error(f"Failed to uninstrument Predict and subclasses: {e}", exc_info=True)

        try:
            unwrap("dspy", "Module.__call__")
            unwrap("dspy", "Module.acall")
            unwrap("dspy", "Module.forward")
        except Exception as e:
            logger.error(f"Failed to uninstrument Module methods: {e}", exc_info=True)

        try:
            unwrap("dspy.adapters.types.tool", "Tool.__call__")
            unwrap("dspy.adapters.types.tool", "Tool.acall")
        except Exception as e:
            logger.error(f"Failed to uninstrument Tool methods: {e}", exc_info=True)

        try:
            unwrap("dspy", "Retrieve.forward")
        except Exception as e:
            logger.error(f"Failed to uninstrument Retrieve.forward: {e}", exc_info=True)

        try:
            unwrap("dspy", "Embedder.__call__")
        except Exception as e:
            logger.error(f"Failed to uninstrument Embedder.__call__: {e}", exc_info=True)

        try:
            unwrap("dspy", "ColBERTv2.__call__")
        except Exception as e:
            logger.error(f"Failed to uninstrument ColBERTv2.__call__: {e}", exc_info=True)
