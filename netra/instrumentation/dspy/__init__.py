import logging
from copy import copy, deepcopy
from typing import Any, Collection, Dict

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import BoundFunctionWrapper, FunctionWrapper, wrap_object

from netra.instrumentation.dspy.version import __version__
from netra.instrumentation.dspy.wrappers import (
    EmbedderCallWrapper,
    LMAsyncCallWrapper,
    LMCallWrapper,
    ModuleAsyncCallWrapper,
    RetrieverForwardWrapper,
    ToolAsyncCallWrapper,
    ToolCallWrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("dspy >= 2.0.0",)  # Changed from "dspy-ai" to "dspy" for v3.0+ support


class CopyableBoundFunctionWrapper(BoundFunctionWrapper):  # type: ignore
    """
    A bound function wrapper that can be copied and deep-copied.
    This allows DSPy classes to be copied when they use lm.copy().
    
    Reference: https://github.com/GrahamDumpleton/wrapt/issues/86#issuecomment-426161271
    """

    def __copy__(self) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(
            copy(self.__wrapped__), self._self_instance, self._self_wrapper
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(
            deepcopy(self.__wrapped__, memo), self._self_instance, self._self_wrapper
        )


class CopyableFunctionWrapper(FunctionWrapper):  # type: ignore
    """
    A function wrapper that can be copied and deep-copied.
    This is essential for DSPy's lm.copy() functionality.
    
    Reference: https://wrapt.readthedocs.io/en/master/wrappers.html#custom-function-wrappers
    """

    __bound_function_wrapper__ = CopyableBoundFunctionWrapper

    def __copy__(self) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(copy(self.__wrapped__), self._self_wrapper)

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(
            deepcopy(self.__wrapped__, memo), self._self_wrapper
        )


class NetraDSPyInstrumentor(BaseInstrumentor):  # type: ignore
    """
    Custom DSPy instrumentor for Netra SDK with comprehensive support for:
    - LM.__call__ method (primary sync interface)
    - LM.acall method (async interface)
    - Predict.forward and subclasses (CHoT, ReAct, etc.)
    - Module.__call__ (user-defined modules)
    - Module.acall (async user-defined modules)
    - Module.forward (direct forward calls)
    - Tool.__call__ (synchronous tool execution)
    - Tool.acall (asynchronous tool execution)
    - Retrieve.forward (retrieval operations)
    - Embedder.__call__ (embedding operations)
    - OpenTelemetry semantic conventions for Generative AI
    - Integration with Netra tracing and monitoring
    
    This implementation uses copyable wrappers to support DSPy's lm.copy() functionality.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Instrument DSPy components"""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Instrument LM.__call__ method (primary sync interface)
        try:
            wrap_object(
                module="dspy",
                name="LM.__call__",
                factory=CopyableFunctionWrapper,
                args=(LMCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.LM.__call__")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"LM.__call__ not available: {e}")

        # Instrument LM.acall method (async interface)
        try:
            wrap_object(
                module="dspy.clients.base_lm",
                name="BaseLM.acall",
                factory=CopyableFunctionWrapper,
                args=(LMAsyncCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.clients.base_lm.BaseLM.acall")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"BaseLM.acall not available: {e}")
        

        # Instrument Module.acall for async user-defined modules
        try:
            wrap_object(
                module="dspy",
                name="Module.acall",
                factory=CopyableFunctionWrapper,
                args=(ModuleAsyncCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.Module.acall")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"Module.acall not available: {e}")


        # Instrument Module.acall for async user-defined modules
        try:
            wrap_object(
                module="dspy",
                name="Module.acall",
                factory=CopyableFunctionWrapper,
                args=(ModuleAsyncCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.Module.acall")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"Module.acall not available: {e}")


        # Instrument Tool.__call__ for synchronous tool execution
        try:
            from dspy.adapters.types.tool import Tool
            wrap_object(
                module="dspy.adapters.types.tool",
                name="Tool.__call__",
                factory=CopyableFunctionWrapper,
                args=(ToolCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.adapters.types.tool.Tool.__call__")
        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            logger.warning(f"Tool.__call__ not available: {e}")
        except Exception as e:
            logger.error(f"Failed to instrument Tool.__call__: {e}", exc_info=True)

        # Instrument Tool.acall for asynchronous tool execution
        try:
            from dspy.adapters.types.tool import Tool
            wrap_object(
                module="dspy.adapters.types.tool",
                name="Tool.acall",
                factory=CopyableFunctionWrapper,
                args=(ToolAsyncCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.adapters.types.tool.Tool.acall")
        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            logger.warning(f"Tool.acall not available: {e}")
        except Exception as e:
            logger.error(f"Failed to instrument Tool.acall: {e}", exc_info=True)

        # Instrument Retrieve.forward for retrieval operations
        try:
            wrap_object(
                module="dspy",
                name="Retrieve.forward",
                factory=CopyableFunctionWrapper,
                args=(RetrieverForwardWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.Retrieve.forward")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"Retrieve.forward not available: {e}")

        # Instrument Embedder.__call__ for embedding operations
        try:
            wrap_object(
                module="dspy",
                name="Embedder.__call__",
                factory=CopyableFunctionWrapper,
                args=(EmbedderCallWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.Embedder.__call__")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"Embedder.__call__ not available: {e}")

        # Instrument retriever models (ColBERTv2, etc.)
        try:
            wrap_object(
                module="dspy",
                name="ColBERTv2.__call__",
                factory=CopyableFunctionWrapper,
                args=(RetrieverForwardWrapper(tracer),),
            )
            logger.debug("Instrumented dspy.ColBERTv2.__call__")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.debug(f"ColBERTv2.__call__ not available: {e}")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument DSPy components"""
        
        # Uninstrument LM methods
        try:
            unwrap("dspy", "LM.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("dspy.clients.base_lm", "BaseLM.acall")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument Predict and subclasses
        try:
            from dspy import Predict

            unwrap("dspy", "Predict.forward")
            
            predict_subclasses = Predict.__subclasses__()
            for predict_subclass in predict_subclasses:
                try:
                    unwrap("dspy", f"{predict_subclass.__name__}.forward")
                except (AttributeError, ModuleNotFoundError):
                    pass
        except (AttributeError, ModuleNotFoundError, ImportError):
            pass

        # Uninstrument Module
        try:
            unwrap("dspy", "Module.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("dspy", "Module.acall")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("dspy", "Module.forward")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument Tool
        try:
            unwrap("dspy.adapters.types.tool", "Tool.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("dspy.adapters.types.tool", "Tool.acall")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument Retrieve
        try:
            unwrap("dspy", "Retrieve.forward")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument Embedder
        try:
            unwrap("dspy", "Embedder.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument retriever models
        try:
            unwrap("dspy", "ColBERTv2.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    from opentelemetry import context as context_api
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
