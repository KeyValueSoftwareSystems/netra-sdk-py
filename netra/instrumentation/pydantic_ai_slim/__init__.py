import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.pydantic_ai.version import __version__
from netra.instrumentation.pydantic_ai.wrappers import (
    agent_iter_wrapper,
    agent_run_stream_wrapper,
    agent_run_sync_wrapper,
    agent_run_wrapper,
    tool_function_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("pydantic-ai-slim >= 0.5.1",)


class NetraPydanticAISlimInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Pydantic AI instrumentor for Netra SDK with enhanced support for:
    - Agent.run, Agent.run_sync, Agent.iter, Agent.run_stream methods
    - Tool function execution tracing
    - OpenTelemetry semantic conventions for Generative AI
    - Integration with Netra tracing and monitoring
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Instrument Pydantic AI Agent methods and tool functions"""
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Instrument Agent.run method
        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "Agent.run",
                agent_run_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Agent.run method not available in this pydantic-ai version")

        # Instrument Agent.run_sync method
        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "Agent.run_sync",
                agent_run_sync_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Agent.run_sync method not available in this pydantic-ai version")

        # Instrument Agent.iter method
        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "Agent.iter",
                agent_iter_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Agent.iter method not available in this pydantic-ai version")

        # Instrument Agent.run_stream method (if available)
        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "Agent.run_stream",
                agent_run_stream_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Agent.run_stream method not available in this pydantic-ai version")

        # Instrument AgentRun methods
        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "AgentRun.run",
                agent_run_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("AgentRun.run method not available in this pydantic-ai version")

        try:
            wrap_function_wrapper(
                "pydantic_ai.agent",
                "AgentRun.run_sync",
                agent_run_sync_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("AgentRun.run_sync method not available in this pydantic-ai version")

        # Instrument tool execution (if tools module exists)
        try:
            wrap_function_wrapper(
                "pydantic_ai.tools",
                "Tool.__call__",
                tool_function_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Tool.__call__ method not available in this pydantic-ai version")

        # Instrument function tools (if function_tools module exists)
        try:
            wrap_function_wrapper(
                "pydantic_ai.tools",
                "FunctionTool.__call__",
                tool_function_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("FunctionTool.__call__ method not available in this pydantic-ai version")

        # Instrument model calls (if models module exists)
        try:
            wrap_function_wrapper(
                "pydantic_ai.models.base",
                "Model.request",
                agent_run_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Model.request method not available in this pydantic-ai version")

        try:
            wrap_function_wrapper(
                "pydantic_ai.models.base",
                "Model.request_stream",
                agent_run_stream_wrapper(tracer),
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Model.request_stream method not available in this pydantic-ai version")

    def _uninstrument(self, **kwargs):  # type: ignore[no-untyped-def]
        """Uninstrument Pydantic AI Agent methods and tool functions"""
        # Uninstrument Agent methods
        try:
            unwrap("pydantic_ai.agent", "Agent.run")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.agent", "Agent.run_sync")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.agent", "Agent.iter")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.agent", "Agent.run_stream")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument AgentRun methods
        try:
            unwrap("pydantic_ai.agent", "AgentRun.run")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.agent", "AgentRun.run_sync")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument tool methods
        try:
            unwrap("pydantic_ai.tools", "Tool.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.tools", "FunctionTool.__call__")
        except (AttributeError, ModuleNotFoundError):
            pass

        # Uninstrument model methods
        try:
            unwrap("pydantic_ai.models.base", "Model.request")
        except (AttributeError, ModuleNotFoundError):
            pass

        try:
            unwrap("pydantic_ai.models.base", "Model.request_stream")
        except (AttributeError, ModuleNotFoundError):
            pass


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    from opentelemetry import context as context_api
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
