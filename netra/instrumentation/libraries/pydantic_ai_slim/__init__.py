import logging
import sys
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.libraries.pydantic_ai.version import __version__
from netra.instrumentation.libraries.pydantic_ai.wrappers import (
    agent_iter_wrapper,
    agent_run_stream_wrapper,
    agent_run_sync_wrapper,
    agent_run_wrapper,
    tool_function_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("pydantic-ai-slim >= 0.5.1",)


# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.pydantic_ai_slim"


# (module, class, method) for every method wrapped in ``_instrument``.
_WRAPPED_METHODS = (
    ("pydantic_ai.agent", "Agent", "run"),
    ("pydantic_ai.agent", "Agent", "run_sync"),
    ("pydantic_ai.agent", "Agent", "iter"),
    ("pydantic_ai.agent", "Agent", "run_stream"),
    ("pydantic_ai.agent", "AgentRun", "run"),
    ("pydantic_ai.agent", "AgentRun", "run_sync"),
    ("pydantic_ai.tools", "Tool", "__call__"),
    ("pydantic_ai.tools", "FunctionTool", "__call__"),
    ("pydantic_ai.models.base", "Model", "request"),
    ("pydantic_ai.models.base", "Model", "request_stream"),
)


def _unwrap_method(module_name: str, class_name: str, method_name: str) -> None:
    # ``unwrap("pkg.mod", "Class.method")`` silently does nothing (it looks up a
    # module attribute literally named "Class.method"), so resolve the class here.
    module = sys.modules.get(module_name)
    owner = getattr(module, class_name, None) if module is not None else None
    if owner is not None:
        unwrap(owner, method_name)


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
        tracer = get_tracer(_TRACER_NAME, __version__, tracer_provider)

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
        for module_name, class_name, method_name in _WRAPPED_METHODS:
            _unwrap_method(module_name, class_name, method_name)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    from opentelemetry import context as context_api
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
