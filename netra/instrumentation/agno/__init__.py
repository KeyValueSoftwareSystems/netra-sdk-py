import logging
from typing import Any, Collection, Optional, Tuple

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.agno.version import __version__
from netra.instrumentation.agno.wrappers import (
    agent_acontinue_run_wrapper,
    agent_arun_wrapper,
    agent_continue_run_wrapper,
    agent_run_wrapper,
    knowledge_search_wrapper,
    memory_add_wrapper,
    memory_search_wrapper,
    team_arun_wrapper,
    team_run_wrapper,
    tool_aexecute_wrapper,
    tool_execute_wrapper,
    vectordb_search_wrapper,
    vectordb_upsert_wrapper,
    workflow_arun_wrapper,
    workflow_run_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("agno >= 1.0.0",)


def _resolve_memory_target() -> Optional[Tuple[str, str]]:
    """Detect the Agno memory module path and class name.

    Agno versions may expose memory through different module paths:
    ``agno.memory.v2.memory.Memory``, ``agno.memory.manager.MemoryManager``,
    or ``agno.memory.memory.Memory``.

    Returns:
        A (module_path, class_name) tuple, or None if no memory class is found.
    """
    candidates = [
        ("agno.memory.v2.memory", "Memory"),
        ("agno.memory.manager", "MemoryManager"),
        ("agno.memory.memory", "Memory"),
    ]
    for module_path, class_name in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            if hasattr(mod, class_name):
                return (module_path, class_name)
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to import %s.%s: %s", module_path, class_name, e)
            continue
    return None


def _resolve_knowledge_target() -> Optional[Tuple[str, str]]:
    """Detect the Agno knowledge module path and class name.

    Returns:
        A (module_path, class_name) tuple, or None if no knowledge class is found.
    """
    candidates = [
        ("agno.knowledge.agent", "AgentKnowledge"),
        ("agno.knowledge.knowledge", "Knowledge"),
        ("agno.knowledge.base", "Knowledge"),
    ]
    for module_path, class_name in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            if hasattr(mod, class_name):
                return (module_path, class_name)
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to import %s.%s: %s", module_path, class_name, e)
            continue
    return None


class NetraAgnoInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Agno instrumentor for Netra SDK.

    Patches Agno Agent, Team, Workflow, Tool, VectorDB, Memory, and Knowledge
    entry points with OpenTelemetry spans following Netra conventions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._memory_target: Optional[Tuple[str, str]] = None
        self._knowledge_target: Optional[Tuple[str, str]] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the package requirements for this instrumentor."""
        return _instruments

    def _instrument(self, **kwargs: Any) -> Any:
        """Patch Agno classes with Netra tracing wrappers."""
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error("Failed to initialize tracer: %s", e)
            return

        try:
            wrap_function_wrapper(
                "agno.agent.agent",
                "Agent.run",
                agent_run_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.agent.agent",
                "Agent.arun",
                agent_arun_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument Agent.run/arun: %s", e)

        try:
            wrap_function_wrapper(
                "agno.agent.agent",
                "Agent.continue_run",
                agent_continue_run_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.agent.agent",
                "Agent.acontinue_run",
                agent_acontinue_run_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument Agent.continue_run/acontinue_run: %s", e)

        try:
            wrap_function_wrapper(
                "agno.tools.function",
                "FunctionCall.execute",
                tool_execute_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.tools.function",
                "FunctionCall.aexecute",
                tool_aexecute_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument FunctionCall.execute/aexecute: %s", e)

        try:
            wrap_function_wrapper(
                "agno.team.team",
                "Team.run",
                team_run_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.team.team",
                "Team.arun",
                team_arun_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument Team.run/arun: %s", e)

        try:
            wrap_function_wrapper(
                "agno.workflow.workflow",
                "Workflow.run_workflow",
                workflow_run_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.workflow.workflow",
                "Workflow.arun_workflow",
                workflow_arun_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument Workflow.run_workflow/arun_workflow: %s", e)

        try:
            wrap_function_wrapper(
                "agno.vectordb.base",
                "VectorDb.search",
                vectordb_search_wrapper(tracer),
            )
            wrap_function_wrapper(
                "agno.vectordb.base",
                "VectorDb.upsert",
                vectordb_upsert_wrapper(tracer),
            )
        except Exception as e:
            logger.error("Failed to instrument VectorDb.search/upsert: %s", e)

        self._memory_target = _resolve_memory_target()
        if self._memory_target:
            mem_module, mem_class = self._memory_target
            try:
                wrap_function_wrapper(
                    mem_module,
                    f"{mem_class}.add_user_memory",
                    memory_add_wrapper(tracer),
                )
                wrap_function_wrapper(
                    mem_module,
                    f"{mem_class}.search_user_memories",
                    memory_search_wrapper(tracer),
                )
            except Exception as e:
                logger.error("Failed to instrument %s memory methods: %s", mem_class, e)

        self._knowledge_target = _resolve_knowledge_target()
        if self._knowledge_target:
            know_module, know_class = self._knowledge_target
            try:
                wrap_function_wrapper(
                    know_module,
                    f"{know_class}.search",
                    knowledge_search_wrapper(tracer),
                )
            except Exception as e:
                logger.error("Failed to instrument %s.search: %s", know_class, e)

    def _uninstrument(self, **_kwargs: Any) -> None:
        """Remove Netra wrappers from Agno classes."""
        try:
            unwrap("agno.agent.agent", "Agent.run")
            unwrap("agno.agent.agent", "Agent.arun")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("netra.instrumentation.agno: failed to uninstrument Agent.run/arun: %s", e)

        try:
            unwrap("agno.agent.agent", "Agent.continue_run")
            unwrap("agno.agent.agent", "Agent.acontinue_run")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("netra.instrumentation.agno: failed to uninstrument Agent.continue_run/acontinue_run: %s", e)

        try:
            unwrap("agno.tools.function", "FunctionCall.execute")
            unwrap("agno.tools.function", "FunctionCall.aexecute")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("netra.instrumentation.agno: failed to uninstrument FunctionCall.execute/aexecute: %s", e)

        try:
            unwrap("agno.team.team", "Team.run")
            unwrap("agno.team.team", "Team.arun")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("netra.instrumentation.agno: failed to uninstrument Team.run/arun: %s", e)

        try:
            unwrap("agno.workflow.workflow", "Workflow.run_workflow")
            unwrap("agno.workflow.workflow", "Workflow.arun_workflow")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error(
                "netra.instrumentation.agno: failed to uninstrument Workflow.run_workflow/arun_workflow: %s", e
            )

        try:
            unwrap("agno.vectordb.base", "VectorDb.search")
            unwrap("agno.vectordb.base", "VectorDb.upsert")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("netra.instrumentation.agno: failed to uninstrument VectorDb.search/upsert: %s", e)

        if self._memory_target:
            mem_module, mem_class = self._memory_target
            try:
                unwrap(mem_module, f"{mem_class}.add_user_memory")
                unwrap(mem_module, f"{mem_class}.search_user_memories")
            except (AttributeError, ModuleNotFoundError) as e:
                logger.error("netra.instrumentation.agno: failed to uninstrument %s memory methods: %s", mem_class, e)

        if self._knowledge_target:
            know_module, know_class = self._knowledge_target
            try:
                unwrap(know_module, f"{know_class}.search")
            except (AttributeError, ModuleNotFoundError) as e:
                logger.error("netra.instrumentation.agno: failed to uninstrument %s.search: %s", know_class, e)


__all__ = ["NetraAgnoInstrumentor"]
