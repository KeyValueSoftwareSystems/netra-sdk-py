import logging
from pathlib import Path
from typing import Any

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Tracer, get_tracer

from netra.instrumentation.hermes_agent.utils import (
    SKILL_KIND_BUNDLE,
    SKILL_KIND_SINGLE,
    SKILL_KIND_STACKED,
)
from netra.instrumentation.hermes_agent.version import __version__
from netra.instrumentation.hermes_agent.wrappers import (
    approval_gate_wrapper,
    handle_function_call_wrapper,
    run_conversation_wrapper,
    skill_invocation_wrapper,
    tool_execution_middleware_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("hermes-agent >= 0.18.0",)


def _is_hermes_agent_environment() -> bool:
    """
    Verify the importable ``agent`` / ``model_tools`` modules belong to a Hermes install.

    Hermes ships generically-named top-level modules (``agent``, ``model_tools``);
    an unrelated package earlier on sys.path could shadow them. Both patch targets
    must expose the expected functions and live under the same install root
    (Hermes ships them together), otherwise nothing is patched.

    Returns:
        bool: True when both patch targets look like a Hermes Agent install.
    """
    try:
        import agent.conversation_loop as conversation_loop
        import model_tools
    except Exception:
        return False

    if not callable(getattr(conversation_loop, "run_conversation", None)):
        return False
    if not callable(getattr(model_tools, "handle_function_call", None)):
        return False

    conversation_loop_file = getattr(conversation_loop, "__file__", None)
    model_tools_file = getattr(model_tools, "__file__", None)
    if not conversation_loop_file or not model_tools_file:
        return False
    # agent/conversation_loop.py and model_tools.py sit under the same repo root.
    return Path(conversation_loop_file).resolve().parent.parent == Path(model_tools_file).resolve().parent


class NetraHermesAgentInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    def instrumentation_dependencies(self) -> tuple[str, ...]:
        """
        Return the list of packages required for this instrumentation to function.

        Args:
            None

        Returns:
            tuple: A tuple of pip requirement strings for the instrumented library.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Set up OpenTelemetry instrumentation for Hermes Agent.

        Wraps agent.conversation_loop.run_conversation (turn spans),
        model_tools.handle_function_call (registry tool spans), and
        agent.tool_executor._run_agent_tool_execution_middleware (inline
        agent-runtime tool spans) with tracing wrappers.

        Args:
            **kwargs: Accepts an optional 'tracer_provider' (TracerProvider) to use
                      instead of the global provider.

        Returns:
            None
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        if not _is_hermes_agent_environment():
            logger.warning(
                "hermes-agent instrumentation skipped: importable 'agent'/'model_tools' "
                "modules do not look like a Hermes Agent install"
            )
            return

        self._instrument_run_conversation(tracer)
        self._instrument_handle_function_call(tracer)
        self._instrument_tool_execution_middleware(tracer)
        self._instrument_skill_invocation(tracer)
        self._instrument_approval_gate(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Remove all custom instrumentation wrappers from Hermes Agent.

        Args:
            **kwargs: Not used; accepted for compatibility with BaseInstrumentor interface.

        Returns:
            None
        """
        self._uninstrument_run_conversation()
        self._uninstrument_handle_function_call()
        self._uninstrument_tool_execution_middleware()
        self._uninstrument_skill_invocation()
        self._uninstrument_approval_gate()

    def _instrument_run_conversation(self, tracer: Tracer) -> None:
        """
        Wrap agent.conversation_loop.run_conversation with a turn-span wrapper.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper("agent.conversation_loop", "run_conversation", run_conversation_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument hermes-agent run_conversation: {e}")

    def _instrument_handle_function_call(self, tracer: Tracer) -> None:
        """
        Wrap model_tools.handle_function_call with a tool-span wrapper.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper("model_tools", "handle_function_call", handle_function_call_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument hermes-agent handle_function_call: {e}")

    def _instrument_tool_execution_middleware(self, tracer: Tracer) -> None:
        """
        Wrap agent.tool_executor._run_agent_tool_execution_middleware with a tool-span wrapper.

        Covers Hermes's inline agent-runtime tools (todo, session_search, memory,
        clarify, read_terminal, delegate_task, context-engine and memory-manager
        tools), which bypass model_tools.handle_function_call.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper(
                "agent.tool_executor",
                "_run_agent_tool_execution_middleware",
                tool_execution_middleware_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument hermes-agent tool execution middleware: {e}")

    def _instrument_skill_invocation(self, tracer: Tracer) -> None:
        """
        Wrap Hermes's skill/bundle message builders with skill-invocation span wrappers.

        Covers the three ways a user invokes a skill from a slash command:
        single (``agent.skill_commands.build_skill_invocation_message``),
        stacked (``agent.skill_commands.build_stacked_skill_invocation_message``),
        and bundle (``agent.skill_bundles.build_bundle_invocation_message``).
        Each is wrapped independently so a missing target (older Hermes) never
        blocks the others.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the wrappers.

        Returns:
            None
        """
        skill_builders = (
            ("agent.skill_commands", "build_skill_invocation_message", SKILL_KIND_SINGLE, "cmd_key"),
            ("agent.skill_commands", "build_stacked_skill_invocation_message", SKILL_KIND_STACKED, "cmd_keys"),
            ("agent.skill_bundles", "build_bundle_invocation_message", SKILL_KIND_BUNDLE, "cmd_key"),
        )
        for module_name, func_name, kind, target_arg in skill_builders:
            try:
                wrapt.wrap_function_wrapper(module_name, func_name, skill_invocation_wrapper(tracer, kind, target_arg))
            except Exception as e:
                logger.error(f"Failed to instrument hermes-agent skill builder {module_name}.{func_name}: {e}")

    def _instrument_approval_gate(self, tracer: Tracer) -> None:
        """
        Wrap tools.approval._run_approval_gate with an approval-lifecycle span wrapper.

        This is Hermes's single decision core for dangerous-action approvals
        (dangerous shell commands and plugin tool-approval escalations), so one
        wrap covers the whole approval lifecycle across CLI and gateway
        surfaces.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper("tools.approval", "_run_approval_gate", approval_gate_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument hermes-agent approval gate: {e}")

    def _uninstrument_run_conversation(self) -> None:
        """
        Remove the tracing wrapper from agent.conversation_loop.run_conversation.

        Args:
            None

        Returns:
            None
        """
        try:
            import agent.conversation_loop as conversation_loop

            unwrap(conversation_loop, "run_conversation")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent run_conversation")

    def _uninstrument_handle_function_call(self) -> None:
        """
        Remove the tracing wrapper from model_tools.handle_function_call.

        Args:
            None

        Returns:
            None
        """
        try:
            # unwrap() cannot take "model_tools" as a string (it requires a dotted
            # path), so the module object is passed directly.
            import model_tools

            unwrap(model_tools, "handle_function_call")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent handle_function_call")

    def _uninstrument_tool_execution_middleware(self) -> None:
        """
        Remove the tracing wrapper from agent.tool_executor._run_agent_tool_execution_middleware.

        Args:
            None

        Returns:
            None
        """
        try:
            import agent.tool_executor as tool_executor

            unwrap(tool_executor, "_run_agent_tool_execution_middleware")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent tool execution middleware")

    def _uninstrument_skill_invocation(self) -> None:
        """
        Remove the tracing wrappers from Hermes's skill/bundle message builders.

        Args:
            None

        Returns:
            None
        """
        try:
            import agent.skill_commands as skill_commands

            unwrap(skill_commands, "build_skill_invocation_message")
            unwrap(skill_commands, "build_stacked_skill_invocation_message")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent skill commands")
        try:
            import agent.skill_bundles as skill_bundles

            unwrap(skill_bundles, "build_bundle_invocation_message")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent skill bundles")

    def _uninstrument_approval_gate(self) -> None:
        """
        Remove the tracing wrapper from tools.approval._run_approval_gate.

        Args:
            None

        Returns:
            None
        """
        try:
            import tools.approval as approval

            unwrap(approval, "_run_approval_gate")
        except (AttributeError, ImportError):
            logger.error("Failed to uninstrument hermes-agent approval gate")
