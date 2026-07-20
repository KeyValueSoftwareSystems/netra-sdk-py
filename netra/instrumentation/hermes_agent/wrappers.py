import logging
from contextvars import ContextVar, Token
from typing import Any, Callable, Optional, Tuple

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.hermes_agent.utils import (
    APPROVAL_SPAN_NAME,
    SKILL_SPAN_NAME,
    SUBAGENT_TURN_SPAN_NAME,
    TURN_SPAN_NAME,
    set_approval_request_attributes,
    set_approval_response_attributes,
    set_skill_request_attributes,
    set_skill_response_attributes,
    set_tool_request_attributes,
    set_tool_response_attributes,
    set_turn_request_attributes,
    set_turn_response_attributes,
)

logger = logging.getLogger(__name__)

# Tool call ids currently being traced on this execution context. Hermes's
# handle_function_call re-enters itself when the tool_search bridge unwraps a
# `tool_call` invocation to the underlying tool, passing the SAME tool_call_id
# (model_tools.py). The recursion happens synchronously on the same thread, so
# a contextvar set is enough to avoid a duplicate span for one logical call.
# NOTE: skip_pre_tool_call_hook cannot be used for this — Hermes also passes it
# on the normal (non-recursive) dispatch path.
_active_tool_call_ids: ContextVar[frozenset[str]] = ContextVar(
    "netra_hermes_agent_active_tool_call_ids", default=frozenset()
)


def _get_arg(args: Tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    """
    Fetch a call argument by position with keyword fallback.

    Args:
        args (Tuple): Positional arguments of the wrapped call.
        kwargs (dict): Keyword arguments of the wrapped call.
        index (int): Positional index of the argument.
        name (str): Keyword name of the argument.

    Returns:
        Any: The argument value, or None when absent.
    """
    if len(args) > index:
        return args[index]
    return kwargs.get(name)


def run_conversation_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces agent.conversation_loop.run_conversation as a turn span.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: A sync wrapper function for run_conversation.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Wrap run_conversation in a hermes-agent turn span (AGENT type).

        Subagent turns (agents constructed with platform="subagent" by
        delegate_task) get the distinct span name ``hermes-agent.subagent.turn``
        plus gen_ai.agent.* identity attributes.

        Args:
            wrapped (Callable): The original run_conversation function.
            instance (Any): Unused; run_conversation is a module-level function.
            args (Tuple): Positional arguments (agent, user_message, ...).
            kwargs (dict): Keyword arguments of the call.

        Returns:
            Any: The dict returned by run_conversation.
        """
        agent = _get_arg(args, kwargs, 0, "agent")
        user_message = _get_arg(args, kwargs, 1, "user_message")
        task_id = _get_arg(args, kwargs, 4, "task_id")

        is_subagent = getattr(agent, "platform", None) == "subagent"
        span_name = SUBAGENT_TURN_SPAN_NAME if is_subagent else TURN_SPAN_NAME

        # Attach the Hermes session id as OTel baggage for the duration of the
        # turn: SessionSpanProcessor stamps baggage onto every span in the turn
        # as `netra.session_id`, which is what the backend groups traces by.
        # Existing baggage wins — an explicit Netra.set_session_id(), or the
        # parent turn of a delegated subagent (so a delegation tree groups
        # under the parent's session). The token is detached in `finally`;
        # Hermes reuses pool threads across turns, so a leaked attach would
        # bleed the session id into later, unrelated turns.
        session_id = getattr(agent, "session_id", None)
        baggage_token: Optional[Token[Context]] = None
        try:
            if session_id and baggage.get_baggage("session_id", otel_context.get_current()) is None:
                baggage_token = otel_context.attach(baggage.set_baggage("session_id", str(session_id)))
        except Exception as e:
            logger.error("Failed to attach hermes-agent session baggage: %s", e)

        try:
            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                try:
                    set_turn_request_attributes(span, agent, user_message, task_id, is_subagent)
                except Exception as e:
                    logger.error("Failed to set hermes-agent turn request attributes: %s", e)

                try:
                    result = wrapped(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                # Sets span status too: ERROR when the result dict reports a
                # failed turn, OK otherwise.
                try:
                    set_turn_response_attributes(span, result)
                except Exception as e:
                    logger.error("Failed to set hermes-agent turn response attributes: %s", e)
                return result
        finally:
            if baggage_token is not None:
                otel_context.detach(baggage_token)

    return wrapper


def tool_execution_middleware_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces agent.tool_executor._run_agent_tool_execution_middleware.

    Hermes dispatches its agent-runtime tools (todo, session_search, memory,
    clarify, read_terminal, delegate_task, context-engine and memory-manager
    tools) inline, bypassing model_tools.handle_function_call — but every one
    of those inline branches funnels through this middleware helper, so
    wrapping it yields a tool span for the whole family. Registry-dispatched
    tools never pass through it, so there is no overlap with
    handle_function_call_wrapper.

    NOTE: the target is a private helper — the most drift-prone of the patch
    targets. Re-verify its keyword-only signature after upstream Hermes merges.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: A sync wrapper function for _run_agent_tool_execution_middleware.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Wrap the inline-tool middleware in a tool span named after the dispatched tool.

        Args:
            wrapped (Callable): The original _run_agent_tool_execution_middleware.
            instance (Any): Unused; the target is a module-level function.
            args (Tuple): Positional arguments (agent,).
            kwargs (dict): Keyword-only arguments (function_name, function_args,
                           effective_task_id, tool_call_id, execute).

        Returns:
            Any: The (function_result, observed_args) tuple from the middleware.
        """
        agent = _get_arg(args, kwargs, 0, "agent")
        function_name = kwargs.get("function_name")
        function_args = kwargs.get("function_args")
        tool_call_id = kwargs.get("tool_call_id")
        session_id = getattr(agent, "session_id", None)
        turn_id = getattr(agent, "_current_turn_id", None)
        api_request_id = getattr(agent, "_current_api_request_id", None)

        with tracer.start_as_current_span(str(function_name), kind=SpanKind.CLIENT) as span:
            try:
                set_tool_request_attributes(
                    span, str(function_name), function_args, tool_call_id, session_id, turn_id, api_request_id
                )
            except Exception as e:
                logger.error("Failed to set hermes-agent inline tool request attributes: %s", e)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            try:
                function_result = result[0] if isinstance(result, tuple) and result else None
                set_tool_response_attributes(span, function_result)
            except Exception as e:
                logger.error("Failed to set hermes-agent inline tool response attributes: %s", e)
            span.set_status(Status(StatusCode.OK))
            return result

    return wrapper


def handle_function_call_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces model_tools.handle_function_call as a tool span.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: A sync wrapper function for handle_function_call.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Wrap handle_function_call in a tool span named after the dispatched tool.

        Recursive re-entries for the same tool_call_id (tool_search bridge
        unwrapping) are passed through without opening a second span.

        Args:
            wrapped (Callable): The original handle_function_call function.
            instance (Any): Unused; handle_function_call is a module-level function.
            args (Tuple): Positional arguments (function_name, function_args, ...).
            kwargs (dict): Keyword arguments of the call.

        Returns:
            Any: The result string returned by handle_function_call.
        """
        function_name = _get_arg(args, kwargs, 0, "function_name")
        function_args = _get_arg(args, kwargs, 1, "function_args")
        tool_call_id = _get_arg(args, kwargs, 3, "tool_call_id")
        session_id = _get_arg(args, kwargs, 4, "session_id")
        turn_id = _get_arg(args, kwargs, 5, "turn_id")
        api_request_id = _get_arg(args, kwargs, 6, "api_request_id")

        active_ids = _active_tool_call_ids.get()
        if tool_call_id and tool_call_id in active_ids:
            return wrapped(*args, **kwargs)

        reset_token: Optional[Token[frozenset[str]]] = None
        if tool_call_id:
            reset_token = _active_tool_call_ids.set(active_ids | {tool_call_id})
        try:
            with tracer.start_as_current_span(str(function_name), kind=SpanKind.CLIENT) as span:
                try:
                    set_tool_request_attributes(
                        span, str(function_name), function_args, tool_call_id, session_id, turn_id, api_request_id
                    )
                except Exception as e:
                    logger.error("Failed to set hermes-agent tool request attributes: %s", e)

                try:
                    result = wrapped(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

                try:
                    set_tool_response_attributes(span, result)
                except Exception as e:
                    logger.error("Failed to set hermes-agent tool response attributes: %s", e)
                span.set_status(Status(StatusCode.OK))
                return result
        finally:
            if reset_token is not None:
                _active_tool_call_ids.reset(reset_token)

    return wrapper


def skill_invocation_wrapper(tracer: Tracer, kind: str, target_arg: str) -> Callable[..., Any]:
    """
    Return a wrapper that traces a Hermes skill-invocation message builder.

    A ``/skill`` (single), ``/a /b …`` (stacked), or ``/bundle`` invocation is
    expanded into a scaffolded user message *before* the turn runs, so the turn
    span sees only the full skill body as opaque input. Wrapping the builder —
    the single choke-point every surface (CLI, TUI, gateway) reaches via a
    call-time ``from agent.skill_commands import …`` — emits a distinct span
    per invocation that records which skill(s) were invoked and the user's
    actual instruction (which the builders receive directly as an argument).

    The builder runs on the caller thread before ``run_conversation``, so this
    span is a standalone event, not a parent of the resulting turn span.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.
        kind (str): Invocation kind — ``single``, ``stacked``, or ``bundle``.
        target_arg (str): Keyword name of the skill-target argument
                          (``cmd_key`` for single/bundle, ``cmd_keys`` for
                          stacked).

    Returns:
        Callable: A sync wrapper function for the skill message builder.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        skill_target = _get_arg(args, kwargs, 0, target_arg)
        user_instruction = _get_arg(args, kwargs, 1, "user_instruction")

        with tracer.start_as_current_span(SKILL_SPAN_NAME, kind=SpanKind.CLIENT) as span:
            try:
                set_skill_request_attributes(span, kind, skill_target, user_instruction)
            except Exception as e:
                logger.error("Failed to set hermes-agent skill request attributes: %s", e)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            try:
                set_skill_response_attributes(span, result)
            except Exception as e:
                logger.error("Failed to set hermes-agent skill response attributes: %s", e)
            span.set_status(Status(StatusCode.OK))
            return result

    return wrapper


def approval_gate_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces agent tool_executor's dangerous-action approval gate.

    Wraps ``tools.approval._run_approval_gate`` — Hermes's documented single
    decision core reused by both dangerous-shell-command checks and plugin
    tool-approval escalations. Because approvals run inside tool execution, the
    span nests under the tool span that triggered it.

    NOTE: the target is a private helper; re-verify its keyword-only signature
    after upstream Hermes merges.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: A sync wrapper function for _run_approval_gate.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # _run_approval_gate is keyword-only.
        pattern_key = kwargs.get("pattern_key")
        description = kwargs.get("description")
        display_target = kwargs.get("display_target")

        with tracer.start_as_current_span(APPROVAL_SPAN_NAME, kind=SpanKind.CLIENT) as span:
            try:
                set_approval_request_attributes(span, pattern_key, description, display_target)
            except Exception as e:
                logger.error("Failed to set hermes-agent approval request attributes: %s", e)

            try:
                result = wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

            try:
                set_approval_response_attributes(span, result)
            except Exception as e:
                logger.error("Failed to set hermes-agent approval response attributes: %s", e)
            span.set_status(Status(StatusCode.OK))
            return result

    return wrapper
