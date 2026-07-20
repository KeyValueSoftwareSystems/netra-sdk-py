import json
import logging
import os
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

from netra.config import Config
from netra.instrumentation.utils import _safe_set_attribute
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

TURN_SPAN_NAME = "hermes-agent.turn"
SUBAGENT_TURN_SPAN_NAME = "hermes-agent.subagent.turn"
SKILL_SPAN_NAME = "hermes-agent.skill.invoke"
APPROVAL_SPAN_NAME = "hermes-agent.approval"

# How a skill was invoked. Single `/skill`, stacked `/a /b …` (up to 5), or a
# `/bundle` that loads a preconfigured group. All three share the same builder
# family in Hermes (agent.skill_commands / agent.skill_bundles).
SKILL_KIND_SINGLE = "single"
SKILL_KIND_STACKED = "stacked"
SKILL_KIND_BUNDLE = "bundle"

ATTR_SESSION_ID = "gen_ai.session.id"
ATTR_AGENT_ID = "gen_ai.agent.id"
ATTR_PARENT_SESSION_ID = "gen_ai.agent.parent_session.id"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_TASK_ID = "gen_ai.hermes_agent.task_id"
ATTR_TURN_ID = "gen_ai.hermes_agent.turn_id"
ATTR_API_REQUEST_ID = "gen_ai.hermes_agent.api_request_id"
ATTR_COMPLETED = "gen_ai.hermes_agent.completed"
ATTR_TURN_ERROR = "gen_ai.hermes_agent.error"

ATTR_SKILL_NAME = "gen_ai.hermes_agent.skill.name"
ATTR_SKILL_INVOCATION_KIND = "gen_ai.hermes_agent.skill.invocation_kind"
ATTR_SKILL_COUNT = "gen_ai.hermes_agent.skill.count"
ATTR_SKILL_LOADED = "gen_ai.hermes_agent.skill.loaded"

ATTR_APPROVAL_PATTERN_KEY = "gen_ai.hermes_agent.approval.pattern_key"
ATTR_APPROVAL_DESCRIPTION = "gen_ai.hermes_agent.approval.description"
ATTR_APPROVAL_APPROVED = "gen_ai.hermes_agent.approval.approved"

NETRA_SPAN_TYPE_ATTR = f"{Config.LIBRARY_NAME}.span.type"

# Content attributes (input/output) are truncated to this length; identifiers are not.
MAX_CONTENT_LENGTH = 10_000


def should_send_prompts() -> bool:
    """Whether prompt/completion content may be recorded on spans.

    Mirrors the convention used by the other Netra instrumentations:
    content capture is on unless TRACELOOP_TRACE_CONTENT is "false"
    (Config sets that env var from its ``trace_content`` setting), and can
    be force-enabled per-context via ``override_enable_content_tracing``.
    """
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


def _serialize(value: Any) -> str:
    """
    Serialize a value to a span-safe string.

    Args:
        value (Any): The value to serialize. Dicts and lists are JSON-encoded;
                     all other types are converted via str().

    Returns:
        str: A string representation of the input value.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _build_message_array(role: str, content: str) -> str:
    """
    Build a JSON-serialized message array for span input/output attributes.

    Args:
        role (str): The conversation role (e.g. "user", "assistant", "tool").
        content (str): The message content to include.

    Returns:
        str: A JSON string of the form ``[{"role": role, "content": content}]``.
    """
    return json.dumps([{"role": role, "content": content}])


def set_turn_request_attributes(
    span: Span,
    agent: Any,
    user_message: Any,
    task_id: Optional[str],
    is_subagent: bool,
) -> None:
    """
    Write request metadata for a Hermes conversation turn to the turn span.

    Args:
        span (Span): The turn span to write attributes to.
        agent (Any): The Hermes AIAgent instance the turn runs on.
        user_message (Any): The user message that started the turn.
        task_id (Optional[str]): Hermes task id for this turn, if provided.
        is_subagent (bool): Whether the agent is a delegated subagent
                            (adds gen_ai.agent.* identity attributes).

    Returns:
        None
    """
    _safe_set_attribute(span, NETRA_SPAN_TYPE_ATTR, SpanType.AGENT.value)
    _safe_set_attribute(span, ATTR_SESSION_ID, getattr(agent, "session_id", None))
    _safe_set_attribute(span, ATTR_TASK_ID, task_id)
    # Subagents can run a different model than their parent (override_model).
    _safe_set_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, getattr(agent, "model", None))
    if is_subagent:
        _safe_set_attribute(span, ATTR_AGENT_ID, getattr(agent, "_subagent_id", None))
        _safe_set_attribute(span, ATTR_PARENT_SESSION_ID, getattr(agent, "_parent_session_id", None))
    if should_send_prompts() and user_message is not None:
        _safe_set_attribute(span, "input", _build_message_array("user", _serialize(user_message)), MAX_CONTENT_LENGTH)


def set_turn_response_attributes(span: Span, result: Any) -> None:
    """
    Write the outcome of a Hermes conversation turn to the turn span, including status.

    ``run_conversation`` reports total API failure by *returning* a result dict
    (``completed=False``, ``failed=True``, ``error=...``) rather than raising, so
    span status is derived from the dict: ERROR when the turn failed, OK
    otherwise. User interrupts return ``completed=False`` without
    ``failed``/``error`` and stay OK; ``gen_ai.hermes_agent.completed`` records
    the distinction.

    Args:
        span (Span): The turn span to write attributes to.
        result (Any): The dict returned by ``run_conversation``; its
                      ``final_response`` entry becomes the ``output`` attribute.

    Returns:
        None
    """
    if not isinstance(result, dict):
        span.set_status(Status(StatusCode.OK))
        return

    final_response = result.get("final_response")
    if should_send_prompts() and final_response is not None:
        _safe_set_attribute(
            span, "output", _build_message_array("assistant", _serialize(final_response)), MAX_CONTENT_LENGTH
        )

    completed = result.get("completed")
    if completed is not None and span.is_recording():
        span.set_attribute(ATTR_COMPLETED, bool(completed))

    error = result.get("error")
    if error:
        _safe_set_attribute(span, ATTR_TURN_ERROR, _serialize(error), MAX_CONTENT_LENGTH)

    if result.get("failed") is True or bool(error):
        span.set_status(Status(StatusCode.ERROR, _serialize(error) if error else "hermes turn failed"))
    else:
        span.set_status(Status(StatusCode.OK))


def set_tool_request_attributes(
    span: Span,
    function_name: str,
    function_args: Any,
    tool_call_id: Optional[str],
    session_id: Optional[str],
    turn_id: Optional[str],
    api_request_id: Optional[str],
) -> None:
    """
    Write request metadata for a Hermes tool call to the tool span.

    Args:
        span (Span): The tool span to write attributes to.
        function_name (str): Name of the dispatched tool.
        function_args (Any): Arguments dict passed to the tool.
        tool_call_id (Optional[str]): The model-issued tool call id.
        session_id (Optional[str]): Hermes session id, if provided.
        turn_id (Optional[str]): Hermes turn id, if provided.
        api_request_id (Optional[str]): Hermes API request id, if provided.

    Returns:
        None
    """
    _safe_set_attribute(span, NETRA_SPAN_TYPE_ATTR, SpanType.TOOL.value)
    _safe_set_attribute(span, ATTR_TOOL_NAME, function_name)
    _safe_set_attribute(span, ATTR_TOOL_CALL_ID, tool_call_id)
    _safe_set_attribute(span, ATTR_SESSION_ID, session_id)
    _safe_set_attribute(span, ATTR_TURN_ID, turn_id)
    _safe_set_attribute(span, ATTR_API_REQUEST_ID, api_request_id)
    if should_send_prompts() and function_args is not None:
        _safe_set_attribute(
            span, "input", _build_message_array("assistant", _serialize(function_args)), MAX_CONTENT_LENGTH
        )


def set_tool_response_attributes(span: Span, result: Any) -> None:
    """
    Write the result of a Hermes tool call to the tool span.

    Hermes tool handlers report failures as error strings rather than raising,
    so the result is recorded verbatim; span status is only set to ERROR for
    real exceptions (handled in the wrapper).

    Args:
        span (Span): The tool span to write attributes to.
        result (Any): The result string returned by ``handle_function_call``.

    Returns:
        None
    """
    if should_send_prompts() and result is not None:
        _safe_set_attribute(span, "output", _build_message_array("tool", _serialize(result)), MAX_CONTENT_LENGTH)


def set_skill_request_attributes(
    span: Span,
    kind: str,
    skill_target: Any,
    user_instruction: Any,
) -> None:
    """
    Write request metadata for a Hermes skill invocation to the skill span.

    A ``/skill`` (or ``/bundle``) invocation is expanded into a scaffolded user
    message *before* the turn runs, so the turn span only ever sees the full
    skill body as opaque input. This span records the invocation as a distinct
    event: which skill(s), how they were invoked, and — the useful part — the
    user's actual instruction rather than the embedded skill body.

    Args:
        span (Span): The skill span to write attributes to.
        kind (str): Invocation kind — ``single``, ``stacked``, or ``bundle``.
        skill_target (Any): The invoked command key (``/slug`` or bundle key)
                            for single/bundle, or the list of ``/slug`` keys
                            for a stacked invocation.
        user_instruction (Any): Text the user typed alongside the command.

    Returns:
        None
    """
    _safe_set_attribute(span, NETRA_SPAN_TYPE_ATTR, SpanType.SPAN.value)
    _safe_set_attribute(span, ATTR_SKILL_INVOCATION_KIND, kind)
    if isinstance(skill_target, (list, tuple)):
        # Strip the leading slash for readability; keep declared order.
        names = [str(name).lstrip("/") for name in skill_target if name]
        _safe_set_attribute(span, ATTR_SKILL_NAME, ", ".join(names))
        if span.is_recording():
            span.set_attribute(ATTR_SKILL_COUNT, len(names))
    elif skill_target is not None:
        _safe_set_attribute(span, ATTR_SKILL_NAME, str(skill_target).lstrip("/"))
    if should_send_prompts() and user_instruction:
        _safe_set_attribute(
            span, "input", _build_message_array("user", _serialize(user_instruction)), MAX_CONTENT_LENGTH
        )


def set_skill_response_attributes(span: Span, result: Any) -> None:
    """
    Write the outcome of a Hermes skill invocation to the skill span.

    The single-skill builder returns the expanded message string (or ``None``
    when the skill was not found); the stacked/bundle builders return a
    ``(message, loaded_names, missing_names)`` tuple. In every case ``None``
    means nothing loaded — a normal not-found outcome, not an error.

    Args:
        span (Span): The skill span to write attributes to.
        result (Any): The value returned by the wrapped skill builder.

    Returns:
        None
    """
    if result is None:
        span.set_attribute(ATTR_SKILL_LOADED, False)
        return

    span.set_attribute(ATTR_SKILL_LOADED, True)
    # Stacked/bundle builders return (message, loaded_names, missing_names);
    # loaded_names is authoritative for the count and the resolved skill names.
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], list):
        loaded_names = [str(name) for name in result[1]]
        _safe_set_attribute(span, ATTR_SKILL_NAME, ", ".join(loaded_names))
        if span.is_recording():
            span.set_attribute(ATTR_SKILL_COUNT, len(loaded_names))


def set_approval_request_attributes(
    span: Span,
    pattern_key: Optional[str],
    description: Optional[str],
    display_target: Any,
) -> None:
    """
    Write request metadata for a Hermes dangerous-action approval gate.

    Args:
        span (Span): The approval span to write attributes to.
        pattern_key (Optional[str]): Allowlist/session key the decision is
                                     stored under.
        description (Optional[str]): Human-facing reason shown in the prompt.
        display_target (Any): The command string or synthetic tool label the
                              approval is gating. Recorded only when content
                              tracing is enabled — it can contain command
                              arguments.

    Returns:
        None
    """
    _safe_set_attribute(span, NETRA_SPAN_TYPE_ATTR, SpanType.SPAN.value)
    _safe_set_attribute(span, ATTR_APPROVAL_PATTERN_KEY, pattern_key)
    _safe_set_attribute(span, ATTR_APPROVAL_DESCRIPTION, description)
    if should_send_prompts() and display_target is not None:
        _safe_set_attribute(span, "input", _build_message_array("user", _serialize(display_target)), MAX_CONTENT_LENGTH)


def set_approval_response_attributes(span: Span, result: Any) -> None:
    """
    Write the outcome of a Hermes approval gate to the approval span.

    ``_run_approval_gate`` returns ``{"approved": bool, "message": str|None}``.
    A denial is a valid outcome, not a failure, so span status stays OK either
    way (only real exceptions, handled in the wrapper, set ERROR).

    Args:
        span (Span): The approval span to write attributes to.
        result (Any): The dict returned by ``_run_approval_gate``.

    Returns:
        None
    """
    if not isinstance(result, dict):
        return
    approved = result.get("approved")
    if approved is not None and span.is_recording():
        span.set_attribute(ATTR_APPROVAL_APPROVED, bool(approved))
