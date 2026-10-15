import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.config import Config
from netra.instrumentation.span_utils import record_span_timing
from netra.span_wrapper import SpanType
from netra.utils import truncate_string

logger = logging.getLogger(__name__)

TURN_SPAN_NAME = "codex.turn"
JOIN_SPAN_NAME = "codex.turn.join"
SUBAGENT_SPAN_NAME = "codex.subagent"
GENERATION_SPAN_NAME = "codex.generation"

GEN_AI_SYSTEM = "openai"

TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"

ATTR_THREAD_ID = "gen_ai.codex.thread_id"
ATTR_TURN_ID = "gen_ai.codex.turn_id"
ATTR_TURN_STATUS = "gen_ai.codex.turn.status"
ATTR_TURN_DURATION_MS = "gen_ai.codex.turn.duration_ms"
ATTR_TURN_STREAM_CLOSED_EARLY = "gen_ai.codex.turn.stream_closed_early"
ATTR_MODEL_PROVIDER = "gen_ai.codex.model_provider"
ATTR_REASONING_EFFORT = "gen_ai.codex.reasoning_effort"
ATTR_SANDBOX = "gen_ai.codex.sandbox"
ATTR_APPROVAL_POLICY = "gen_ai.codex.approval_policy"
ATTR_AUTO_APPROVE = "gen_ai.codex.auto_approve"
ATTR_CWD = "gen_ai.codex.cwd"
ATTR_MODEL_CONTEXT_WINDOW = "gen_ai.codex.model_context_window"
ATTR_REQUEST_INDEX = "gen_ai.codex.request_index"

# Codex talks to the model through the OpenAI Responses API.
LLM_REQUEST_TYPE = "response"

ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_TOOL_STATUS = "gen_ai.codex.tool.status"
ATTR_TOOL_DURATION_MS = "gen_ai.codex.tool.duration_ms"
ATTR_COMMAND = "gen_ai.codex.command"
ATTR_COMMAND_CWD = "gen_ai.codex.command.cwd"
ATTR_COMMAND_EXIT_CODE = "gen_ai.codex.command.exit_code"
ATTR_COMMAND_SOURCE = "gen_ai.codex.command.source"
ATTR_FILE_CHANGES = "gen_ai.codex.file_changes"
ATTR_MCP_SERVER = "gen_ai.codex.mcp.server"
ATTR_SUBAGENT_RECEIVER_THREAD_IDS = "gen_ai.codex.subagent.receiver_thread_ids"

NETRA_SPAN_TYPE_ATTR = f"{Config.LIBRARY_NAME}.span.type"

# Content attributes (input/output, command output, tool arguments) are truncated to this length.
MAX_CONTENT_LENGTH = 10_000

# A generation span's input is the turn's transcript so far. Keep the user input plus the
# most recent messages so long agentic turns don't produce unbounded attributes.
MAX_GENERATION_INPUT_MESSAGES = 30

# Thread metadata is captured from thread/start|resume|fork responses and read back
# when a turn starts; bound it so long-lived processes with many threads don't grow it forever.
_MAX_THREAD_INFO = 1024

_USAGE_TOKEN_FIELDS: dict[str, str] = {
    "input_tokens": SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    "output_tokens": SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    "total_tokens": SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    "cached_input_tokens": SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    "cache_write_input_tokens": SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
    "reasoning_output_tokens": SpanAttributes.LLM_USAGE_REASONING_TOKENS,
}

# ThreadItem types that become TOOL child spans. Everything else is either turn
# input/output (userMessage, agentMessage) or recorded as an event on the turn span.
_TOOL_ITEM_TYPES = frozenset(
    {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "webSearch",
        "imageGeneration",
        "collabAgentToolCall",
    }
)


def should_send_prompts() -> bool:
    """
    Whether prompt/completion content may be recorded on spans.

    Content capture is on unless TRACELOOP_TRACE_CONTENT is "false" (Config sets that
    env var from its ``trace_content`` setting), and can be force-enabled per-context
    via ``override_enable_content_tracing``.

    Returns:
        bool: True if content may be recorded.
    """
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


def _serialize(value: Any) -> str:
    """
    Serialize a value to a span-safe string.

    Args:
        value (Any): The value to serialize. Pydantic models are dumped to JSON, dicts and
                     lists are JSON-encoded, everything else goes through str().

    Returns:
        str: A string representation of the input value.
    """
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        except Exception:
            return str(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _to_str(value: Any) -> Optional[str]:
    """
    Reduce an SDK value (enum, pydantic RootModel, tagged union, path) to a plain string.

    Args:
        value (Any): The value to reduce.

    Returns:
        Optional[str]: The string form, or None if the value is None.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    root = getattr(value, "root", None)
    if root is not None and root is not value:
        return _to_str(root)
    tag = getattr(value, "type", None)
    if tag is not None:
        return _to_str(tag)
    return str(value)


def _get(obj: Any, name: str, alias: Optional[str] = None) -> Any:
    """
    Read a field from a pydantic model or a wire-format dict.

    Args:
        obj (Any): A pydantic model, dict, or None.
        name (str): The snake_case attribute name.
        alias (Optional[str]): The camelCase wire key, if it differs.

    Returns:
        Any: The field value, or None if absent.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        value = obj.get(name)
        if value is None and alias:
            value = obj.get(alias)
        return value
    return getattr(obj, name, None)


def _set(span: Span, key: str, value: Any) -> None:
    """
    Set a span attribute, skipping None and coercing non-primitive values to strings.

    Args:
        span (Span): The span to write to.
        key (str): The attribute key.
        value (Any): The attribute value.

    Returns:
        None
    """
    if value is None:
        return
    if not isinstance(value, (str, bool, int, float)):
        value = _to_str(value)
        if value is None:
            return
    span.set_attribute(key, value)


def _set_content(span: Span, key: str, value: Any) -> None:
    """
    Set a content attribute (subject to ``trace_content`` gating and truncation).

    Args:
        span (Span): The span to write to.
        key (str): The attribute key.
        value (Any): The content value; serialized to a string.

    Returns:
        None
    """
    if value is None or not should_send_prompts():
        return
    span.set_attribute(key, truncate_string(_serialize(value), MAX_CONTENT_LENGTH))


def _ms_to_ns(ms: Any) -> Optional[int]:
    """
    Convert a millisecond wall-clock timestamp to nanoseconds.

    Args:
        ms (Any): Milliseconds since epoch, or None.

    Returns:
        Optional[int]: Nanoseconds since epoch, or None.
    """
    if isinstance(ms, (int, float)) and not isinstance(ms, bool):
        return int(ms * 1_000_000)
    return None


# --------------------------------------------------------------------------------------
# Thread metadata
# --------------------------------------------------------------------------------------

_thread_info: dict[str, dict[str, Any]] = {}
_thread_info_lock = threading.Lock()


def record_thread_info(response: Any) -> None:
    """
    Remember the resolved model/sandbox/approval settings returned by thread/start|resume|fork.

    ``api.Thread`` only keeps the thread id, so this is the only place the resolved
    model is visible before a turn runs.

    Args:
        response (Any): A ThreadStartResponse / ThreadResumeResponse / ThreadForkResponse.

    Returns:
        None
    """
    thread_id = _get(_get(response, "thread"), "id")
    if not thread_id:
        return
    info = {
        "model": _get(response, "model"),
        "model_provider": _get(response, "model_provider", "modelProvider"),
        "reasoning_effort": _to_str(_get(response, "reasoning_effort", "reasoningEffort")),
        "approval_policy": _to_str(_get(response, "approval_policy", "approvalPolicy")),
        "sandbox": _to_str(_get(response, "sandbox")),
        "cwd": _to_str(_get(response, "cwd")),
    }
    with _thread_info_lock:
        _thread_info.pop(thread_id, None)
        _thread_info[thread_id] = info
        while len(_thread_info) > _MAX_THREAD_INFO:
            _thread_info.pop(next(iter(_thread_info)))


def get_thread_info(thread_id: str) -> dict[str, Any]:
    """
    Return the remembered settings for a thread.

    Args:
        thread_id (str): The Codex thread id.

    Returns:
        dict: The thread settings, or an empty dict if unknown.
    """
    with _thread_info_lock:
        return dict(_thread_info.get(thread_id, {}))


# --------------------------------------------------------------------------------------
# Turn state
# --------------------------------------------------------------------------------------


@dataclass
class TurnState:
    """Everything needed to build one ``codex.turn`` span from its notification stream."""

    tracer: Tracer
    span: Span
    ctx: Context
    client: Any
    thread_id: str
    model: Optional[str] = None
    turn_id: Optional[str] = None
    consumers: int = 0
    owner: Optional[object] = None
    completed: bool = False
    ended: bool = False
    first_message_seen: bool = False
    user_messages_seen: int = 0
    final_answer: Optional[str] = None
    last_message: Optional[str] = None
    item_spans: dict[str, Span] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    # Per-model-request (generation) bookkeeping. Codex reports usage once per model
    # request, so each `thread/tokenUsage/updated` closes one generation span covering
    # the messages and tool calls the model produced since the previous one.
    transcript: list[dict[str, str]] = field(default_factory=list)
    gen_outputs: list[dict[str, str]] = field(default_factory=list)
    pending_tool_results: list[dict[str, str]] = field(default_factory=list)
    recorded_tool_calls: set[str] = field(default_factory=set)
    gen_start_ns: Optional[int] = None
    gen_last_output_ns: Optional[int] = None
    last_tool_completed_ns: Optional[int] = None
    request_index: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


# Keyed by turn id. TurnHandle / AsyncTurnHandle are slots dataclasses without
# __weakref__, so per-turn state cannot live on the handle itself.
_turns: dict[str, TurnState] = {}
_turns_lock = threading.Lock()


def _input_messages(input_items: Any, params: Any) -> list[dict[str, str]]:
    """
    Build the ``input`` message array from wire-format turn input.

    Images are replaced by a placeholder (they are usually base64 data URLs), and
    ExternalMessage content (``tool_output``) is recorded with the ``tool`` role since
    it carries tool-level, not user, authority.

    Args:
        input_items (Any): The wire input passed to ``_start_turn`` (str, dict or list of dicts).
        params (Any): The TurnStartParams (model or dict), for ``tool_output``.

    Returns:
        list[dict[str, str]]: Messages of the form ``{"role", "content"}``.
    """
    if isinstance(input_items, str):
        input_items = [{"type": "text", "text": input_items}]
    elif isinstance(input_items, dict):
        input_items = [input_items]

    parts: list[str] = []
    for item in input_items or []:
        item_type = _to_str(_get(item, "type"))
        if item_type == "text":
            text = _get(item, "text")
            if text:
                parts.append(str(text))
        elif item_type == "mention":
            parts.append(f"[mention:{_get(item, 'name') or ''}]")
        elif item_type == "skill":
            parts.append(f"[skill:{_get(item, 'name') or ''}]")
        elif item_type:
            parts.append(f"[{item_type}]")

    messages: list[dict[str, str]] = []
    if parts:
        messages.append({"role": "user", "content": "\n".join(parts)})
    tool_output = _get(params, "tool_output", "toolOutput")
    if tool_output:
        messages.append({"role": "tool", "content": _serialize(tool_output)})
    return messages


def _uses_default_approval_handler(client: Any) -> Optional[bool]:
    """
    Whether the client auto-accepts command and file-change approvals (the SDK default).

    Args:
        client (Any): The CodexClient instance.

    Returns:
        Optional[bool]: True/False, or None if it cannot be determined.
    """
    try:
        handler = getattr(client, "_approval_handler", None)
        default = getattr(client, "_default_approval_handler", None)
        if handler is None or default is None:
            return None
        return bool(handler == default)
    except Exception:
        return None


def start_turn(tracer: Tracer, client: Any, thread_id: str, input_items: Any, params: Any) -> TurnState:
    """
    Start a ``codex.turn`` span (parented to the current context) and record request attributes.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        client (Any): The CodexClient starting the turn.
        thread_id (str): The Codex thread id.
        input_items (Any): The wire-format turn input.
        params (Any): The TurnStartParams (model or dict), or None.

    Returns:
        TurnState: State for the new turn; not yet registered under a turn id.
    """
    span = tracer.start_span(TURN_SPAN_NAME, kind=SpanKind.CLIENT)
    state = TurnState(tracer=tracer, span=span, ctx=trace.set_span_in_context(span), client=client, thread_id=thread_id)
    state.gen_start_ns = time.time_ns()
    try:
        info = get_thread_info(thread_id)
        model = _get(params, "model") or info.get("model")
        state.model = model

        _set(span, NETRA_SPAN_TYPE_ATTR, SpanType.AGENT.value)
        _set(span, SpanAttributes.LLM_SYSTEM, GEN_AI_SYSTEM)
        _set(span, ATTR_THREAD_ID, thread_id)
        _set(span, SpanAttributes.LLM_REQUEST_MODEL, model)
        _set(span, ATTR_MODEL_PROVIDER, info.get("model_provider"))
        _set(span, ATTR_REASONING_EFFORT, _to_str(_get(params, "effort")) or info.get("reasoning_effort"))
        _set(
            span,
            ATTR_APPROVAL_POLICY,
            _to_str(_get(params, "approval_policy", "approvalPolicy")) or info.get("approval_policy"),
        )
        _set(span, ATTR_SANDBOX, _to_str(_get(params, "sandbox_policy", "sandboxPolicy")) or info.get("sandbox"))
        _set(span, ATTR_CWD, _to_str(_get(params, "cwd")) or info.get("cwd"))
        _set(span, ATTR_AUTO_APPROVE, _uses_default_approval_handler(client))

        messages = _input_messages(input_items, params)
        state.transcript = [_bounded_message(message) for message in messages]
        if messages and should_send_prompts():
            for index, message in enumerate(messages):
                _set_content(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", message["role"])
                _set_content(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", message["content"])
            _set_content(span, "input", json.dumps(messages))
    except Exception as e:
        logger.error("Cannot set codex turn request attributes: %s", e)
    return state


def register_turn(state: TurnState, turn_id: str) -> bool:
    """
    Register turn state under its id once ``turn/start`` has returned.

    Args:
        state (TurnState): The state created by ``start_turn``.
        turn_id (str): The turn id from the ``turn/start`` response.

    Returns:
        bool: False if a turn with this id is already active (input joined an active turn).
    """
    state.turn_id = turn_id
    _set(state.span, ATTR_TURN_ID, turn_id)
    with _turns_lock:
        if turn_id in _turns:
            return False
        _turns[turn_id] = state
        return True


def end_join_span(state: TurnState) -> None:
    """
    Close the span for input that joined an already-active turn.

    The active turn keeps its own span; this short span records the extra input.

    Args:
        state (TurnState): The provisional state created for the joining input.

    Returns:
        None
    """
    try:
        state.span.update_name(JOIN_SPAN_NAME)
        state.span.end()
    except Exception as e:
        logger.error("Failed to end codex join span: %s", e)


def acquire_turn(turn_id: Optional[str]) -> Optional[TurnState]:
    """
    Register a stream consumer for a turn.

    Args:
        turn_id (Optional[str]): The turn id of the handle being streamed.

    Returns:
        Optional[TurnState]: The turn state, or None if the turn was not started through
                             the instrumented ``_start_turn``.
    """
    if not turn_id:
        return None
    with _turns_lock:
        state = _turns.get(turn_id)
        if state is not None:
            state.consumers += 1
        return state


def release_turn(state: TurnState, token: object) -> None:
    """
    Unregister a stream consumer, ending the turn span when the last one finishes.

    Args:
        state (TurnState): The turn state.
        token (object): The consumer token passed to ``handle_notification``.

    Returns:
        None
    """
    with _turns_lock:
        state.consumers -= 1
        last = state.consumers <= 0
        if last and state.turn_id is not None:
            _turns.pop(state.turn_id, None)
    with state.lock:
        if state.owner is token:
            state.owner = None
    if last:
        finish_turn(state)


def finish_open_turns_for_client(client: Any) -> None:
    """
    End turn spans whose handle was never fully streamed before the client closed.

    Args:
        client (Any): The CodexClient being closed.

    Returns:
        None
    """
    with _turns_lock:
        orphaned = [state for state in _turns.values() if state.client is client]
        for state in orphaned:
            if state.turn_id is not None:
                _turns.pop(state.turn_id, None)
    for state in orphaned:
        finish_turn(state)


def record_exception(state: TurnState, error: BaseException) -> None:
    """
    Record an exception on the turn span and mark it as failed.

    Args:
        state (TurnState): The turn state.
        error (BaseException): The exception raised while starting or streaming the turn.

    Returns:
        None
    """
    try:
        state.span.record_exception(error)
        state.span.set_status(Status(StatusCode.ERROR, str(error)))
    except Exception as e:
        logger.error("Failed to record codex turn exception: %s", e)


# --------------------------------------------------------------------------------------
# Notification handling
# --------------------------------------------------------------------------------------


def handle_notification(tracer: Tracer, state: TurnState, notification: Any, token: object) -> None:
    """
    Update the turn span (and its child spans) from one turn notification.

    When several handles stream the same turn, only one of them (the owner) records,
    so child spans and usage are not duplicated.

    Args:
        tracer (Tracer): The OpenTelemetry tracer for child spans.
        state (TurnState): The turn state.
        notification (Any): An ``openai_codex.models.Notification``.
        token (object): Identifies the consuming stream.

    Returns:
        None
    """
    with state.lock:
        if state.ended:
            return
        if state.owner is None:
            state.owner = token
        if state.owner is not token:
            return
        try:
            _dispatch(tracer, state, getattr(notification, "method", None), getattr(notification, "payload", None))
        except Exception as e:
            logger.error("Failed to record codex notification: %s", e)


def _dispatch(tracer: Tracer, state: TurnState, method: Optional[str], payload: Any) -> None:
    """
    Route a notification to its handler.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        state (TurnState): The turn state.
        method (Optional[str]): The JSON-RPC notification method.
        payload (Any): The typed notification payload.

    Returns:
        None
    """
    if method == "item/started":
        _on_item_started(tracer, state, payload)
    elif method == "item/completed":
        _on_item_completed(tracer, state, payload)
    elif method == "thread/tokenUsage/updated":
        _on_token_usage(state, payload)
    elif method == "turn/completed":
        _on_turn_completed(state, payload)
    elif method == "model/rerouted":
        to_model = _get(payload, "to_model", "toModel")
        state.span.add_event(
            "codex.model_rerouted",
            {
                "from_model": _to_str(_get(payload, "from_model", "fromModel")) or "",
                "to_model": _to_str(to_model) or "",
                "reason": _to_str(_get(payload, "reason")) or "",
            },
        )
        if to_model:
            state.model = str(to_model)
    elif method == "error":
        error = _get(payload, "error")
        state.span.add_event(
            "codex.error",
            {
                "message": _to_str(_get(error, "message")) or "",
                "will_retry": bool(_get(payload, "will_retry", "willRetry")),
            },
        )


def _item_root(payload: Any) -> Any:
    """
    Unwrap the concrete ThreadItem variant from an item notification.

    Args:
        payload (Any): An ItemStarted/ItemCompleted notification payload.

    Returns:
        Any: The concrete item (``ThreadItem.root``), or None.
    """
    item = _get(payload, "item")
    root = getattr(item, "root", None)
    return root if root is not None else item


def _tool_span_name(item_type: str, item: Any) -> str:
    """
    Return the child span name for a tool-type item.

    Args:
        item_type (str): The ThreadItem ``type``.
        item (Any): The concrete ThreadItem.

    Returns:
        str: The span name.
    """
    if item_type == "commandExecution":
        return "shell"
    if item_type == "fileChange":
        return "apply_patch"
    if item_type == "mcpToolCall":
        return f"{_get(item, 'server') or 'mcp'}.{_get(item, 'tool') or 'tool'}"
    if item_type == "dynamicToolCall":
        tool = _get(item, "tool") or "tool"
        namespace = _get(item, "namespace")
        return f"{namespace}.{tool}" if namespace else str(tool)
    if item_type == "webSearch":
        return "web_search"
    if item_type == "imageGeneration":
        return "image_generation"
    return SUBAGENT_SPAN_NAME


def _on_item_started(tracer: Tracer, state: TurnState, payload: Any) -> None:
    """
    Open a child span for a tool-type item; record first-message timing for agent messages.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        state (TurnState): The turn state.
        payload (Any): The ItemStartedNotification.

    Returns:
        None
    """
    item = _item_root(payload)
    item_type = _to_str(_get(item, "type"))
    item_id = _get(item, "id")
    started_ns = _ms_to_ns(_get(payload, "started_at_ms", "startedAtMs"))

    if item_type == "agentMessage" and not state.first_message_seen:
        state.first_message_seen = True
        event_time = started_ns / 1e9 if started_ns else time.time()
        record_span_timing(
            state.span, TIME_TO_FIRST_TOKEN, event_time, reference_time=state.start_time, record_event_timestamp=True
        )
        record_span_timing(state.span, RELATIVE_TIME_TO_FIRST_TOKEN, event_time, use_root_span=True)
        return

    if item_type in _TOOL_ITEM_TYPES and item_id and item_id not in state.item_spans:
        state.item_spans[item_id] = _start_tool_span(tracer, state, item_type, item, started_ns)
        _record_tool_call(state, item_type, item, started_ns)


def _start_tool_span(tracer: Tracer, state: TurnState, item_type: str, item: Any, start_ns: Optional[int]) -> Span:
    """
    Start a TOOL (or AGENT, for sub-agents) child span under the turn span.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        state (TurnState): The turn state.
        item_type (str): The ThreadItem ``type``.
        item (Any): The concrete ThreadItem.
        start_ns (Optional[int]): Start time in ns since epoch.

    Returns:
        Span: The started span.
    """
    span = tracer.start_span(
        _tool_span_name(item_type, item), context=state.ctx, kind=SpanKind.INTERNAL, start_time=start_ns
    )
    span_type = SpanType.AGENT if item_type == "collabAgentToolCall" else SpanType.TOOL
    _set(span, NETRA_SPAN_TYPE_ATTR, span_type.value)
    _set(span, ATTR_TOOL_NAME, _tool_span_name(item_type, item))
    _set(span, ATTR_TOOL_CALL_ID, _get(item, "id"))
    _set(span, ATTR_THREAD_ID, state.thread_id)
    _set(span, ATTR_TURN_ID, state.turn_id)
    return span


def _on_item_completed(tracer: Tracer, state: TurnState, payload: Any) -> None:
    """
    Close the item's child span with its final attributes, or record a non-tool item.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        state (TurnState): The turn state.
        payload (Any): The ItemCompletedNotification.

    Returns:
        None
    """
    item = _item_root(payload)
    item_type = _to_str(_get(item, "type"))
    item_id = _get(item, "id")
    completed_ns = _ms_to_ns(_get(payload, "completed_at_ms", "completedAtMs"))

    if item_type == "agentMessage":
        text = _get(item, "text")
        phase = _to_str(_get(item, "phase"))
        if text:
            if phase == "final_answer":
                state.final_answer = text
            elif phase is None:
                state.last_message = text
            state.gen_outputs.append(_bounded_message({"role": "assistant", "content": str(text)}))
            state.gen_last_output_ns = completed_ns or time.time_ns()
        return

    if item_type == "userMessage":
        state.user_messages_seen += 1
        # The first user message is the turn input; later ones arrive via steer().
        if state.user_messages_seen > 1 and should_send_prompts():
            state.span.add_event(
                "codex.steer",
                {"content": truncate_string(_user_message_text(item), MAX_CONTENT_LENGTH)},
            )
        return

    if item_type in _TOOL_ITEM_TYPES:
        span = state.item_spans.pop(item_id, None) if item_id else None
        if span is None:
            span = _start_tool_span(tracer, state, item_type, item, completed_ns)
        try:
            _set_tool_attributes(span, item_type, item)
        except Exception as e:
            logger.error("Cannot set codex %s attributes: %s", item_type, e)
        finally:
            span.end(end_time=completed_ns)
        _record_tool_call(state, item_type, item, completed_ns)
        _record_tool_result(state, item_type, item, completed_ns)
        return

    if item_type == "reasoning":
        attributes: dict[str, Any] = {}
        if should_send_prompts():
            summary = _get(item, "summary")
            if summary:
                attributes["summary"] = truncate_string(_serialize(summary), MAX_CONTENT_LENGTH)
        state.span.add_event("codex.reasoning", attributes, timestamp=completed_ns)
        state.gen_last_output_ns = completed_ns or time.time_ns()
        return

    if item_type == "plan":
        attributes = {}
        if should_send_prompts() and (text := _get(item, "text")):
            attributes["text"] = truncate_string(str(text), MAX_CONTENT_LENGTH)
        state.span.add_event("codex.plan", attributes, timestamp=completed_ns)
        return

    if item_type:
        # imageView, sleep, hookPrompt, review mode, contextCompaction, subAgentActivity,
        # functionCallOutput and any variant added after this instrumentor was written.
        state.span.add_event(f"codex.{item_type}", timestamp=completed_ns)


def _user_message_text(item: Any) -> str:
    """
    Extract the text parts of a userMessage item.

    Args:
        item (Any): The UserMessage ThreadItem.

    Returns:
        str: The concatenated text content.
    """
    parts = []
    for content in _get(item, "content") or []:
        content = getattr(content, "root", content)
        if _to_str(_get(content, "type")) == "text" and (text := _get(content, "text")):
            parts.append(str(text))
    return "\n".join(parts)


def _set_tool_attributes(span: Span, item_type: str, item: Any) -> None:
    """
    Write the completed item's details to its child span.

    Args:
        span (Span): The tool span.
        item_type (str): The ThreadItem ``type``.
        item (Any): The completed concrete ThreadItem.

    Returns:
        None
    """
    status = _to_str(_get(item, "status"))
    _set(span, ATTR_TOOL_STATUS, status)
    _set(span, ATTR_TOOL_DURATION_MS, _get(item, "duration_ms", "durationMs"))
    failed = status in ("failed", "declined")

    if item_type == "commandExecution":
        exit_code = _get(item, "exit_code", "exitCode")
        _set(span, ATTR_COMMAND_CWD, _to_str(_get(item, "cwd")))
        _set(span, ATTR_COMMAND_EXIT_CODE, exit_code)
        _set(span, ATTR_COMMAND_SOURCE, _to_str(_get(item, "source")))
        _set_content(span, ATTR_COMMAND, _get(item, "command"))
        _set_content(span, "input", _get(item, "command"))
        _set_content(span, "output", _get(item, "aggregated_output", "aggregatedOutput"))
        failed = failed or (isinstance(exit_code, int) and exit_code != 0)
    elif item_type == "fileChange":
        # Paths and change kinds only; diffs can be large and contain file contents.
        changes = [
            {"path": _get(change, "path"), "kind": _to_str(_get(change, "kind"))}
            for change in _get(item, "changes") or []
        ]
        _set_content(span, ATTR_FILE_CHANGES, changes)
        _set_content(span, "input", changes)
    elif item_type == "mcpToolCall":
        _set(span, ATTR_MCP_SERVER, _get(item, "server"))
        _set_content(span, "input", _get(item, "arguments"))
        _set_content(span, "output", _get(item, "result"))
        if (error := _get(item, "error")) is not None:
            failed = True
            span.set_attribute("error.message", truncate_string(_serialize(error), MAX_CONTENT_LENGTH))
    elif item_type == "dynamicToolCall":
        _set_content(span, "input", _get(item, "arguments"))
        _set_content(span, "output", _get(item, "content_items", "contentItems"))
        failed = failed or _get(item, "success") is False
    elif item_type == "webSearch":
        _set_content(span, "input", _get(item, "query"))
        _set_content(span, "output", _get(item, "results"))
    elif item_type == "imageGeneration":
        # Never record `result`: it is the generated image data.
        _set_content(span, "input", _get(item, "revised_prompt", "revisedPrompt"))
        if (failure := _get(item, "failure")) is not None:
            failed = True
            span.set_attribute("error.message", truncate_string(_serialize(failure), MAX_CONTENT_LENGTH))
    elif item_type == "collabAgentToolCall":
        _set(span, SpanAttributes.LLM_REQUEST_MODEL, _get(item, "model"))
        _set(span, ATTR_REASONING_EFFORT, _to_str(_get(item, "reasoning_effort", "reasoningEffort")))
        _set(span, "gen_ai.codex.subagent.tool", _to_str(_get(item, "tool")))
        receivers = _get(item, "receiver_thread_ids", "receiverThreadIds")
        if receivers:
            span.set_attribute(ATTR_SUBAGENT_RECEIVER_THREAD_IDS, [str(r) for r in receivers])
        _set_content(span, "input", _get(item, "prompt"))
        _set_content(span, "output", _get(item, "agents_states", "agentsStates"))

    if failed:
        span.set_status(Status(StatusCode.ERROR, status or "failed"))


def _on_token_usage(state: TurnState, payload: Any) -> None:
    """
    Close the current model request as a generation span carrying its usage.

    ``thread/tokenUsage/updated`` fires once per model request; ``last`` is that
    request's usage and ``total`` is thread-cumulative, so only ``last`` is recorded.

    Args:
        state (TurnState): The turn state.
        payload (Any): The ThreadTokenUsageUpdatedNotification.

    Returns:
        None
    """
    token_usage = _get(payload, "token_usage", "tokenUsage")
    _set(state.span, ATTR_MODEL_CONTEXT_WINDOW, _get(token_usage, "model_context_window", "modelContextWindow"))
    _emit_generation(state, _get(token_usage, "last"))


def _on_turn_completed(state: TurnState, payload: Any) -> None:
    """
    Record the turn's final status, error and duration.

    Args:
        state (TurnState): The turn state.
        payload (Any): The TurnCompletedNotification.

    Returns:
        None
    """
    turn = _get(payload, "turn")
    if _get(turn, "id") not in (None, state.turn_id):
        return
    state.completed = True
    status = _to_str(_get(turn, "status"))
    _set(state.span, ATTR_TURN_STATUS, status)
    _set(state.span, ATTR_TURN_DURATION_MS, _get(turn, "duration_ms", "durationMs"))
    if status == "failed":
        error = _get(turn, "error")
        message = _to_str(_get(error, "message")) or "turn failed"
        state.span.set_status(Status(StatusCode.ERROR, message))


def _bounded_message(message: dict[str, str]) -> dict[str, str]:
    """
    Truncate a transcript message's content so per-turn state stays bounded.

    Args:
        message (dict): A ``{"role", "content", ...}`` message.

    Returns:
        dict: The message with its content truncated to ``MAX_CONTENT_LENGTH``.
    """
    return {**message, "content": truncate_string(message.get("content") or "", MAX_CONTENT_LENGTH)}


def _tool_call_arguments(item_type: str, item: Any) -> Any:
    """
    Return the arguments the model issued for a tool-type item.

    Args:
        item_type (str): The ThreadItem ``type``.
        item (Any): The concrete ThreadItem.

    Returns:
        Any: A JSON-serializable view of the call's arguments.
    """
    if item_type == "commandExecution":
        return {"command": _get(item, "command")}
    if item_type == "fileChange":
        # Paths and change kinds only; diffs can be large and contain file contents.
        return {
            "changes": [
                {"path": _get(change, "path"), "kind": _to_str(_get(change, "kind"))}
                for change in _get(item, "changes") or []
            ]
        }
    if item_type in ("mcpToolCall", "dynamicToolCall"):
        return _get(item, "arguments")
    if item_type == "webSearch":
        return {"query": _get(item, "query")}
    if item_type == "imageGeneration":
        return {"prompt": _get(item, "revised_prompt", "revisedPrompt")}
    if item_type == "collabAgentToolCall":
        return {"prompt": _get(item, "prompt")}
    return None


def _tool_call_result(item_type: str, item: Any) -> Any:
    """
    Return the result fed back to the model for a completed tool-type item.

    Args:
        item_type (str): The ThreadItem ``type``.
        item (Any): The completed concrete ThreadItem.

    Returns:
        Any: The tool result (never image bytes or file diffs).
    """
    if item_type == "commandExecution":
        return _get(item, "aggregated_output", "aggregatedOutput")
    if item_type == "mcpToolCall":
        error = _get(item, "error")
        return error if error is not None else _get(item, "result")
    if item_type == "dynamicToolCall":
        return _get(item, "content_items", "contentItems")
    if item_type == "webSearch":
        return _get(item, "results")
    if item_type == "imageGeneration":
        failure = _get(item, "failure")
        return failure if failure is not None else _to_str(_get(item, "status"))
    if item_type == "collabAgentToolCall":
        return _get(item, "agents_states", "agentsStates")
    return _to_str(_get(item, "status"))


def _record_tool_call(state: TurnState, item_type: str, item: Any, at_ns: Optional[int]) -> None:
    """
    Add a tool call to the current model request's output (once per item).

    Args:
        state (TurnState): The turn state.
        item_type (str): The ThreadItem ``type``.
        item (Any): The concrete ThreadItem.
        at_ns (Optional[int]): When the model issued the call, in ns since epoch.

    Returns:
        None
    """
    item_id = _get(item, "id")
    if not item_id or item_id in state.recorded_tool_calls:
        return
    state.recorded_tool_calls.add(item_id)
    call = {"name": _tool_span_name(item_type, item), "arguments": _tool_call_arguments(item_type, item)}
    state.gen_outputs.append(
        _bounded_message({"role": "assistant", "content": _serialize(call), "tool_call_id": str(item_id)})
    )
    state.gen_last_output_ns = at_ns or time.time_ns()


def _record_tool_result(state: TurnState, item_type: str, item: Any, at_ns: Optional[int]) -> None:
    """
    Queue a tool result as input for the next model request.

    Args:
        state (TurnState): The turn state.
        item_type (str): The ThreadItem ``type``.
        item (Any): The completed concrete ThreadItem.
        at_ns (Optional[int]): When the tool finished, in ns since epoch.

    Returns:
        None
    """
    item_id = str(_get(item, "id") or "")
    result = _tool_call_result(item_type, item)
    message = _bounded_message(
        {"role": "tool", "content": "" if result is None else _serialize(result), "tool_call_id": item_id}
    )
    finished_ns = at_ns or time.time_ns()
    state.last_tool_completed_ns = max(state.last_tool_completed_ns or 0, finished_ns)
    if any(output.get("tool_call_id") == item_id for output in state.gen_outputs):
        # The call belongs to the request still open; its result is the next request's input.
        state.pending_tool_results.append(message)
    else:
        # The issuing request was already closed (the tool outlived it).
        state.transcript.append(message)
        if not state.gen_outputs:
            state.gen_start_ns = max(state.gen_start_ns or 0, finished_ns)


def _set_messages(span: Span, prefix: str, key: str, messages: list[dict[str, str]]) -> None:
    """
    Write messages as indexed ``gen_ai.prompt``/``gen_ai.completion`` attributes and as a JSON array.

    Args:
        span (Span): The generation span.
        prefix (str): ``SpanAttributes.LLM_PROMPTS`` or ``SpanAttributes.LLM_COMPLETIONS``.
        key (str): ``"input"`` or ``"output"``.
        messages (list): The messages to record.

    Returns:
        None
    """
    if not messages or not should_send_prompts():
        return
    for index, message in enumerate(messages):
        span.set_attribute(f"{prefix}.{index}.role", message["role"])
        span.set_attribute(f"{prefix}.{index}.content", message["content"])
        if tool_call_id := message.get("tool_call_id"):
            span.set_attribute(f"{prefix}.{index}.tool_call_id", tool_call_id)
    span.set_attribute(key, json.dumps(messages))


def _generation_input(state: TurnState) -> list[dict[str, str]]:
    """
    Return the transcript sent to the model for the current request, bounded in size.

    Args:
        state (TurnState): The turn state.

    Returns:
        list: The first message (the turn input) plus the most recent messages.
    """
    transcript = state.transcript
    if len(transcript) <= MAX_GENERATION_INPUT_MESSAGES:
        return list(transcript)
    return [transcript[0], *transcript[-(MAX_GENERATION_INPUT_MESSAGES - 1) :]]


def _emit_generation(state: TurnState, usage: Any) -> None:
    """
    Emit one GENERATION span for the model request that just finished, then start the next one.

    The span runs from when the request's input was ready (turn start, or the last tool
    result fed back) to the model's last output in the request. Its input is the turn's
    transcript so far and its output is the assistant messages and tool calls produced.

    Args:
        state (TurnState): The turn state.
        usage (Any): The request's ``TokenUsageBreakdown`` (``last``), or None if unknown.

    Returns:
        None
    """
    end_ns = state.gen_last_output_ns or time.time_ns()
    start_ns = min(state.gen_start_ns or end_ns, end_ns)
    state.request_index += 1
    span = None
    try:
        span = state.tracer.start_span(
            GENERATION_SPAN_NAME, context=state.ctx, kind=SpanKind.CLIENT, start_time=start_ns
        )
        _set(span, NETRA_SPAN_TYPE_ATTR, SpanType.GENERATION.value)
        _set(span, SpanAttributes.LLM_SYSTEM, GEN_AI_SYSTEM)
        _set(span, SpanAttributes.LLM_REQUEST_TYPE, LLM_REQUEST_TYPE)
        _set(span, SpanAttributes.LLM_REQUEST_MODEL, state.model)
        _set(span, SpanAttributes.LLM_RESPONSE_MODEL, state.model)
        _set(span, ATTR_THREAD_ID, state.thread_id)
        _set(span, ATTR_TURN_ID, state.turn_id)
        _set(span, ATTR_REQUEST_INDEX, state.request_index)
        _set_messages(span, SpanAttributes.LLM_PROMPTS, "input", _generation_input(state))
        _set_messages(span, SpanAttributes.LLM_COMPLETIONS, "output", state.gen_outputs)
        for name, attr in _USAGE_TOKEN_FIELDS.items():
            value = _get(usage, name)
            if isinstance(value, int) and not isinstance(value, bool):
                span.set_attribute(attr, value)
    except Exception as e:
        logger.error("Cannot record codex generation span: %s", e)
    finally:
        if span is not None:
            span.end(end_time=end_ns)

    state.transcript.extend(state.gen_outputs)
    state.transcript.extend(state.pending_tool_results)
    state.gen_outputs = []
    state.pending_tool_results = []
    state.gen_start_ns = max(end_ns, state.last_tool_completed_ns or 0)
    state.gen_last_output_ns = None


def finish_turn(state: TurnState) -> None:
    """
    Write the turn output, close any open child spans, and end the turn span.

    Safe to call more than once; only the first call has an effect.

    Args:
        state (TurnState): The turn state.

    Returns:
        None
    """
    with state.lock:
        if state.ended:
            return
        state.ended = True
    span = state.span
    try:
        for item_span in state.item_spans.values():
            try:
                item_span.set_attribute(ATTR_TOOL_STATUS, "incomplete")
                item_span.end()
            except Exception:
                pass
        state.item_spans.clear()

        if not state.completed:
            span.set_attribute(ATTR_TURN_STREAM_CLOSED_EARLY, True)
        _set(span, SpanAttributes.LLM_RESPONSE_MODEL, state.model)

        output = state.final_answer if state.final_answer is not None else state.last_message
        if output is not None and should_send_prompts():
            _set_content(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _set_content(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output)
            _set_content(span, "output", json.dumps([{"role": "assistant", "content": output}]))

        # Output produced after the last usage report (e.g. an interrupted request) still
        # gets a generation span, just without usage.
        if state.gen_outputs:
            _emit_generation(state, None)
    except Exception as e:
        logger.error("Failed to finalize codex turn span: %s", e)
    finally:
        try:
            span.end()
        except Exception as e:
            logger.error("Failed to end codex turn span: %s", e)
