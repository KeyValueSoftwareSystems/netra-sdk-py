import json
import logging
import threading
import time
from typing import Any, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer
from opentelemetry.trace.status import Status

from netra.config import Config
from netra.instrumentation.utils import record_span_timing
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"

# Registry correlating ToolUseBlocks with their open spans and ToolResultBlocks.
# Each entry holds the open span (ended when the result arrives) and the span context
# (used to parent subagent messages under the tool call that spawned them).
# Keyed by tool_use_id; entries are removed on consumption.
_tool_call_registry: dict[str, Any] = {}
_tool_call_registry_lock = threading.Lock()

# Custom attribute keys using the gen_ai.claude_code namespace
ATTR_SESSION_ID = "gen_ai.session.id"
ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_PARENT_TOOL_USE_ID = "gen_ai.parent_tool_use_id"

ATTR_CLAUDE_CODE_VERSION = "gen_ai.claude_code.version"
ATTR_CLAUDE_CODE_CWD = "gen_ai.claude_code.cwd"
ATTR_CLAUDE_CODE_PERMISSION_MODE = "gen_ai.claude_code.permission_mode"
ATTR_CLAUDE_CODE_AVAILABLE_TOOLS = "gen_ai.claude_code.available_tools"
ATTR_CLAUDE_CODE_MCP_SERVERS = "gen_ai.claude_code.mcp_servers"

ATTR_CLAUDE_CODE_NUM_TURNS = "gen_ai.claude_code.num_turns"
ATTR_CLAUDE_CODE_WEB_SEARCH_REQUESTS = "gen_ai.claude_code.web_search_requests"
ATTR_CLAUDE_CODE_WEB_FETCH_REQUESTS = "gen_ai.claude_code.web_fetch_requests"
ATTR_CLAUDE_CODE_PERMISSION_DENIALS = "gen_ai.claude_code.permission_denials"
ATTR_CLAUDE_CODE_ERRORS = "gen_ai.claude_code.errors"
ATTR_CLAUDE_CODE_STRUCTURED_OUTPUT = "gen_ai.claude_code.structured_output"

ATTR_CLAUDE_CODE_MAX_TURNS = "gen_ai.claude_code.max_turns"
ATTR_CLAUDE_CODE_CONTINUE_CONVERSATION = "gen_ai.claude_code.continue_conversation"


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
        role (str): The conversation role (e.g. "user", "assistant", "system", "tool").
        content (str): The message content to include.

    Returns:
        str: A JSON string of the form ``[{"role": role, "content": content}]``.
    """
    return json.dumps([{"role": role, "content": content}])


def _set_input_conversation(span: Span, role: str, content: str, prompt_index: int = 0) -> int:
    """
    Write a single prompt entry to the span at the given index.

    Args:
        span (Span): The OpenTelemetry span to write attributes to.
        role (str): The conversation role (e.g. "user", "assistant", "system").
        content (str): The message content to record.
        prompt_index (int): The current index in the prompts attribute list. Defaults to 0.

    Returns:
        int: The incremented prompt index after writing.
    """
    if role and content:
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", role)
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content", content)
        prompt_index += 1
    return prompt_index


def _set_output_conversation(span: Span, role: str, content: str, completion_index: int = 0) -> int:
    """
    Write a single completion entry to the span at the given index.

    Args:
        span (Span): The OpenTelemetry span to write attributes to.
        role (str): The conversation role (e.g. "user", "assistant", "system").
        content (str): The message content to record.
        completion_index (int): The current index in the completions attribute list. Defaults to 0.

    Returns:
        int: The incremented completion index after writing.
    """
    if role and content:
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{completion_index}.role", role)
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{completion_index}.content", content)
        completion_index += 1
    return completion_index


_NETRA_SPAN_TYPE_ATTR = "netra.span.type"
_MODEL_USAGE_SPAN_PREFIX = "claude-agent.usage"

_USAGE_TOKEN_FIELDS: dict[str, str] = {
    "input_tokens": SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    "output_tokens": SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    "total_tokens": SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
    "cache_creation_input_tokens": SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
    "cache_read_input_tokens": SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
}

_MODEL_USAGE_TOKEN_FIELDS: dict[str, str] = {
    "inputTokens": SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    "outputTokens": SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    "cacheReadInputTokens": SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    "cacheCreationInputTokens": SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
    "webSearchRequests": ATTR_CLAUDE_CODE_WEB_SEARCH_REQUESTS,
}


def _create_model_usage_spans(tracer: Tracer, root_span: Span, model_usage: dict[str, Any]) -> None:
    """
    Create one child span per model entry in the model_usage dict.

    Args:
        tracer: The OpenTelemetry tracer used to create the child spans.
        root_span: The root span; usage spans are created as its direct children.
        model_usage: A dict mapping model name to a per-model usage dict whose
                     camelCase keys are mapped to span attributes via
                     ``_MODEL_USAGE_TOKEN_FIELDS``.

    Returns:
        None
    """
    parent_ctx = trace.set_span_in_context(root_span)
    for model_name, usage_dict in model_usage.items():
        if not isinstance(usage_dict, dict):
            continue
        span = None
        try:
            span = tracer.start_span(f"{_MODEL_USAGE_SPAN_PREFIX}.{model_name}", context=parent_ctx)
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model_name)
            span.set_attribute(_NETRA_SPAN_TYPE_ATTR, SpanType.USAGE)
            for field, attr in _MODEL_USAGE_TOKEN_FIELDS.items():
                if (value := usage_dict.get(field)) is not None:
                    span.set_attribute(attr, value)
        except Exception as e:
            logger.error(f"Cannot create usage span for model={model_name}: {e}")
        finally:
            if span is not None:
                span.end()


def _set_usage(span: Span, usage: dict[str, Any]) -> None:
    """
    Write token usage attributes to the span.

    Args:
        span (Span): The span to write token counts to.
        usage (dict): A dict containing token fields such as input_tokens, output_tokens etc.

    Returns:
        None
    """
    for key, attr in _USAGE_TOKEN_FIELDS.items():
        if (value := usage.get(key)) is not None:
            span.set_attribute(attr, value)

    if server_tool_use := usage.get("server_tool_use"):
        if isinstance(server_tool_use, dict):
            if (v := server_tool_use.get("web_search_requests")) is not None:
                span.set_attribute(ATTR_CLAUDE_CODE_WEB_SEARCH_REQUESTS, v)
            if (v := server_tool_use.get("web_fetch_requests")) is not None:
                span.set_attribute(ATTR_CLAUDE_CODE_WEB_FETCH_REQUESTS, v)


def _start_tool_call_span(
    tracer: Tracer,
    parent_ctx: Context,
    block: ToolUseBlock,
    message: AssistantMessage,
    first_token_time: Optional[float] = None,
    start_time: Optional[float] = None,
) -> None:
    """
    Open a child span for a ToolUseBlock and store it in the registry for later correlation.

    The span is left open; it will be ended by ``_end_tool_call_span`` when the
    corresponding ToolResultBlock arrives.  The span context is also stored so that
    subagent messages (those whose ``parent_tool_use_id`` matches this tool_use_id)
    can be parented under this span rather than the root span.

    Args:
        tracer (Tracer): The OpenTelemetry tracer used to create the child span.
        parent_ctx (Context): The parent span context to attach the tool call span to.
        block (ToolUseBlock): The tool use block describing the call (id, name, input).
        message (AssistantMessage): The assistant message that contains this block;
                                    used to copy model and session metadata onto the span.
        first_token_time (Optional[float]): Time when the containing AssistantMessage was
                                            received. Used to record TIME_TO_FIRST_TOKEN.
                                            Defaults to now.
        start_time (Optional[float]): Time when the dispatch loop started. Used as
                                      reference for TIME_TO_FIRST_TOKEN.

    Returns:
        None
    """
    tool_span = None
    try:
        tool_span = tracer.start_span(block.name, context=parent_ctx, kind=SpanKind.CLIENT)
        tool_ctx = trace.set_span_in_context(tool_span)
        try:
            tool_input = _serialize(block.input)
            tool_span.set_attribute(f"{Config.LIBRARY_NAME}.span.type", "TOOL")
            tool_span.set_attribute("gen_ai.tool.name", block.name)
            tool_span.set_attribute("gen_ai.tool.call.id", block.id)
            tool_span.set_attribute("input", _build_message_array("assistant", tool_input))
            if message.model:
                tool_span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, message.model)
            if message.session_id:
                tool_span.set_attribute(ATTR_SESSION_ID, message.session_id)
            if message.parent_tool_use_id:
                tool_span.set_attribute(ATTR_PARENT_TOOL_USE_ID, message.parent_tool_use_id)
            token_time = first_token_time if first_token_time is not None else time.time()
            record_span_timing(
                tool_span, TIME_TO_FIRST_TOKEN, token_time, reference_time=start_time, record_event_timestamp=True
            )
            record_span_timing(tool_span, RELATIVE_TIME_TO_FIRST_TOKEN, token_time, use_root_span=True)
        except Exception as e:
            logger.error(f"Cannot set tool call span attributes for tool={block.name}: {e}")

        with _tool_call_registry_lock:
            _tool_call_registry[block.id] = {"name": block.name, "span": tool_span, "ctx": tool_ctx}
    except Exception as e:
        if tool_span is not None:
            tool_span.end()
        logger.error(f"Error creating tool call span for tool={block.name}: {e}")


def _end_tool_call_span(block: ToolResultBlock) -> None:
    """
    Finalize the open tool call span using the result from a ToolResultBlock.

    Looks up the span opened by ``_start_tool_call_span`` via the tool_use_id,
    sets the output attribute in the standard array format, marks the span as
    errored if the result is an error, and then ends the span.

    Args:
        block (ToolResultBlock): The tool result block containing the output content,
                                  the tool_use_id for registry lookup, and an error flag.

    Returns:
        None
    """
    try:
        tool_entry = None
        with _tool_call_registry_lock:
            tool_entry = _tool_call_registry.pop(block.tool_use_id, None)

        if tool_entry is None:
            logger.warning(f"No open tool call span found for tool_use_id={block.tool_use_id}")
            return

        span = tool_entry["span"]
        try:
            result_content = _serialize(block.content)
            span.set_attribute("output", _build_message_array("tool", result_content))
            _set_output_conversation(span, "tool", result_content)

            if block.is_error:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("tool.is_error", True)
        except Exception as e:
            logger.error(f"Cannot set tool result attributes for tool_use_id={block.tool_use_id}: {e}")
        finally:
            span.end()
    except Exception as e:
        logger.error(f"Error finalizing tool call span for tool_use_id={block.tool_use_id}: {e}")


def set_request_attributes(span: Span, prompt: Any, options: ClaudeAgentOptions | None) -> None:
    """
    Write request metadata (model, system prompt, user prompt) to the root span.

    Populates both the structured ``gen_ai.prompts.*`` attributes and the ``input``
    attribute as a JSON array of ``{"role", "content"}`` objects.

    Args:
        span (Span): The root OpenTelemetry span to write attributes to.
        prompt (Any): The user prompt string for the request.
        options (ClaudeAgentOptions | None): Agent options containing model, system prompt,
                                             and execution settings.

    Returns:
        None
    """
    prompt_index = 0
    input_messages: list[dict[str, str]] = []

    try:
        if options and isinstance(options, ClaudeAgentOptions):
            if model := options.model:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
            if system_prompt := options.system_prompt:
                if isinstance(system_prompt, str):
                    prompt_index = _set_input_conversation(span, "system", system_prompt, prompt_index)
                    input_messages.append({"role": "system", "content": system_prompt})
                elif isinstance(system_prompt, dict):
                    try:
                        serialized = json.dumps(system_prompt)
                    except Exception:
                        serialized = str(system_prompt)
                    prompt_index = _set_input_conversation(span, "system", serialized, prompt_index)
                    input_messages.append({"role": "system", "content": serialized})

            if options.permission_mode:
                span.set_attribute(ATTR_CLAUDE_CODE_PERMISSION_MODE, options.permission_mode)
            if options.max_turns is not None:
                span.set_attribute(ATTR_CLAUDE_CODE_MAX_TURNS, options.max_turns)
            if options.cwd is not None:
                span.set_attribute(ATTR_CLAUDE_CODE_CWD, str(options.cwd))
            if options.session_id:
                span.set_attribute(ATTR_SESSION_ID, options.session_id)
            if options.continue_conversation is not None:
                span.set_attribute(ATTR_CLAUDE_CODE_CONTINUE_CONVERSATION, options.continue_conversation)
    except Exception as e:
        logger.error(f"Cannot extract options from request: {e}")

    try:
        if prompt and isinstance(prompt, str):
            _set_input_conversation(span, "user", prompt, prompt_index)
            input_messages.append({"role": "user", "content": prompt})
    except Exception as e:
        logger.error(f"Cannot extract prompt from request: {e}")

    if input_messages:
        span.set_attribute("input", json.dumps(input_messages))


def set_system_message_attributes(span: Span, message: SystemMessage) -> None:
    """
    Write model info from a SystemMessage to the root span.

    Args:
        span (Span): The root OpenTelemetry span to write attributes to.
        message (SystemMessage): The system message containing model metadata.

    Returns:
        None
    """
    try:
        if (data := message.data) is None:
            return

        if model := data.get("model"):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
        if session_id := data.get("session_id"):
            span.set_attribute(ATTR_SESSION_ID, session_id)
        if cwd := data.get("cwd"):
            span.set_attribute(ATTR_CLAUDE_CODE_CWD, cwd)
        if permission_mode := data.get("permissionMode"):
            span.set_attribute(ATTR_CLAUDE_CODE_PERMISSION_MODE, permission_mode)
        if version := data.get("claude_code_version"):
            span.set_attribute(ATTR_CLAUDE_CODE_VERSION, version)
        if tools := data.get("tools"):
            span.set_attribute(ATTR_CLAUDE_CODE_AVAILABLE_TOOLS, _serialize(tools))
        if mcp_servers := data.get("mcp_servers"):
            span.set_attribute(ATTR_CLAUDE_CODE_MCP_SERVERS, _serialize(mcp_servers))
    except Exception as e:
        logger.error(f"Cannot extract attributes from SystemMessage: {e}")


def set_assistant_message_attributes(
    tracer: Tracer,
    parent_ctx: Context,
    message: AssistantMessage,
    first_token_time: Optional[float] = None,
    start_time: Optional[float] = None,
) -> None:
    """
    Create child spans for each content block in an AssistantMessage.

    Handles three block types:

    - ``TextBlock`` — creates a ``claude-agent.assistant`` child span.
    - ``ThinkingBlock`` — creates a ``claude-agent.thinking`` child span.
    - ``ToolUseBlock`` — opens a span named after the tool and stores it in the
      tool call registry; the span is ended when the corresponding ToolResultBlock
      arrives via ``set_user_message_attributes``.

    When ``message.parent_tool_use_id`` is set the message originates from a subagent.
    In that case TextBlock and ThinkingBlock spans are parented under the open tool call
    span (looked up from the registry) rather than the root span, preserving the
    subagent call hierarchy in the trace.

    Args:
        tracer (Tracer): The OpenTelemetry tracer used to create child spans.
        parent_ctx (Context): The root span context; used as fallback parent when no
                              subagent tool call context is available.
        message (AssistantMessage): The assistant message containing one or more content blocks.
        first_token_time (Optional[float]): Time (seconds since epoch) when this message was
                                            received. Defaults to ``time.time()``.
        start_time (Optional[float]): Time (seconds since epoch) when the dispatch loop started.
                                      Used as the reference for TTFT.

    Returns:
        None
    """
    # Resolve parent context: subagent messages should nest under the tool call that spawned them.
    effective_ctx = parent_ctx
    if message.parent_tool_use_id:
        with _tool_call_registry_lock:
            tool_entry = _tool_call_registry.get(message.parent_tool_use_id)
        if tool_entry and "ctx" in tool_entry:
            effective_ctx = tool_entry["ctx"]

    token_time = first_token_time if first_token_time is not None else time.time()

    for block in message.content:
        try:
            if isinstance(block, ToolUseBlock):
                _start_tool_call_span(tracer, effective_ctx, block, message, token_time, start_time)
                continue

            role, content, span_name = None, None, None
            if isinstance(block, TextBlock):
                role, span_name, content = "assistant", "claude-agent.assistant", block.text
            elif isinstance(block, ThinkingBlock):
                role, span_name, content = "assistant", "claude-agent.thinking", block.thinking

            if not (role and content and span_name):
                continue

            with tracer.start_as_current_span(span_name, effective_ctx) as span:
                try:
                    if message.model:
                        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, message.model)
                    if message.message_id:
                        span.set_attribute(ATTR_RESPONSE_ID, message.message_id)
                    if message.session_id:
                        span.set_attribute(ATTR_SESSION_ID, message.session_id)
                    if message.stop_reason:
                        span.set_attribute(SpanAttributes.LLM_RESPONSE_STOP_REASON, message.stop_reason)
                    if message.parent_tool_use_id:
                        span.set_attribute(ATTR_PARENT_TOOL_USE_ID, message.parent_tool_use_id)
                    if message.error:
                        span.set_status(Status(StatusCode.ERROR, str(message.error)))
                        span.set_attribute("gen_ai.error", str(message.error))
                    if message.usage and isinstance(message.usage, dict):
                        _set_usage(span, message.usage)
                    _set_output_conversation(span, role, content)
                    span.set_attribute("output", _build_message_array(role, content))
                    record_span_timing(
                        span, TIME_TO_FIRST_TOKEN, token_time, reference_time=start_time, record_event_timestamp=True
                    )
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, token_time, use_root_span=True)
                except Exception as e:
                    logger.error(f"Cannot set assistant span attributes: {e}")
        except Exception as e:
            logger.error(f"Cannot process assistant message block: {e}")


def set_user_message_attributes(tracer: Tracer, parent_ctx: Context, message: UserMessage) -> None:
    """
    Finalize open tool call spans for each ToolResultBlock in a UserMessage.

    Each ToolResultBlock is correlated with its originating ToolUseBlock via the
    tool_use_id.  The open tool call span (created by ``set_assistant_message_attributes``)
    is retrieved from the registry, populated with the tool result output, and ended.

    Args:
        tracer (Tracer): The OpenTelemetry tracer (unused directly; kept for interface
                         consistency with other message handlers).
        parent_ctx (Context): The parent span context (unused directly; tool spans were
                              already parented when opened).
        message (UserMessage): The user message containing one or more content blocks.

    Returns:
        None
    """
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            _end_tool_call_span(block)


def set_result_message_attributes(
    tracer: Tracer,
    span: Span,
    message: ResultMessage,
) -> None:
    """
    Write the final result text and token usage to the root span.

    When the message contains a ``model_usage`` dict, per-model usage child
    spans are created via ``_create_model_usage_spans``.  Otherwise, aggregated
    flat usage from ``message.usage`` is written directly to the root span as
    a fallback.

    Args:
        tracer: The OpenTelemetry tracer used to create per-model usage child spans.
        span: The root OpenTelemetry span to write result attributes to.
        message: The result message containing the final text, structured output,
                 usage data, and execution statistics.

    Returns:
        None
    """
    if result := message.result:
        _set_output_conversation(span, "assistant", result)

    # structured_output is the canonical LLM response when present; fall back to result.
    if message.structured_output is not None:
        span.set_attribute("output", _build_message_array("assistant", _serialize(message.structured_output)))
    elif message.result:
        span.set_attribute("output", _build_message_array("assistant", message.result))

    try:
        if message.num_turns is not None:
            span.set_attribute(ATTR_CLAUDE_CODE_NUM_TURNS, message.num_turns)
        if message.session_id:
            span.set_attribute(ATTR_SESSION_ID, message.session_id)

        if message.stop_reason:
            span.set_attribute(SpanAttributes.LLM_RESPONSE_STOP_REASON, message.stop_reason)
        if message.is_error:
            span.set_status(Status(StatusCode.ERROR))
        if message.structured_output is not None:
            span.set_attribute(ATTR_CLAUDE_CODE_STRUCTURED_OUTPUT, _serialize(message.structured_output))
        if message.permission_denials:
            span.set_attribute(ATTR_CLAUDE_CODE_PERMISSION_DENIALS, _serialize(message.permission_denials))
        if message.errors:
            span.set_attribute(ATTR_CLAUDE_CODE_ERRORS, _serialize(message.errors))
    except Exception as e:
        logger.error(f"Cannot set result message base attributes: {e}")

    if model_usage := getattr(message, "model_usage", None):
        _create_model_usage_spans(tracer, span, model_usage)
    elif usage := getattr(message, "usage", None):
        _set_usage(span, usage)
