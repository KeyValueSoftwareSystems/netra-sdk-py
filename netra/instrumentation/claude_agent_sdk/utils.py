import json
import logging
import threading
from typing import Any

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
from opentelemetry.context import Context
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, Tracer

from netra.config import Config

logger = logging.getLogger(__name__)

# Registry to correlate ToolUseBlocks (AssistantMessage) with ToolResultBlocks (UserMessage).
# Keyed by tool_use_id; entries are removed on consumption.
_tool_call_registry: dict[str, Any] = {}
_tool_call_registry_lock = threading.Lock()


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


def _set_conversation(span: Span, role: str, content: str, prompt_index: int = 0) -> int:
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


def _set_usage(span: Span, usage: dict[str, Any]) -> None:
    """
    Write token usage attributes to the span.

    Args:
        span (Span): The span to write token counts to.
        usage (dict): A dict containing token fields such as input_tokens, output_tokens,
                      total_tokens, cache_creation_input_tokens, and cache_read_input_tokens.

    Returns:
        None
    """
    token_fields = {
        "input_tokens": SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        "output_tokens": SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        "total_tokens": SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        "cache_creation_input_tokens": SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
        "cache_read_input_tokens": SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
    }
    for key, attr in token_fields.items():
        if (value := usage.get(key)) is not None:
            span.set_attribute(attr, value)


def _set_tool_result(tracer: Tracer, parent_ctx: Context, block: ToolResultBlock) -> None:
    """
    Create a child span for a tool result, correlating it with the originating tool call.

    Args:
        tracer (Tracer): The OpenTelemetry tracer used to create the child span.
        parent_ctx (Context): The parent span context to attach the child span to.
        block (ToolResultBlock): The tool result block containing the output content
                                  and the tool_use_id for correlation lookup.

    Returns:
        None
    """
    try:
        tool_call = None
        try:
            with _tool_call_registry_lock:
                tool_call = _tool_call_registry.pop(block.tool_use_id, None)
        except Exception as e:
            logger.error(f"Cannot extract tool call metadata for tool_use_id={block.tool_use_id}: {e}")

        tool_name = tool_call["name"] if tool_call else "unrecognized_tool"

        with tracer.start_as_current_span(tool_name, parent_ctx) as span:
            try:
                if tool_call:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.input", _serialize(tool_call.get("input", {})))

                span.set_attribute(f"{Config.LIBRARY_NAME}.span.type", "TOOL")
                span.set_attribute(f"output", _serialize(block.content))

            except Exception as e:
                logger.error(f"Cannot set tool result attributes for tool_use_id={block.tool_use_id}: {e}")

    except Exception as e:
        logger.error(f"Error creating tool result span: {e}")


def set_request_attributes(span: Span, prompt: Any, options: ClaudeAgentOptions | None) -> int:
    """
    Write request metadata (model, system prompt, user prompt) to the root span.

    Args:
        span (Span): The root OpenTelemetry span to write attributes to.
        prompt (Any): The user prompt string for the request.
        options (ClaudeAgentOptions | None): Agent options containing model and system prompt.

    Returns:
        int: The next available prompt_index so callers can continue indexing result messages without gaps.
    """
    prompt_index = 0

    try:
        if options and isinstance(options, ClaudeAgentOptions):
            if model := options.model:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
            if system_prompt := options.system_prompt:
                prompt_index = _set_conversation(span, "system", system_prompt, prompt_index)
    except Exception as e:
        logger.error(f"Cannot extract options from request: {e}")

    try:
        if prompt and isinstance(prompt, str):
            prompt_index = _set_conversation(span, "user", prompt, prompt_index)
    except Exception as e:
        logger.error(f"Cannot extract prompt from request: {e}")

    return prompt_index


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
        if model := message.data.get("model"):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    except Exception as e:
        logger.error(f"Cannot extract model from SystemMessage: {e}")


def set_assistant_message_attributes(tracer: Tracer, parent_ctx: Context, message: AssistantMessage) -> None:
    """
    Create child spans for each content block in an AssistantMessage.

    Handles TextBlock, ThinkingBlock, and ToolUseBlock content. ToolUseBlocks are
    registered in the tool call registry for later correlation with their tool results.

    Args:
        tracer (Tracer): The OpenTelemetry tracer used to create child spans.
        parent_ctx (Context): The parent span context to attach child spans to.
        message (AssistantMessage): The assistant message containing one or more content blocks.

    Returns:
        None
    """
    for block in message.content:
        try:
            role, content, span_name = None, None, None

            if isinstance(block, TextBlock):
                role, span_name, content = "assistant", "claude-agent.assistant", block.text
            elif isinstance(block, ThinkingBlock):
                role, span_name, content = "assistant", "claude-agent.thinking", block.thinking
            elif isinstance(block, ToolUseBlock):
                with _tool_call_registry_lock:
                    _tool_call_registry[block.id] = {"name": block.name, "input": block.input}

            if role and content and span_name:
                with tracer.start_as_current_span(span_name, parent_ctx) as span:
                    if message.model:
                        span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, message.model)
                    _set_conversation(span, role, content)
        except Exception as e:
            logger.error(f"Cannot process assistant message block: {e}")


def set_user_message_attributes(tracer: Tracer, parent_ctx: Context, message: UserMessage) -> None:
    """
    Create a child tool-result span for each ToolResultBlock in a UserMessage.

    Args:
        tracer (Tracer): The OpenTelemetry tracer used to create child spans.
        parent_ctx (Context): The parent span context to attach child spans to.
        message (UserMessage): The user message containing one or more content blocks.

    Returns:
        None
    """
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            _set_tool_result(tracer, parent_ctx, block)


def set_result_message_attributes(span: Span, message: ResultMessage, prompt_index: int = 0) -> None:
    """
    Write the final result text and token usage to the root span.

    Args:
        span (Span): The root OpenTelemetry span to write attributes to.
        message (ResultMessage): The result message containing the final text and usage data.
        prompt_index (int): The current prompt index for recording the result conversation. Defaults to 0.

    Returns:
        None
    """
    if result := message.result:
        _set_conversation(span, "assistant", result, prompt_index)

    if usage := message.usage:
        _set_usage(span, usage)
