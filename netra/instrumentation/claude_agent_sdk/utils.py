import json
import logging
import threading
from typing import Any
from opentelemetry.context import Context
from opentelemetry.trace import Span, Tracer
from opentelemetry.semconv_ai import SpanAttributes
from claude_agent_sdk import (
    ClaudeAgentOptions,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from netra.config import Config

logger = logging.getLogger(__name__)

# Registry to correlate ToolUseBlocks (AssistantMessage) with ToolResultBlocks (UserMessage).
# Keyed by tool_use_id; entries are removed on consumption.
_tool_call_registry: dict[str, Any] = {}
_tool_call_registry_lock = threading.Lock()


def _serialize(value: Any) -> str:
    """Serialize a value to a span-safe string; JSON for dicts/lists, plain string otherwise."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _set_conversation(span: Span, role: str, content: str, prompt_index: int = 0) -> int:
    """Write a single prompt entry to the span and return the incremented index."""
    if role and content:
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", role)
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content", content)
        prompt_index += 1
    return prompt_index


def _set_usage(span: Span, usage: dict) -> None:
    """Write token usage attributes to the span."""
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
    """Create a child span for each tool result, correlating with the originating tool call."""
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
                span.set_attribute(f"{Config.LIBRARY_NAME}.entity.output", _serialize(block.content))

            except Exception as e:
                logger.error(f"Cannot set tool result attributes for tool_use_id={block.tool_use_id}: {e}")

    except Exception as e:
        logger.error(f"Error creating tool result span: {e}")


def set_request_attributes(span: Span, prompt: Any, options: ClaudeAgentOptions | None) -> int:
    """
    Write request metadata to the root span.

    Returns the next available prompt_index so callers can continue indexing result messages without gaps
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
    """Write model info from a SystemMessage to the root span."""
    try:
        if model := message.data.get("model"):
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    except Exception as e:
        logger.error(f"Cannot extract model from SystemMessage: {e}")


def set_assistant_message_attributes(tracer: Tracer, parent_ctx: Context, message: AssistantMessage) -> None:
    """
    Create child spans for each content block in an AssistantMessage.

    ToolUseBlocks are registered for later correlation with their tool results.
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
    """Create a child tool-result span for each ToolResultBlock in a UserMessage."""
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            _set_tool_result(tracer, parent_ctx, block)


def set_result_message_attributes(span: Span, message: ResultMessage, prompt_index: int = 0) -> None:
    """Write the final result text and token usage to the root span."""
    if result := message.result:
        _set_conversation(span, "assistant", result, prompt_index)

    if usage := message.usage:
        _set_usage(span, usage)
