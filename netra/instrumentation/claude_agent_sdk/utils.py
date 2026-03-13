import json
import logging
from typing import Any, Dict
from opentelemetry.trace import Span
from opentelemetry.semconv_ai import SpanAttributes
from claude_agent_sdk import (
    ClaudeAgentOptions,
    UserMessage,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock
)

from netra import Netra
from netra.session_manager import ConversationType

logger = logging.getLogger(__name__)

def set_request_attributes(span: Span, kwargs: Dict[str, Any], prompt_index: int) -> None:
    try:
        options = kwargs.get("options")
        if options and isinstance(options, ClaudeAgentOptions):
            if model := options.model:
                span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, model)
    except Exception:
        logger.error(f"Cannot extract model from request")
    
    try:
        prompt = kwargs.get("prompt", "")
        if prompt:
            prompt_index = _set_conversation(span, ConversationType.INPUT, "user", prompt, prompt_index)
    except Exception:
        logger.error(f"Cannot extract prompt from request")
        
    return prompt_index

def set_response_message_attributes(span: Span, message: Any, prompt_index: int):
    try:
        if message.model:
            span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, message.model)
    except Exception as e:
        pass

    try:
        if isinstance(message, AssistantMessage):
            prompt_index = _set_assistant_message_attributes(span, message, prompt_index)

        elif isinstance(message, UserMessage):
            prompt_index = _set_user_message_attributes(span, message, prompt_index)

        elif isinstance(message, ResultMessage):
            _set_result_message_attributes(span, message)
        
    except Exception as e:
        logger.error(f"Cannot extract data from message", e)

    return prompt_index

def _set_conversation(
    span: Span, 
    conversationType: ConversationType, 
    role: str, 
    content: str, 
    prompt_index: int = 0
):
    if content and role:
        Netra.add_conversation(conversationType, role, content)
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", role)
        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content", content)
        prompt_index += 1
    return prompt_index

def _set_user_message_attributes(span: Span, message: UserMessage, prompt_index: int):
    for block in message.content:
        if isinstance(block, ToolResultBlock):
            prompt_index = _set_conversation(span, ConversationType.INPUT, "tool_result", block.content, prompt_index) 
    return prompt_index

def _set_assistant_message_attributes(span: Span, message: AssistantMessage, prompt_index: int):
    role, content = None, None
    for block in message.content:
        if isinstance(block, TextBlock):
            role = "assistant"
            content = block.text
        elif isinstance(block, ThinkingBlock):
            role = "assistant"
            content = block.thinking
        elif isinstance(block, ToolUseBlock):
            role = "tool"
            content = f"Tool `{block.name}` invoked using attributes\n{json.dumps(block.input, indent=2)}"
        prompt_index = _set_conversation(span, ConversationType.OUTPUT, role, content, prompt_index)
            
    return prompt_index

def _set_result_message_attributes(span: Span, message: ResultMessage):
    if not message.usage:
        return

    usage = message.usage
    
    if (input_tokens := usage.get("input_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)

    if (output_tokens := usage.get("output_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)

    if (total_tokens := usage.get("total_tokens")) is not None:
        span.set_attribute(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    if (cache_creation_input_tokens := usage.get("cache_creation_input_tokens")) is not None:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS,
            cache_creation_input_tokens,
        )

    if (cache_read_input_tokens := usage.get("cache_read_input_tokens")) is not None:
        span.set_attribute(
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            cache_read_input_tokens,
        )
