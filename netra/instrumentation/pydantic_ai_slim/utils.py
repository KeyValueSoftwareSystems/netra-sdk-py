from typing import Any, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

# Constants for consistent truncation and limits
MAX_CONTENT_LENGTH = 1000
MAX_ARGS_LENGTH = 500
MAX_ITEMS_TO_PROCESS = 5
MAX_FUNCTIONS_TO_PROCESS = 3


def _safe_set_attribute(span: Any, key: str, value: Any, max_length: Optional[int] = None) -> None:
    """Safely set span attribute with optional truncation and null checks."""
    if not span.is_recording() or value is None:
        return

    str_value = str(value)
    if max_length and len(str_value) > max_length:
        str_value = str_value[:max_length]

    span.set_attribute(key, str_value)


def _safe_get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get attribute from object with default fallback."""
    return getattr(obj, attr_name, default) if hasattr(obj, attr_name) else default


def _handle_span_error(span: Any, exception: Exception) -> None:
    """Common error handling for spans."""
    span.set_status(Status(StatusCode.ERROR, str(exception)))
    _safe_set_attribute(span, "error.type", type(exception).__name__)
    _safe_set_attribute(span, "error.message", str(exception))


def _set_timing_attributes(span: Any, start_time: float, end_time: float) -> None:
    """Set timing attributes on span."""
    duration_ms = (end_time - start_time) * 1000
    _safe_set_attribute(span, "llm.response.duration", duration_ms)


def _set_assistant_response_content(span: Any, result: Any, finish_reason: str = "completed") -> None:
    """Set assistant response content in OpenAI wrapper format."""
    if not span.is_recording():
        return

    # Set the assistant response in the same format as OpenAI wrapper
    index = 0  # Always use index 0 for pydantic_ai responses
    _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", "assistant")
    _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason", finish_reason)

    # Get the output content from the result
    output = _safe_get_attribute(result, "output")
    if output is not None:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", output, MAX_CONTENT_LENGTH)


def set_pydantic_request_attributes(
    span: Any,
    kwargs: Dict[str, Any],
    operation_type: str,
    model_name: Optional[str] = None,
    include_model: bool = False,
) -> None:
    """Set request attributes on span for pydantic_ai."""
    if not span.is_recording():
        return

    # Set operation type
    _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_TYPE}", operation_type)

    # Set model only if explicitly requested (for CallToolsNode spans)
    if include_model and model_name:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_MODEL}", model_name)

    # Set temperature and max_tokens if available
    _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", kwargs.get("temperature"))
    _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs.get("max_tokens"))


def set_pydantic_response_attributes(span: Any, result: Any) -> None:
    """Set response attributes on span for pydantic_ai."""
    if not span.is_recording():
        return

    # Set response model if available
    model_name = _safe_get_attribute(result, "model_name")
    _safe_set_attribute(span, f"{SpanAttributes.LLM_RESPONSE_MODEL}", model_name)

    # Set usage information
    if hasattr(result, "usage"):
        usage = result.usage()
        if usage:
            _safe_set_attribute(
                span, f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", _safe_get_attribute(usage, "request_tokens")
            )
            _safe_set_attribute(
                span, f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", _safe_get_attribute(usage, "response_tokens")
            )
            _safe_set_attribute(
                span, f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", _safe_get_attribute(usage, "total_tokens")
            )

            # Set any additional details from usage
            details = _safe_get_attribute(usage, "details")
            if details:
                for key, value in details.items():
                    if value:
                        _safe_set_attribute(span, f"gen_ai.usage.details.{key}", value)

    # Set output content if available
    output = _safe_get_attribute(result, "output")
    if output is not None:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def get_node_span_name(node: Any) -> str:
    """Get appropriate span name for a node type"""
    node_type = type(node).__name__
    if "UserPrompt" in node_type:
        return "pydantic_ai.node.user_prompt"
    elif "ModelRequest" in node_type:
        return "pydantic_ai.node.model_request"
    elif "CallTools" in node_type:
        return "pydantic_ai.node.call_tools"
    elif "End" in node_type:
        return "pydantic_ai.node.end"
    else:
        return f"pydantic_ai.node.{node_type.lower()}"


def set_node_attributes(span: Any, node: Any) -> None:
    """Set attributes on span based on node type and content"""
    if not span.is_recording():
        return

    node_type = type(node).__name__
    _safe_set_attribute(span, "pydantic_ai.node.type", node_type)

    # UserPromptNode attributes
    if "UserPrompt" in node_type:
        _set_user_prompt_node_attributes(span, node)

    # ModelRequestNode attributes
    elif "ModelRequest" in node_type:
        _set_model_request_node_attributes(span, node)

    # CallToolsNode attributes
    elif "CallTools" in node_type:
        _set_call_tools_node_attributes(span, node)

    # End node attributes
    elif "End" in node_type:
        _set_end_node_attributes(span, node)

    # Generic node attributes for any other node types
    else:
        _set_generic_node_attributes(span, node)


def _set_user_prompt_node_attributes(span: Any, node: Any) -> None:
    """Set attributes specific to UserPromptNode."""
    # User prompt content
    user_prompt = _safe_get_attribute(node, "user_prompt")
    if user_prompt:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", user_prompt, MAX_CONTENT_LENGTH)
        _safe_set_attribute(span, "pydantic_ai.user_prompt", user_prompt, MAX_CONTENT_LENGTH)

    # Instructions
    instructions = _safe_get_attribute(node, "instructions")
    _safe_set_attribute(span, "pydantic_ai.instructions", instructions, MAX_CONTENT_LENGTH)

    # Instructions functions
    instructions_functions = _safe_get_attribute(node, "instructions_functions")
    if instructions_functions:
        _safe_set_attribute(span, "pydantic_ai.instructions_functions_count", len(instructions_functions))
        for i, func in enumerate(instructions_functions[:MAX_FUNCTIONS_TO_PROCESS]):
            func_name = _safe_get_attribute(func, "__name__")
            _safe_set_attribute(span, f"pydantic_ai.instructions_functions.{i}.name", func_name)

    # System prompts
    system_prompts = _safe_get_attribute(node, "system_prompts")
    if system_prompts:
        _safe_set_attribute(span, "pydantic_ai.system_prompts_count", len(system_prompts))
        for i, prompt in enumerate(system_prompts[:MAX_FUNCTIONS_TO_PROCESS]):
            _safe_set_attribute(span, f"pydantic_ai.system_prompts.{i}", prompt, MAX_ARGS_LENGTH)

    # System prompt functions
    system_prompt_functions = _safe_get_attribute(node, "system_prompt_functions")
    if system_prompt_functions:
        _safe_set_attribute(span, "pydantic_ai.system_prompt_functions_count", len(system_prompt_functions))

    # System prompt dynamic functions
    system_prompt_dynamic_functions = _safe_get_attribute(node, "system_prompt_dynamic_functions")
    if system_prompt_dynamic_functions:
        _safe_set_attribute(
            span, "pydantic_ai.system_prompt_dynamic_functions_count", len(system_prompt_dynamic_functions)
        )
        for key in list(system_prompt_dynamic_functions.keys())[:MAX_FUNCTIONS_TO_PROCESS]:
            func_type = type(system_prompt_dynamic_functions[key]).__name__
            _safe_set_attribute(span, f"pydantic_ai.system_prompt_dynamic_functions.{key}", func_type)


def _set_model_request_node_attributes(span: Any, node: Any) -> None:
    """Set attributes specific to ModelRequestNode."""
    request = _safe_get_attribute(node, "request")
    if not request:
        return

    # Request parts
    parts = _safe_get_attribute(request, "parts")
    if parts:
        _safe_set_attribute(span, "pydantic_ai.request.parts_count", len(parts))

        for i, part in enumerate(parts[:MAX_ITEMS_TO_PROCESS]):
            part_type = type(part).__name__
            _safe_set_attribute(span, f"pydantic_ai.request.parts.{i}.type", part_type)

            # Content for text parts
            content = _safe_get_attribute(part, "content")
            if content:
                _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.content", content, MAX_CONTENT_LENGTH)
                _safe_set_attribute(span, f"pydantic_ai.request.parts.{i}.content", content, MAX_CONTENT_LENGTH)

            # Role for message parts
            role = _safe_get_attribute(part, "role")
            if role:
                _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", role)
                _safe_set_attribute(span, f"pydantic_ai.request.parts.{i}.role", role)

            # Timestamp, tool call information
            _safe_set_attribute(
                span, f"pydantic_ai.request.parts.{i}.timestamp", _safe_get_attribute(part, "timestamp")
            )
            _safe_set_attribute(
                span, f"pydantic_ai.request.parts.{i}.tool_name", _safe_get_attribute(part, "tool_name")
            )
            _safe_set_attribute(
                span, f"pydantic_ai.request.parts.{i}.tool_call_id", _safe_get_attribute(part, "tool_call_id")
            )
            _safe_set_attribute(
                span, f"pydantic_ai.request.parts.{i}.args", _safe_get_attribute(part, "args"), MAX_ARGS_LENGTH
            )

    # Request metadata
    _safe_set_attribute(span, "pydantic_ai.request.model_name", _safe_get_attribute(request, "model_name"))
    _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", _safe_get_attribute(request, "temperature"))
    _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", _safe_get_attribute(request, "max_tokens"))


def _set_call_tools_node_attributes(span: Any, node: Any) -> None:
    """Set attributes specific to CallToolsNode."""
    response = _safe_get_attribute(node, "model_response")
    if not response:
        return

    # Response parts
    parts = _safe_get_attribute(response, "parts")
    if parts:
        _safe_set_attribute(span, "pydantic_ai.response.parts_count", len(parts))

        for i, part in enumerate(parts[:MAX_ITEMS_TO_PROCESS]):
            part_type = type(part).__name__
            _safe_set_attribute(span, f"pydantic_ai.response.parts.{i}.type", part_type)

            # Content for text parts
            content = _safe_get_attribute(part, "content")
            if content:
                _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", content, MAX_CONTENT_LENGTH)
                _safe_set_attribute(span, f"pydantic_ai.response.parts.{i}.content", content, MAX_CONTENT_LENGTH)
                _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", "assistant")

            # Tool call information
            _safe_set_attribute(
                span, f"pydantic_ai.response.parts.{i}.tool_name", _safe_get_attribute(part, "tool_name")
            )
            _safe_set_attribute(
                span, f"pydantic_ai.response.parts.{i}.tool_call_id", _safe_get_attribute(part, "tool_call_id")
            )
            _safe_set_attribute(
                span, f"pydantic_ai.response.parts.{i}.args", _safe_get_attribute(part, "args"), MAX_ARGS_LENGTH
            )

    # Usage information
    usage = _safe_get_attribute(response, "usage")
    if usage:
        _safe_set_attribute(span, "pydantic_ai.usage.requests", _safe_get_attribute(usage, "requests"))

        # Token usage with dual attributes for compatibility
        request_tokens = _safe_get_attribute(usage, "request_tokens")
        if request_tokens is not None:
            _safe_set_attribute(span, f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", request_tokens)
            _safe_set_attribute(span, "pydantic_ai.usage.request_tokens", request_tokens)

        response_tokens = _safe_get_attribute(usage, "response_tokens")
        if response_tokens is not None:
            _safe_set_attribute(span, f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", response_tokens)
            _safe_set_attribute(span, "pydantic_ai.usage.response_tokens", response_tokens)

        total_tokens = _safe_get_attribute(usage, "total_tokens")
        if total_tokens is not None:
            _safe_set_attribute(span, f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", total_tokens)
            _safe_set_attribute(span, "pydantic_ai.usage.total_tokens", total_tokens)

        # Additional usage details
        details = _safe_get_attribute(usage, "details")
        if details:
            for key, value in details.items():
                if value is not None:
                    _safe_set_attribute(span, f"pydantic_ai.usage.details.{key}", value)

    # Model information (only for CallToolsNode)
    model_name = _safe_get_attribute(response, "model_name")
    if model_name:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_RESPONSE_MODEL}", model_name)
        _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_MODEL}", model_name)
        _safe_set_attribute(span, "pydantic_ai.response.model_name", model_name)

    # Timestamp
    _safe_set_attribute(span, "pydantic_ai.response.timestamp", _safe_get_attribute(response, "timestamp"))

    # Tool execution results (if available)
    tool_results = _safe_get_attribute(node, "tool_results")
    if tool_results:
        _safe_set_attribute(span, "pydantic_ai.tool_results_count", len(tool_results))
        for i, result in enumerate(tool_results[:MAX_FUNCTIONS_TO_PROCESS]):
            _safe_set_attribute(span, f"pydantic_ai.tool_results.{i}", result, MAX_ARGS_LENGTH)


def _set_end_node_attributes(span: Any, node: Any) -> None:
    """Set attributes specific to End node."""
    data = _safe_get_attribute(node, "data")
    if not data:
        return

    # Final output
    output = _safe_get_attribute(data, "output")
    if output is not None:
        _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        _safe_set_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH)
        _safe_set_attribute(span, "pydantic_ai.final_output", output, MAX_CONTENT_LENGTH)

    # Cost information
    cost = _safe_get_attribute(data, "cost")
    if cost is not None:
        _safe_set_attribute(span, "pydantic_ai.cost", float(cost))

    # Usage summary
    usage = _safe_get_attribute(data, "usage")
    if usage:
        _safe_set_attribute(span, "pydantic_ai.final_usage.total_tokens", _safe_get_attribute(usage, "total_tokens"))
        _safe_set_attribute(
            span, "pydantic_ai.final_usage.request_tokens", _safe_get_attribute(usage, "request_tokens")
        )
        _safe_set_attribute(
            span, "pydantic_ai.final_usage.response_tokens", _safe_get_attribute(usage, "response_tokens")
        )

    # Messages history
    messages = _safe_get_attribute(data, "messages")
    if messages:
        _safe_set_attribute(span, "pydantic_ai.messages_count", len(messages))

    # New messages
    new_messages = _safe_get_attribute(data, "new_messages")
    if new_messages:
        _safe_set_attribute(span, "pydantic_ai.new_messages_count", len(new_messages))


def _set_generic_node_attributes(span: Any, node: Any) -> None:
    """Set attributes for any other node types."""
    # Try to extract common attributes that might be available
    for attr_name in ["content", "message", "value", "result", "error"]:
        attr_value = _safe_get_attribute(node, attr_name)
        _safe_set_attribute(span, f"pydantic_ai.{attr_name}", attr_value, MAX_CONTENT_LENGTH)
