import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)

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


class InstrumentedAgentRun:
    """Wrapper for AgentRun that creates spans for each node iteration"""

    def __init__(self, agent_run: Any, tracer: Tracer, parent_span_name: str, parent_span: Any) -> None:
        self._agent_run = agent_run
        self._tracer = tracer
        self._parent_span_name = parent_span_name
        self._parent_span = parent_span  # Keep reference to parent span

    async def __aenter__(self) -> "InstrumentedAgentRun":
        # Enter the original agent run context
        await self._agent_run.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        # Exit the original agent run context
        return await self._agent_run.__aexit__(exc_type, exc_val, exc_tb)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._instrumented_iter()

    async def _instrumented_iter(self) -> AsyncIterator[Any]:
        """Async iterator that creates spans for each node"""
        async for node in self._agent_run:
            # Create span for this node as child of parent span
            span_name = get_node_span_name(node)
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
            ) as span:
                try:
                    # Set node attributes
                    set_node_attributes(span, node)

                    # For End nodes, also set assistant message on parent span
                    if hasattr(node, "__class__") and "End" in node.__class__.__name__:
                        self._set_assistant_message_on_parent(node)

                    span.set_status(Status(StatusCode.OK))
                    yield node
                except Exception as e:
                    _handle_span_error(span, e)
                    raise

    def _set_assistant_message_on_parent(self, node: Any) -> None:
        """Set assistant message from End node on the parent span"""
        if not self._parent_span or not self._parent_span.is_recording():
            return

        # Extract the same data that _set_end_node_attributes uses
        data = _safe_get_attribute(node, "data")
        if not data:
            return

        # Get the final output and set it on parent span
        output = _safe_get_attribute(data, "output")
        if output is not None:
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(
                self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH
            )
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

    async def next(self, node: Any = None) -> Any:
        """Manual iteration with instrumentation"""
        if hasattr(self._agent_run, "next"):
            next_node = await self._agent_run.next(node)
            # Create span for the returned node
            if next_node:
                span_name = get_node_span_name(next_node)
                with self._tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                ) as span:
                    set_node_attributes(span, next_node)
                    span.set_status(Status(StatusCode.OK))
            return next_node
        else:
            raise AttributeError("AgentRun does not have a 'next' method")

    @property
    def result(self) -> Any:
        """Access to the final result"""
        return getattr(self._agent_run, "result", None)

    @property
    def ctx(self) -> Any:
        """Access to the context"""
        return getattr(self._agent_run, "ctx", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the wrapped AgentRun"""
        return getattr(self._agent_run, name)


def agent_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.run method."""

    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # type: ignore[type-arg]
        async def async_wrapper() -> Any:
            if should_suppress_instrumentation():
                return await wrapped(*args, **kwargs)

            # Extract key parameters
            user_prompt = args[0] if args else kwargs.get("user_prompt", "")

            # Use start_as_current_span for async non-streaming operations
            with tracer.start_as_current_span(
                "pydantic_ai.agent.run", kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent.run"}
            ) as span:
                try:
                    # Set request attributes
                    set_pydantic_request_attributes(span, kwargs, "agent.run")

                    if user_prompt:
                        _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                        _safe_set_attribute(
                            span, f"{SpanAttributes.LLM_PROMPTS}.0.content", user_prompt, MAX_CONTENT_LENGTH
                        )

                    # Execute the original async method
                    start_time = time.time()
                    result = await wrapped(*args, **kwargs)
                    end_time = time.time()

                    # Set response attributes
                    _set_timing_attributes(span, start_time, end_time)
                    set_pydantic_response_attributes(span, result)

                    # Set assistant response content
                    _set_assistant_response_content(span, result, "completed")

                    span.set_status(Status(StatusCode.OK))

                    # Return instrumented AgentRun that will capture child nodes
                    return InstrumentedAgentRun(result, tracer, "pydantic_ai.agent.run", span)

                except Exception as e:
                    _handle_span_error(span, e)
                    raise

        return async_wrapper()

    return wrapper


def agent_run_sync_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.run_sync method."""

    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # type: ignore[type-arg]
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Extract key parameters
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")

        # Create parent span for the entire run_sync operation
        with tracer.start_as_current_span(
            "pydantic_ai.agent.run_sync",
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                # Set request attributes
                set_pydantic_request_attributes(span, kwargs, "agent.run_sync")

                if user_prompt:
                    _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                    _safe_set_attribute(
                        span, f"{SpanAttributes.LLM_PROMPTS}.0.content", user_prompt, MAX_CONTENT_LENGTH
                    )

                start_time = time.time()

                # Use iter method to capture all nodes, then get final result
                import asyncio

                async def _run_with_instrumentation() -> Any:
                    # Use the iter method directly without suppressing instrumentation
                    # This will create the proper span hierarchy: run_sync -> iter -> nodes
                    async with instance.iter(*args, **kwargs) as agent_run:
                        async for node in agent_run:
                            pass  # Just iterate through to capture all nodes
                        return agent_run.result

                # Run the async instrumentation in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(_run_with_instrumentation())
                finally:
                    loop.close()

                end_time = time.time()

                # Set response attributes
                _set_timing_attributes(span, start_time, end_time)
                set_pydantic_response_attributes(span, result)

                # Set assistant response content in OpenAI wrapper format
                _set_assistant_response_content(span, result, "completed")

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                _handle_span_error(span, e)
                raise

    return wrapper


class InstrumentedAgentRunContext:
    """Context manager that keeps the parent span active during iteration"""

    def __init__(
        self, agent_run: Any, tracer: Tracer, span: Any, user_prompt: str, model_name: str, kwargs: Dict[str, Any]
    ) -> None:
        self._agent_run = agent_run
        self._tracer = tracer
        self._span = span
        self._user_prompt = user_prompt
        self._model_name = model_name
        self._kwargs = kwargs
        self._context_token = None

    async def __aenter__(self) -> "InstrumentedAgentRunContext":
        # Enter the original agent run context
        result = await self._agent_run.__aenter__()

        # Set the parent span as the current active span context using OpenTelemetry's trace context
        # This ensures that child spans created by InstrumentedAgentRun will be children of this span
        from opentelemetry import trace

        span_context = trace.set_span_in_context(self._span)
        self._context_token = context_api.attach(span_context)

        # Set request attributes now that we're in the context
        set_pydantic_request_attributes(self._span, self._kwargs, "agent.iter")

        if self._user_prompt:
            _safe_set_attribute(self._span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _safe_set_attribute(
                self._span, f"{SpanAttributes.LLM_PROMPTS}.0.content", self._user_prompt, MAX_CONTENT_LENGTH
            )

        return InstrumentedAgentRun(result, self._tracer, "pydantic_ai.agent.iter", self._span)  # type: ignore[return-value]

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        try:
            # Exit the original agent run context
            result = await self._agent_run.__aexit__(exc_type, exc_val, exc_tb)

            if exc_type is None:
                _safe_set_attribute(self._span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "streaming")
                self._span.set_status(Status(StatusCode.OK))
            else:
                _handle_span_error(self._span, exc_val)

            return result
        finally:
            # Detach the context token to restore previous context
            if self._context_token is not None:
                context_api.detach(self._context_token)
            # End the parent span
            self._span.end()


def agent_iter_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.iter method."""

    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # type: ignore[type-arg]
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Extract key parameters
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")
        model_name = kwargs.get("model", getattr(instance, "model", None))

        # Execute the original method to get AgentRun
        start_time = time.time()
        agent_run = wrapped(*args, **kwargs)
        end_time = time.time()

        # Create parent span that will stay active during iteration
        # Use start_span (not start_as_current_span) because we need to manage it manually
        # The InstrumentedAgentRunContext will set it as current when needed
        span = tracer.start_span(
            "pydantic_ai.agent.iter",
            kind=SpanKind.CLIENT,
        )

        # Set initial timing
        _set_timing_attributes(span, start_time, end_time)

        # Return context manager that will manage the span lifecycle and hierarchy
        return InstrumentedAgentRunContext(agent_run, tracer, span, user_prompt, model_name, kwargs)  # type: ignore[arg-type]

    return wrapper


class InstrumentedAgentRunFromStream:
    """Wrapper for AgentRun from stream that creates spans for each node iteration"""

    def __init__(self, agent_run: Any, tracer: Tracer, parent_span: Any) -> None:
        self._agent_run = agent_run
        self._tracer = tracer
        self._parent_span = parent_span

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._instrumented_iter()

    async def _instrumented_iter(self) -> AsyncIterator[Any]:
        """Async iterator that creates spans for each node"""
        try:
            async for node in self._agent_run:
                # Create span for this node as child of parent span
                span_name = get_node_span_name(node)

                # Set parent context explicitly to ensure proper parent-child relationship
                parent_context = None
                if self._parent_span:
                    parent_context = context_api.set_value("current_span", self._parent_span)

                with self._tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                    context=parent_context,
                ) as span:
                    try:
                        # Set node attributes
                        set_node_attributes(span, node)

                        # For End nodes, also set assistant message on both current span and parent span
                        if hasattr(node, "__class__") and "End" in node.__class__.__name__:
                            self._set_assistant_content_on_spans(span, node)

                        span.set_status(Status(StatusCode.OK))

                        # Yield the node to the user
                        yield node
                    except Exception as e:
                        _handle_span_error(span, e)
                        # Still yield the node even if span creation failed
                        yield node
        except Exception as e:
            # Handle iteration errors
            logger.error(f"Error during node iteration: {e}")
            raise

    def _set_assistant_content_on_spans(self, current_span: Any, node: Any) -> None:
        """Set assistant message from End node on both current span and parent span"""
        # Extract the same data that _set_end_node_attributes uses
        data = _safe_get_attribute(node, "data")
        if not data:
            return

        # Get the final output
        output = _safe_get_attribute(data, "output")
        if output is None:
            return

        # Set assistant content on current iter span
        if current_span and current_span.is_recording():
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH)
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

        # Set assistant content on parent run_stream span
        if self._parent_span and self._parent_span.is_recording():
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(
                self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH
            )
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the wrapped AgentRun"""
        return getattr(self._agent_run, name)


class InstrumentedStreamedRunResultIterable:
    """Iterable wrapper for StreamedRunResult that provides node iteration capability"""

    def __init__(self, streamed_run_result: Any, tracer: Tracer, parent_span: Any) -> None:
        self._streamed_run_result = streamed_run_result
        self._tracer = tracer
        self._parent_span = parent_span
        # Extract the agent instance to create an iterable AgentRun
        # We need to get the agent from the streamed result to create an iter() call
        self._agent = None
        self._user_prompt = None
        self._deps = None

        # Try to extract agent and prompt from the streamed result
        if hasattr(streamed_run_result, "_agent"):
            self._agent = streamed_run_result._agent
        if hasattr(streamed_run_result, "_user_prompt"):
            self._user_prompt = streamed_run_result._user_prompt
        if hasattr(streamed_run_result, "_deps"):
            self._deps = streamed_run_result._deps

    def __aiter__(self) -> Any:
        return self._instrumented_iter()

    async def _instrumented_iter(self) -> Any:
        """Create an AgentRun using agent.iter() and iterate over nodes with instrumentation"""
        if not self._agent or not self._user_prompt:
            logger.error("Cannot iterate: missing agent or user_prompt from StreamedRunResult")
            return

        try:
            # Create an AgentRun using agent.iter() with the same prompt and deps
            iter_kwargs = {}
            if self._deps is not None:
                iter_kwargs["deps"] = self._deps

            async with self._agent.iter(self._user_prompt, **iter_kwargs) as agent_run:
                async for node in agent_run:
                    # Create span for this node as child of parent span
                    span_name = get_node_span_name(node)

                    # Create span as child of parent span using proper OpenTelemetry context
                    if self._parent_span:
                        # Create a context with the parent span as the current span
                        parent_context = context_api.set_value(
                            context_api.get_current(), "current_span", self._parent_span
                        )
                        # Start span with parent context
                        span = self._tracer.start_span(span_name, kind=SpanKind.INTERNAL, context=parent_context)
                    else:
                        # Fallback to current span if no parent
                        span = self._tracer.start_span(span_name, kind=SpanKind.INTERNAL)

                    try:
                        # Set node attributes
                        set_node_attributes(span, node)

                        # For End nodes, also set assistant message on both current span and parent span
                        if hasattr(node, "__class__") and "End" in node.__class__.__name__:
                            self._set_assistant_content_on_spans(span, node)

                        span.set_status(Status(StatusCode.OK))

                        # Yield the node to the user
                        yield node
                    except Exception as e:
                        _handle_span_error(span, e)
                        # Still yield the node even if span creation failed
                        yield node
                    finally:
                        # Always end the span
                        span.end()
        except Exception as e:
            # Handle iteration errors
            logger.error(f"Error during node iteration: {e}")
            raise

    def _set_assistant_content_on_spans(self, current_span, node) -> None:  # type: ignore[no-untyped-def]
        """Set assistant message from End node on both current span and parent span"""
        # Extract the same data that _set_end_node_attributes uses
        data = _safe_get_attribute(node, "data")
        if not data:
            return

        # Get the final output
        output = _safe_get_attribute(data, "output")
        if output is None:
            return

        # Set assistant content on current iter span
        if current_span and current_span.is_recording():
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH)
            _safe_set_attribute(current_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

        # Set assistant content on parent run_stream span
        if self._parent_span and self._parent_span.is_recording():
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(
                self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH
            )
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

    def __getattr__(self, name) -> Any:  # type: ignore[no-untyped-def]
        """Delegate other attributes to the wrapped StreamedRunResult"""
        return getattr(self._streamed_run_result, name)


class InstrumentedStreamedRunResult:
    """Wrapper for StreamedRunResult that creates spans during streaming"""

    def __init__(
        self, streamed_result: Any, tracer: Tracer, span: Any, start_time: float, request_kwargs: Dict[str, Any]
    ) -> None:
        self._streamed_result = streamed_result
        self._tracer = tracer
        self._span = span
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._parent_span = span  # Keep for backward compatibility
        self._agent_run = None

    async def __aenter__(self) -> Any:
        # Enter the original streamed result and store it
        self._streamed_run_result = await self._streamed_result.__aenter__()

        # StreamedRunResult is not iterable, but users expect to iterate over nodes
        # We need to create an iterable wrapper that provides node iteration capability
        return InstrumentedStreamedRunResultIterable(self._streamed_run_result, self._tracer, self._span)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:  # type: ignore[no-untyped-def]
        try:
            result = await self._streamed_result.__aexit__(exc_type, exc_val, exc_tb)
            # Finalize the parent span when streaming is complete
            self._finalize_parent_span()
            return result
        except Exception as e:
            # Handle any errors and still finalize the span
            if self._span and self._span.is_recording():
                self._span.set_status(Status(StatusCode.ERROR, str(e)))
                self._span.record_exception(e)
                self._span.end()
            raise

    def _set_assistant_message_on_parent_span(self, node) -> None:  # type: ignore[no-untyped-def]
        """Set assistant message from End node on the parent span"""
        if not self._parent_span or not self._parent_span.is_recording():
            return

        # Extract the same data that _set_end_node_attributes uses
        data = _safe_get_attribute(node, "data")
        if not data:
            return

        # Get the final output and set it on parent span
        output = _safe_get_attribute(data, "output")
        if output is not None:
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
            _safe_set_attribute(
                self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output, MAX_CONTENT_LENGTH
            )
            _safe_set_attribute(self._parent_span, f"{SpanAttributes.LLM_COMPLETIONS}.0.finish_reason", "completed")

    def _finalize_parent_span(self) -> None:
        """Finalize parent span when streaming is complete"""
        if not self._span or not self._span.is_recording():
            return

        # Calculate duration
        end_time = time.time()
        end_time - self._start_time

        # Set timing attributes
        _set_timing_attributes(self._span, self._start_time, end_time)

        # Set response attributes if we have access to the final result
        try:
            if hasattr(self._streamed_result, "result"):
                final_result = self._streamed_result.result()
                set_pydantic_response_attributes(self._span, final_result)
                _set_assistant_response_content(self._span, final_result, "streaming")
        except Exception:
            # Ignore errors when trying to get final result
            pass

        # Mark span as successful and end it
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the wrapped StreamedRunResult"""
        return getattr(self._streamed_result, name)


def agent_run_stream_wrapper(tracer: Tracer) -> Callable:  # type: ignore[type-arg]
    """Wrapper for Agent.run_stream method."""

    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # type: ignore[type-arg]
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Extract key parameters
        user_prompt = args[0] if args else kwargs.get("user_prompt", "")

        # Use start_span for streaming operations - returns span directly (not context manager)
        span = tracer.start_span(
            "pydantic_ai.agent.run_stream", kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent.run_stream"}
        )

        try:
            # Set request attributes
            set_pydantic_request_attributes(span, kwargs, "agent.run_stream")

            if user_prompt:
                _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                _safe_set_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", user_prompt, MAX_CONTENT_LENGTH)

            # Execute the original method to get the async context manager
            start_time = time.time()
            original_result = wrapped(*args, **kwargs)

            # Return instrumented StreamedRunResult that will manage the span lifecycle
            return InstrumentedStreamedRunResult(original_result, tracer, span, start_time, kwargs)

        except Exception as e:
            # Handle error and end span manually since we're not using context manager
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    return wrapper


def tool_function_wrapper(tracer: Tracer) -> Callable:  # type: ignore[type-arg]
    """Wrapper for tool function calls."""

    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # type: ignore[type-arg]
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        function_name = getattr(wrapped, "__name__", "unknown_tool")

        # Create span for tool execution
        with tracer.start_as_current_span(
            f"pydantic_ai.tool.{function_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            try:
                # Set span attributes
                _safe_set_attribute(span, f"{SpanAttributes.LLM_REQUEST_TYPE}", "tool.call")
                _safe_set_attribute(span, "tool.name", function_name)

                # Add function arguments (be careful with sensitive data)
                if args:
                    _safe_set_attribute(span, "tool.args", str(args), MAX_ARGS_LENGTH)
                if kwargs:
                    _safe_set_attribute(span, "tool.kwargs", str(kwargs), MAX_ARGS_LENGTH)

                # Execute the original method
                start_time = time.time()
                result = wrapped(*args, **kwargs)
                end_time = time.time()

                # Set result attributes
                _safe_set_attribute(span, "tool.result", str(result), MAX_ARGS_LENGTH)
                _set_timing_attributes(span, start_time, end_time)

                # Set comprehensive response attributes if result has pydantic_ai structure
                if hasattr(result, "usage") or hasattr(result, "output"):
                    set_pydantic_response_attributes(span, result)

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                _handle_span_error(span, e)
                raise

    return wrapper
