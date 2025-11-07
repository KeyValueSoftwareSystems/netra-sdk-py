"""
DSPy API wrappers for Netra SDK instrumentation.

This module contains wrapper functions for DSPy LM methods with
proper span handling for async operations.
"""

import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)

# Span names
DSPY_ACALL_SPAN_NAME = "dspy.lm.acall"
DSPY_AFORWARD_SPAN_NAME = "dspy.lm.aforward"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def model_as_dict(obj: Any) -> Dict[str, Any]:
    """Convert DSPy model object to dictionary"""
    if hasattr(obj, "model_dump"):
        result = obj.model_dump()
        return result if isinstance(result, dict) else {}
    elif hasattr(obj, "to_dict"):
        result = obj.to_dict()
        return result if isinstance(result, dict) else {}
    elif isinstance(obj, dict):
        return obj
    else:
        return {}


def extract_messages_from_args(prompt: Optional[str], messages: Optional[list]) -> list:
    """Extract messages from DSPy acall/aforward arguments"""
    if messages:
        return messages if isinstance(messages, list) else []
    elif prompt:
        return [{"role": "user", "content": prompt}]
    return []


def set_request_attributes(
    span: Span, 
    prompt: Optional[str], 
    messages: Optional[list], 
    kwargs: Dict[str, Any],
    lm_instance: Any
) -> None:
    """Set request attributes on span"""
    if not span.is_recording():
        return

    # Set operation type
    span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TYPE}", "chat")
    
    # Model information
    if hasattr(lm_instance, "model") and lm_instance.model:
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MODEL}", lm_instance.model)
    
    # Model type
    if hasattr(lm_instance, "model_type") and lm_instance.model_type:
        span.set_attribute("gen_ai.dspy.model_type", lm_instance.model_type)
    
    # Temperature
    temperature = kwargs.get("temperature")
    if temperature is None and hasattr(lm_instance, "kwargs") and lm_instance.kwargs.get("temperature") is not None:
        temperature = lm_instance.kwargs.get("temperature")
    if temperature is not None:
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", temperature)
    
    # Max tokens
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is None and hasattr(lm_instance, "kwargs") and lm_instance.kwargs.get("max_tokens") is not None:
        max_tokens = lm_instance.kwargs.get("max_tokens")
    if max_tokens is not None:
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", max_tokens)
    
    # Cache setting
    cache = kwargs.get("cache")
    if cache is None and hasattr(lm_instance, "cache"):
        cache = lm_instance.cache
    if cache is not None:
        span.set_attribute("gen_ai.dspy.cache", cache)
    
    # Rollout ID
    rollout_id = kwargs.get("rollout_id")
    if rollout_id is not None:
        span.set_attribute("gen_ai.dspy.rollout_id", rollout_id)
    
    # Extract and set messages
    msg_list = extract_messages_from_args(prompt, messages)
    if msg_list:
        for index, message in enumerate(msg_list):
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", role)
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(content))


def set_response_attributes(span: Span, response: Any) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        return
    
    response_dict = model_as_dict(response)
    
    # Model from response
    if response_dict.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_RESPONSE_MODEL}", response_dict["model"])
    
    # Response ID
    if response_dict.get("id"):
        span.set_attribute("gen_ai.response.id", response_dict["id"])
    
    # Usage information
    usage = response_dict.get("usage", {})
    if usage:
        if usage.get("prompt_tokens") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", usage["prompt_tokens"])
        if usage.get("completion_tokens") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", usage["completion_tokens"])
        if usage.get("total_tokens") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", usage["total_tokens"])
    
    # Choices/Completions
    choices = response_dict.get("choices", [])
    if choices and isinstance(choices, list):
        for index, choice in enumerate(choices):
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if message:
                    role = message.get("role", "assistant")
                    content = message.get("content", "")
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", role)
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", str(content))
                
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason", finish_reason)
    
    # Cache hit information
    if hasattr(response, "cache_hit"):
        span.set_attribute("gen_ai.dspy.cache_hit", response.cache_hit)


def acall_wrapper(tracer: Tracer) -> Callable:
    """Wrapper for DSPy LM.acall method"""
    
    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        
        # Extract prompt and messages from args/kwargs
        prompt = kwargs.get("prompt") or (args[0] if len(args) > 0 else None)
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)
        
        with tracer.start_as_current_span(
            DSPY_ACALL_SPAN_NAME,
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                # Set request attributes
                set_request_attributes(span, prompt, messages, kwargs, instance)
                
                # Call the original method
                response = wrapped(*args, **kwargs)
                
                # Set response attributes
                set_response_attributes(span, response)
                
                # Set success status
                span.set_status(Status(StatusCode.OK))
                
                return response
                
            except Exception as e:
                # Set error status
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    return wrapper


def aforward_wrapper(tracer: Tracer) -> Callable:
    """Wrapper for DSPy LM.aforward method"""
    
    def wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        
        # Extract prompt and messages from args/kwargs
        prompt = kwargs.get("prompt") or (args[0] if len(args) > 0 else None)
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else None)
        
        with tracer.start_as_current_span(
            DSPY_AFORWARD_SPAN_NAME,
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                # Set request attributes
                set_request_attributes(span, prompt, messages, kwargs, instance)
                
                # Call the original async method
                response = wrapped(*args, **kwargs)
                
                # Set response attributes
                set_response_attributes(span, response)
                
                # Set success status
                span.set_status(Status(StatusCode.OK))
                
                return response
                
            except Exception as e:
                # Set error status
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    return wrapper

