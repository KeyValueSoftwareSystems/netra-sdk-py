"""Netra decorator utilities.

This module provides decorators for common patterns in Netra SDK.
Decorators can be applied to both functions and classes.
"""

import functools
import inspect
import json
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

from opentelemetry import trace

from .config import Config

# Type variables for preserving function and class signatures
F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


def _serialize_value(value: Any) -> str:
    """Safely serialize a value to string for span attributes."""
    try:
        if isinstance(value, (str, int, float, bool, type(None))):
            return str(value)
        elif isinstance(value, (list, dict, tuple)):
            return json.dumps(value, default=str)[:1000]  # Limit size
        else:
            return str(value)[:1000]  # Limit size
    except Exception:
        return str(type(value).__name__)


def _add_span_attributes(
    span: trace.Span, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any], entity_type: str
) -> None:
    """Helper function to add span attributes from function parameters."""
    # Set entity type
    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.type", entity_type)

    # Add input attributes
    try:
        # Get parameter names from function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        input_data = {}

        # Add positional arguments
        for i, arg in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                # Skip 'self' and 'cls' parameters for class methods
                if param_name not in ("self", "cls"):
                    input_data[param_name] = _serialize_value(arg)

        # Add keyword arguments
        for key, value in kwargs.items():
            input_data[key] = _serialize_value(value)

        if input_data:
            span.set_attribute(f"{Config.LIBRARY_NAME}.entity.input", json.dumps(input_data))

    except Exception as e:
        span.set_attribute(f"{Config.LIBRARY_NAME}.input_error", str(e))


def _add_output_attributes(span: trace.Span, result: Any) -> None:
    """Helper function to add output attributes to span."""
    try:
        serialized_output = _serialize_value(result)
        span.set_attribute(f"{Config.LIBRARY_NAME}", serialized_output)
    except Exception as e:
        span.set_attribute(f"{Config.LIBRARY_NAME}.entity.output_error", str(e))


def _create_function_wrapper(func: F, entity_type: str, name: Optional[str] = None) -> F:
    """Create a wrapper for a function with span tracking."""
    # Get the module name to use for the tracer
    module_name = func.__module__

    # Check if the function is a coroutine function (async)
    is_async = inspect.iscoroutinefunction(func)

    # Create span name
    span_name = name if name is not None else func.__name__

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the tracer for this module
            tracer = trace.get_tracer(module_name)

            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add input attributes
                _add_span_attributes(span, func, args, kwargs, entity_type)

                try:
                    # Execute the wrapped async function and await its result
                    result = await func(*args, **kwargs)

                    # Add output attributes
                    _add_output_attributes(span, result)

                    return result
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                    raise

        return cast(F, async_wrapper)
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the tracer for this module
            tracer = trace.get_tracer(module_name)

            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add input attributes
                _add_span_attributes(span, func, args, kwargs, entity_type)

                try:
                    # Execute the wrapped function
                    result = func(*args, **kwargs)

                    # Add output attributes
                    _add_output_attributes(span, result)

                    return result
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                    raise

        return cast(F, sync_wrapper)


def _wrap_class_methods(cls: C, entity_type: str, name: Optional[str] = None) -> C:
    """Wrap all methods of a class with span tracking."""
    class_name = name if name is not None else cls.__name__

    # Get all attributes defined in this class (not inherited)
    for attr_name in cls.__dict__:
        attr = getattr(cls, attr_name)

        # Skip private methods and special methods
        if attr_name.startswith("_"):
            continue

        # Check if it's a callable method (function)
        if callable(attr) and inspect.isfunction(attr):
            # Create method name for span
            method_span_name = f"{class_name}.{attr_name}"

            # Wrap the method
            wrapped_method = _create_function_wrapper(attr, entity_type, method_span_name)
            setattr(cls, attr_name, wrapped_method)

    return cls


def workflow(
    target: Union[F, C, None] = None, *, name: Optional[str] = None
) -> Union[F, C, Callable[[Union[F, C]], Union[F, C]]]:
    """Decorator that wraps functions or classes with OpenTelemetry span tracking for workflows.

    Can be applied to functions, async functions, or classes.
    When applied to classes, all public methods are wrapped.

    Args:
        target: The function or class to wrap, or None if called with parameters
        name: Optional custom name for the span. If not provided, the function/class name is used.

    Returns:
        The wrapped function or class with span tracking

    Example:
        @workflow
        def my_workflow(data):
            return process_data(data)

        @workflow(name="Custom Workflow")
        async def async_workflow(data):
            return await process_data_async(data)

        @workflow
        class MyWorkflowClass:
            def process(self, data):
                return data * 2
    """

    def decorator(obj: Union[F, C]) -> Union[F, C]:
        if inspect.isclass(obj):
            return cast(Union[F, C], _wrap_class_methods(cast(C, obj), "workflow", name))
        else:
            return cast(Union[F, C], _create_function_wrapper(cast(F, obj), "workflow", name))

    # Handle both @workflow and @workflow(name="...")
    if target is not None:
        return decorator(target)
    return decorator


def agent(
    target: Union[F, C, None] = None, *, name: Optional[str] = None
) -> Union[F, C, Callable[[Union[F, C]], Union[F, C]]]:
    """Decorator that wraps functions or classes with OpenTelemetry span tracking for agents.

    Can be applied to functions, async functions, or classes.
    When applied to classes, all public methods are wrapped.

    Args:
        target: The function or class to wrap, or None if called with parameters
        name: Optional custom name for the span. If not provided, the function/class name is used.

    Returns:
        The wrapped function or class with span tracking

    Example:
        @agent
        def my_agent(query):
            return process_query(query)

        @agent(name="Custom Agent")
        async def async_agent(query):
            return await process_query_async(query)

        @agent
        class MyAgentClass:
            def respond(self, query):
                return f"Response to: {query}"
    """

    def decorator(obj: Union[F, C]) -> Union[F, C]:
        if inspect.isclass(obj):
            return cast(Union[F, C], _wrap_class_methods(cast(C, obj), "agent", name))
        else:
            return cast(Union[F, C], _create_function_wrapper(cast(F, obj), "agent", name))

    # Handle both @agent and @agent(name="...")
    if target is not None:
        return decorator(target)
    return decorator


def task(
    target: Union[F, C, None] = None, *, name: Optional[str] = None
) -> Union[F, C, Callable[[Union[F, C]], Union[F, C]]]:
    """Decorator that wraps functions or classes with OpenTelemetry span tracking for tasks.

    Can be applied to functions, async functions, or classes.
    When applied to classes, all public methods are wrapped.

    Args:
        target: The function or class to wrap, or None if called with parameters
        name: Optional custom name for the span. If not provided, the function/class name is used.

    Returns:
        The wrapped function or class with span tracking

    Example:
        @task
        def my_task(data):
            return process_data(data)

        @task(name="Custom Task")
        async def async_task(data):
            return await process_data_async(data)

        @task
        class MyTaskClass:
            def execute(self, data):
                return data.upper()
    """

    def decorator(obj: Union[F, C]) -> Union[F, C]:
        if inspect.isclass(obj):
            return cast(Union[F, C], _wrap_class_methods(cast(C, obj), "task", name))
        else:
            return cast(Union[F, C], _create_function_wrapper(cast(F, obj), "task", name))

    # Handle both @task and @task(name="...")
    if target is not None:
        return decorator(target)
    return decorator
