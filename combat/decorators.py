"""Combat decorator utilities.

This module provides decorators for common patterns in Combat SDK.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, cast

from opentelemetry import trace

# Type variable for preserving function signature in decorators
F = TypeVar("F", bound=Callable[..., Any])


def workflow(func_or_name: Any = None, *, name: Optional[str] = None) -> Any:
    """Decorator that wraps a function with OpenTelemetry span tracking.

    This decorator starts a new span with the function name as the span name,
    executes the function within the span context, and automatically ends
    the span when the function completes or raises an exception.

    Works with both synchronous and asynchronous functions.

    Args:
        func_or_name: The function to wrap with span tracking, or None if called with parameters
        name: Optional custom name for the workflow span. If not provided, the function name is used.

    Returns:
        The wrapped function with span tracking

    Example:
        @workflow
        def process_data(data):
            # This function will be automatically tracked in a span named "process_data"
            return transformed_data

        @workflow(name="Custom Workflow")
        def process_data(data):
            # This function will be automatically tracked in a span named "Custom Workflow"
            return transformed_data

        @workflow
        async def process_data_async(data):
            # This async function will be properly tracked with the span lasting for the entire async execution
            return transformed_data
    """

    def decorator(func: F) -> F:
        # Get the module name to use for the tracer
        module_name = func.__module__

        # Check if the function is a coroutine function (async)
        is_async = inspect.iscoroutinefunction(func)

        # Define wrapper for synchronous functions
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the tracer for this module
            tracer = trace.get_tracer(module_name)

            # Create a span name from the provided name or function name
            span_name = name if name is not None else func.__name__

            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add function parameters as span attributes if they're serializable
                _add_span_attributes(span, func, args, kwargs)

                # Execute the wrapped function
                return func(*args, **kwargs)

        # Define wrapper for asynchronous functions
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the tracer for this module
            tracer = trace.get_tracer(module_name)

            # Create a span name from the provided name or function name
            span_name = name if name is not None else func.__name__

            # Start a new span
            with tracer.start_as_current_span(span_name) as span:
                # Add function parameters as span attributes if they're serializable
                _add_span_attributes(span, func, args, kwargs)

                # Execute the wrapped async function and await its result
                return await func(*args, **kwargs)

        # Return the appropriate wrapper based on whether the function is async or not
        return cast(F, async_wrapper if is_async else sync_wrapper)

    # Helper function to add span attributes
    def _add_span_attributes(
        span: trace.Span, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        # Add positional arguments as span attributes
        if args:
            try:
                # Get parameter names from function signature
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())

                # Add positional arguments as span attributes
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        # Only add primitive types that can be serialized
                        if isinstance(arg, (str, int, float, bool)):
                            span.set_attribute(f"arg.{param_names[i]}", str(arg))
            except Exception:
                # Skip attribute setting if it fails
                pass

        # Add keyword arguments as span attributes
        for key, value in kwargs.items():
            # Only add primitive types that can be serialized
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"kwarg.{key}", str(value))

    # Handle both @workflow and @workflow(name="...")
    if callable(func_or_name):
        return decorator(func_or_name)
    return decorator
