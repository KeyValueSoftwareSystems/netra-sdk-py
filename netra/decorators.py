"""Netra decorator utilities.

This module provides decorators for common patterns in Netra SDK.
Decorators can be applied to both functions and classes.
"""

import functools
import inspect
import json
from typing import Any, Awaitable, Callable, Dict, Optional, ParamSpec, Tuple, TypeVar, Union, cast

from opentelemetry import trace

from .config import Config
from .session_manager import SessionManager

P = ParamSpec("P")
R = TypeVar("R")

F_Callable = TypeVar("F_Callable", bound=Callable[..., Any])
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
    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.type", entity_type)

    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        input_data = {}

        for i, arg in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                if param_name not in ("self", "cls"):
                    input_data[param_name] = _serialize_value(arg)

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
        span.set_attribute(f"{Config.LIBRARY_NAME}.entity.output", serialized_output)
    except Exception as e:
        span.set_attribute(f"{Config.LIBRARY_NAME}.entity.output_error", str(e))


def _create_function_wrapper(func: Callable[P, R], entity_type: str, name: Optional[str] = None) -> Callable[P, R]:
    module_name = func.__name__
    is_async = inspect.iscoroutinefunction(func)
    span_name = name if name is not None else func.__name__

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Push entity to stack before span starts so SessionSpanProcessor can capture it
            SessionManager.push_entity(entity_type, span_name)

            tracer = trace.get_tracer(module_name)
            with tracer.start_as_current_span(span_name) as span:
                _add_span_attributes(span, func, args, kwargs, entity_type)
                try:
                    result = await cast(Awaitable[Any], func(*args, **kwargs))
                    _add_output_attributes(span, result)
                    return result
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    raise
                finally:
                    # Pop entity from stack after function call is done
                    SessionManager.pop_entity(entity_type)

        return cast(Callable[P, R], async_wrapper)

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Push entity to stack before span starts so SessionSpanProcessor can capture it
            SessionManager.push_entity(entity_type, span_name)

            tracer = trace.get_tracer(module_name)
            with tracer.start_as_current_span(span_name) as span:
                _add_span_attributes(span, func, args, kwargs, entity_type)
                try:
                    result = func(*args, **kwargs)
                    _add_output_attributes(span, result)
                    return result
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    raise
                finally:
                    # Pop entity from stack after function call is done
                    SessionManager.pop_entity(entity_type)

        return cast(Callable[P, R], sync_wrapper)


def _wrap_class_methods(cls: C, entity_type: str, name: Optional[str] = None) -> C:
    class_name = name if name is not None else cls.__name__
    for attr_name in cls.__dict__:
        attr = getattr(cls, attr_name)
        if attr_name.startswith("_"):
            continue
        if callable(attr) and inspect.isfunction(attr):
            method_span_name = f"{class_name}.{attr_name}"
            wrapped_method = _create_function_wrapper(attr, entity_type, method_span_name)
            setattr(cls, attr_name, wrapped_method)
    return cls


def workflow(
    target: Union[Callable[P, R], C, None] = None, *, name: Optional[str] = None
) -> Union[Callable[P, R], C, Callable[[Callable[P, R]], Callable[P, R]]]:
    def decorator(obj: Union[Callable[P, R], C]) -> Union[Callable[P, R], C]:
        if inspect.isclass(obj):
            return _wrap_class_methods(cast(C, obj), "workflow", name)
        else:
            return _create_function_wrapper(cast(Callable[P, R], obj), "workflow", name)

    if target is not None:
        return decorator(target)
    return decorator


def agent(
    target: Union[Callable[P, R], C, None] = None, *, name: Optional[str] = None
) -> Union[Callable[P, R], C, Callable[[Callable[P, R]], Callable[P, R]]]:
    def decorator(obj: Union[Callable[P, R], C]) -> Union[Callable[P, R], C]:
        if inspect.isclass(obj):
            return _wrap_class_methods(cast(C, obj), "agent", name)
        else:
            return _create_function_wrapper(cast(Callable[P, R], obj), "agent", name)

    if target is not None:
        return decorator(target)
    return decorator


def task(
    target: Union[Callable[P, R], C, None] = None, *, name: Optional[str] = None
) -> Union[Callable[P, R], C, Callable[[Callable[P, R]], Callable[P, R]]]:
    def decorator(obj: Union[Callable[P, R], C]) -> Union[Callable[P, R], C]:
        if inspect.isclass(obj):
            return _wrap_class_methods(cast(C, obj), "task", name)
        else:
            # When obj is a function, it should be type Callable[P, R]
            return _create_function_wrapper(cast(Callable[P, R], obj), "task", name)

    if target is not None:
        return decorator(target)
    return decorator
