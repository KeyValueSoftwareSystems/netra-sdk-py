"""Netra decorator utilities.

This module provides decorators for common patterns in Netra SDK.
Decorators can be applied to both functions and classes.
"""

import functools
import inspect
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
)

try:
    # Optional import: only present in FastAPI/Starlette environments
    from starlette.responses import StreamingResponse
except Exception:  # pragma: no cover - starlette may not be installed in some environments
    StreamingResponse = None

from opentelemetry import trace

from .config import Config
from .session_manager import SessionManager
from .span_wrapper import SpanType

logger = logging.getLogger(__name__)

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


def _is_streaming_response(obj: Any) -> bool:
    """Return True if obj is a Starlette StreamingResponse instance."""
    if StreamingResponse is None:
        return False
    try:
        return isinstance(obj, StreamingResponse)
    except Exception:
        return False


def _is_async_generator(obj: Any) -> bool:
    return inspect.isasyncgen(obj)


def _is_sync_generator(obj: Any) -> bool:
    return inspect.isgenerator(obj)


def _wrap_async_generator_with_span(
    span: trace.Span,
    agen: AsyncGenerator[Any, None],
    span_name: str,
    entity_type: str,
) -> AsyncGenerator[Any, None]:
    """Wrap an async generator so the span remains current for the full iteration and ends afterwards."""

    async def _wrapped() -> AsyncGenerator[Any, None]:
        # Activate span for the entire iteration
        with trace.use_span(span, end_on_exit=False):
            try:
                async for item in agen:
                    yield item
            except Exception as e:
                try:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                finally:
                    span.end()
                    # De-register and pop entity at the very end for streaming lifecycle
                    try:
                        SessionManager.unregister_span(span_name, span)
                    except Exception:
                        logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                    SessionManager.pop_entity(entity_type)
                raise
            else:
                # Normal completion
                span.end()
                try:
                    SessionManager.unregister_span(span_name, span)
                except Exception:
                    logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                SessionManager.pop_entity(entity_type)

    return _wrapped()


def _wrap_sync_generator_with_span(
    span: trace.Span,
    gen: Generator[Any, None, None],
    span_name: str,
    entity_type: str,
) -> Generator[Any, None, None]:
    """Wrap a sync generator so the span remains current for the full iteration and ends afterwards."""

    def _wrapped() -> Generator[Any, None, None]:
        with trace.use_span(span, end_on_exit=False):
            try:
                for item in gen:
                    yield item
            except Exception as e:
                try:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                finally:
                    span.end()
                    try:
                        SessionManager.unregister_span(span_name, span)
                    except Exception:
                        logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                    SessionManager.pop_entity(entity_type)
                raise
            else:
                span.end()
                try:
                    SessionManager.unregister_span(span_name, span)
                except Exception:
                    logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                SessionManager.pop_entity(entity_type)

    return _wrapped()


def _wrap_streaming_response_with_span(
    span: trace.Span,
    resp: Any,
    span_name: str,
    entity_type: str,
) -> Any:
    """Wrap StreamingResponse.body_iterator with a generator that keeps span current and ends it afterwards."""
    try:
        body_iter = getattr(resp, "body_iterator", None)
        if body_iter is None:
            return resp
        # Async iterator
        if inspect.isasyncgen(body_iter) or hasattr(body_iter, "__aiter__"):

            async def _aiter_wrapper():  # type: ignore[no-untyped-def]
                with trace.use_span(span, end_on_exit=False):
                    try:
                        async for chunk in body_iter:
                            yield chunk
                    except Exception as e:
                        try:
                            span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                            span.record_exception(e)
                        finally:
                            span.end()
                            try:
                                SessionManager.unregister_span(span_name, span)
                            except Exception:
                                logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                            SessionManager.pop_entity(entity_type)
                        raise
                    else:
                        span.end()
                        try:
                            SessionManager.unregister_span(span_name, span)
                        except Exception:
                            logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                        SessionManager.pop_entity(entity_type)

            resp.body_iterator = _aiter_wrapper()  # type: ignore[no-untyped-call]
            return resp

        # Sync iterator
        if inspect.isgenerator(body_iter) or hasattr(body_iter, "__iter__"):

            def _iter_wrapper():  # type: ignore[no-untyped-def]
                with trace.use_span(span, end_on_exit=False):
                    try:
                        for chunk in body_iter:
                            yield chunk
                    except Exception as e:
                        try:
                            span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                            span.record_exception(e)
                        finally:
                            span.end()
                            try:
                                SessionManager.unregister_span(span_name, span)
                            except Exception:
                                logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                            SessionManager.pop_entity(entity_type)
                        raise
                    else:
                        span.end()
                        try:
                            SessionManager.unregister_span(span_name, span)
                        except Exception:
                            logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                        SessionManager.pop_entity(entity_type)

            resp.body_iterator = _iter_wrapper()  # type: ignore[no-untyped-call]
            return resp
    except Exception:
        logger.exception("Failed to wrap StreamingResponse with span '%s'", span_name)
    return resp


def _create_function_wrapper(
    func: Callable[P, R],
    entity_type: str,
    name: Optional[str] = None,
    as_type: Optional[SpanType] = SpanType.SPAN,
) -> Callable[P, R]:
    module_name = func.__name__
    is_async = inspect.iscoroutinefunction(func)
    span_name = name if name is not None else func.__name__

    if is_async:

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Push entity before span starts so processors can capture it
            SessionManager.push_entity(entity_type, span_name)

            tracer = trace.get_tracer(module_name)
            span = tracer.start_span(span_name)
            # Set span type if provided

            if not isinstance(as_type, SpanType):
                logger.error("Invalid span type: %s", as_type)
                return
            try:
                span.set_attribute("netra.span.type", as_type.value)
            except Exception:
                pass
            # Register and activate span
            try:
                SessionManager.register_span(span_name, span)
                SessionManager.set_current_span(span)
            except Exception:
                logger.exception("Failed to register span '%s' with SessionManager", span_name)

            with trace.use_span(span, end_on_exit=False):
                _add_span_attributes(span, func, args, kwargs, entity_type)
                try:
                    result = await cast(Awaitable[Any], func(*args, **kwargs))
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                    span.end()
                    try:
                        SessionManager.unregister_span(span_name, span)
                    except Exception:
                        logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                    SessionManager.pop_entity(entity_type)
                    raise

            # If result is streaming, defer span end to when stream completes
            if _is_streaming_response(result):
                return _wrap_streaming_response_with_span(span, result, span_name, entity_type)
            if _is_async_generator(result):
                return _wrap_async_generator_with_span(span, result, span_name, entity_type)
            if _is_sync_generator(result):
                return _wrap_sync_generator_with_span(span, result, span_name, entity_type)

            # Non-streaming: finalize now
            _add_output_attributes(span, result)
            span.end()
            try:
                SessionManager.unregister_span(span_name, span)
            except Exception:
                logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
            SessionManager.pop_entity(entity_type)
            return result

        return cast(Callable[P, R], async_wrapper)

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Push entity before span starts so processors can capture it
            SessionManager.push_entity(entity_type, span_name)

            tracer = trace.get_tracer(module_name)
            span = tracer.start_span(span_name)
            # Set span type if provided
            if as_type is not None:
                if not isinstance(as_type, SpanType):
                    logger.error("Invalid span type: %s", as_type)
                    return
                try:
                    span.set_attribute("netra.span.type", as_type.value)
                except Exception:
                    pass
            # Register and activate span
            try:
                SessionManager.register_span(span_name, span)
                SessionManager.set_current_span(span)
            except Exception:
                logger.exception("Failed to register span '%s' with SessionManager", span_name)

            with trace.use_span(span, end_on_exit=False):
                _add_span_attributes(span, func, args, kwargs, entity_type)
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(e))
                    span.record_exception(e)
                    span.end()
                    try:
                        SessionManager.unregister_span(span_name, span)
                    except Exception:
                        logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
                    SessionManager.pop_entity(entity_type)
                    raise

            # If result is streaming, defer span end to when stream completes
            if _is_streaming_response(result):
                return _wrap_streaming_response_with_span(span, result, span_name, entity_type)
            if _is_async_generator(result):
                return _wrap_async_generator_with_span(span, result, span_name, entity_type)  # type: ignore[arg-type]
            if _is_sync_generator(result):
                return _wrap_sync_generator_with_span(span, result, span_name, entity_type)  # type: ignore[arg-type]

            # Non-streaming: finalize now
            _add_output_attributes(span, result)
            span.end()
            try:
                SessionManager.unregister_span(span_name, span)
            except Exception:
                logger.exception("Failed to unregister span '%s' from SessionManager", span_name)
            SessionManager.pop_entity(entity_type)
            return result

        return cast(Callable[P, R], sync_wrapper)


def _wrap_class_methods(
    cls: C,
    entity_type: str,
    name: Optional[str] = None,
    as_type: Optional[SpanType] = SpanType.SPAN,
) -> C:
    class_name = name if name is not None else cls.__name__
    for attr_name in cls.__dict__:
        attr = getattr(cls, attr_name)
        if attr_name.startswith("_"):
            continue
        if callable(attr) and inspect.isfunction(attr):
            method_span_name = f"{class_name}.{attr_name}"
            wrapped_method = _create_function_wrapper(attr, entity_type, method_span_name, as_type=as_type)
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
            return _wrap_class_methods(cast(C, obj), "agent", name, as_type=SpanType.AGENT)
        else:
            return _create_function_wrapper(cast(Callable[P, R], obj), "agent", name, as_type=SpanType.AGENT)

    if target is not None:
        return decorator(target)
    return decorator


def task(
    target: Union[Callable[P, R], C, None] = None, *, name: Optional[str] = None
) -> Union[Callable[P, R], C, Callable[[Callable[P, R]], Callable[P, R]]]:
    def decorator(obj: Union[Callable[P, R], C]) -> Union[Callable[P, R], C]:
        if inspect.isclass(obj):
            return _wrap_class_methods(cast(C, obj), "task", name, as_type=SpanType.TOOL)
        else:
            # When obj is a function, it should be type Callable[P, R]
            return _create_function_wrapper(cast(Callable[P, R], obj), "task", name, as_type=SpanType.TOOL)

    if target is not None:
        return decorator(target)
    return decorator


def span(
    target: Union[Callable[P, R], C, None] = None,
    *,
    name: Optional[str] = None,
    as_type: Optional[SpanType] = SpanType.SPAN,
) -> Union[Callable[P, R], C, Callable[[Callable[P, R]], Callable[P, R]]]:
    def decorator(obj: Union[Callable[P, R], C]) -> Union[Callable[P, R], C]:
        if inspect.isclass(obj):
            return _wrap_class_methods(cast(C, obj), "span", name, as_type=as_type)
        else:
            # When obj is a function, it should be type Callable[P, R]
            return _create_function_wrapper(cast(Callable[P, R], obj), "span", name, as_type=as_type)

    if target is not None:
        return decorator(target)
    return decorator
