import logging
from typing import Any, AsyncIterator, Callable, Iterator, Tuple

from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from netra.instrumentation.libraries.codex.utils import (
    acquire_turn,
    end_join_span,
    finish_open_turns_for_client,
    finish_turn,
    handle_notification,
    record_exception,
    record_thread_info,
    register_turn,
    release_turn,
    start_turn,
)

logger = logging.getLogger(__name__)

# Only W3C trace context goes to the app-server's JSON-RPC `trace` field; baggage stays in-process.
_trace_context_propagator = TraceContextTextMapPropagator()


def _arg(args: Tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    """
    Read a call argument by position or keyword.

    Args:
        args (Tuple): Positional arguments.
        kwargs (dict): Keyword arguments.
        index (int): The argument's position (excluding self).
        name (str): The argument's keyword name.
        default (Any): Value to return when the argument is absent.

    Returns:
        Any: The argument value.
    """
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def thread_response_wrapper() -> Callable[..., Any]:
    """
    Return a wrapper for CodexClient.thread_start / thread_resume / thread_fork.

    The async client delegates to these sync methods, so one wrapper covers both APIs.

    Returns:
        Callable: A wrapt wrapper that records the resolved thread settings.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        response = wrapped(*args, **kwargs)
        try:
            record_thread_info(response)
        except Exception as e:
            logger.error("Failed to record codex thread settings: %s", e)
        return response

    return wrapper


def start_turn_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper for CodexClient._start_turn that opens the ``codex.turn`` span.

    ``_start_turn`` is the single entry point for ``Thread.turn``/``Thread.run`` and,
    via ``copy_context().run``, their async counterparts. The span is active while
    ``turn/start`` is written, so its context is injected into the request.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.

    Returns:
        Callable: A wrapt wrapper.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Raw client.turn_start() callers (for_handle=False) consume notifications without a
        # TurnHandle, so nothing would ever end the span; they still get trace context injected.
        if not _arg(args, kwargs, 3, "for_handle", False):
            return wrapped(*args, **kwargs)

        try:
            state = start_turn(
                tracer,
                instance,
                _arg(args, kwargs, 0, "thread_id"),
                _arg(args, kwargs, 1, "input_items"),
                _arg(args, kwargs, 2, "params"),
            )
        except Exception as e:
            logger.error("Failed to start codex turn span: %s", e)
            return wrapped(*args, **kwargs)

        try:
            with trace.use_span(state.span, end_on_exit=False):
                result = wrapped(*args, **kwargs)
        except BaseException as e:
            record_exception(state, e)
            finish_turn(state)
            raise

        try:
            turn_id = result[0].turn.id
            if not register_turn(state, turn_id):
                end_join_span(state)
        except Exception as e:
            logger.error("Failed to register codex turn: %s", e)
            finish_turn(state)
        return result

    return wrapper


def write_message_wrapper() -> Callable[..., Any]:
    """
    Return a wrapper for CodexClient._write_message that injects trace context into ``turn/start``.

    The app-server parents its request span (and the core turn spans under it) from the
    request's ``trace`` field. The SDK never sets it, and the ``TRACEPARENT`` env fallback
    is frozen at process spawn, so per-request injection is the only correct parent.

    Returns:
        Callable: A wrapt wrapper.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            payload = _arg(args, kwargs, 0, "payload")
            if isinstance(payload, dict) and payload.get("method") == "turn/start" and "trace" not in payload:
                carrier: dict[str, str] = {}
                _trace_context_propagator.inject(carrier)
                if traceparent := carrier.get("traceparent"):
                    payload = {
                        **payload,
                        "trace": {"traceparent": traceparent, "tracestate": carrier.get("tracestate")},
                    }
                    if args:
                        args = (payload, *args[1:])
                    else:
                        kwargs = {**kwargs, "payload": payload}
        except Exception as e:
            logger.error("Failed to inject trace context into codex turn/start: %s", e)
        return wrapped(*args, **kwargs)

    return wrapper


def close_wrapper() -> Callable[..., Any]:
    """
    Return a wrapper for CodexClient.close that ends turn spans whose handle was never streamed.

    Returns:
        Callable: A wrapt wrapper.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            finish_open_turns_for_client(instance)
        except Exception as e:
            logger.error("Failed to finish open codex turns on close: %s", e)
        return wrapped(*args, **kwargs)

    return wrapper


def stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper for TurnHandle.stream.

    ``Thread.run`` and ``TurnHandle.run`` both consume ``stream()``, so this sees every
    turn exactly once regardless of which API the caller uses.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.

    Returns:
        Callable: A wrapt wrapper returning a generator.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return _traced_stream(tracer, getattr(instance, "id", None), wrapped(*args, **kwargs))

    return wrapper


def _traced_stream(tracer: Tracer, turn_id: Any, iterator: Iterator[Any]) -> Iterator[Any]:
    """
    Yield notifications from a turn stream while recording them on the turn span.

    The consumer is registered on first ``next()``, so a stream that is created but never
    iterated does not hold the turn open (``CodexClient.close`` ends it instead).

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        turn_id (Any): The handle's turn id.
        iterator (Iterator): The original ``stream()`` generator.

    Returns:
        Iterator: The same notifications, unchanged.
    """
    state = acquire_turn(turn_id)
    token = object()
    try:
        for notification in iterator:
            if state is not None:
                handle_notification(tracer, state, notification, token)
            yield notification
    except GeneratorExit:
        raise
    except Exception as e:
        if state is not None:
            record_exception(state, e)
        raise
    finally:
        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        finally:
            if state is not None:
                release_turn(state, token)


def async_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper for AsyncTurnHandle.stream.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.

    Returns:
        Callable: A wrapt wrapper returning an async generator.
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return _traced_async_stream(tracer, getattr(instance, "id", None), wrapped(*args, **kwargs))

    return wrapper


async def _traced_async_stream(tracer: Tracer, turn_id: Any, aiterator: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """
    Async counterpart of ``_traced_stream``.

    Args:
        tracer (Tracer): The OpenTelemetry tracer.
        turn_id (Any): The handle's turn id.
        aiterator (AsyncIterator): The original ``stream()`` async generator.

    Returns:
        AsyncIterator: The same notifications, unchanged.
    """
    state = acquire_turn(turn_id)
    token = object()
    try:
        async for notification in aiterator:
            if state is not None:
                handle_notification(tracer, state, notification, token)
            yield notification
    except GeneratorExit:
        raise
    except Exception as e:
        if state is not None:
            record_exception(state, e)
        raise
    finally:
        try:
            aclose = getattr(aiterator, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            if state is not None:
                release_turn(state, token)
