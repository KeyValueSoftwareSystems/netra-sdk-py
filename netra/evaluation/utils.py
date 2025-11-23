import asyncio
import threading
from typing import Any, Coroutine, Dict, Optional, TypeVar

from opentelemetry import baggage, trace

T = TypeVar("T")


def run_async_safely(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run the given coroutine asynchronously and return the result.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        result: Dict[str, T] = {}
        err: Dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as e:  # noqa: BLE001
                err["e"] = e

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if "e" in err:
            raise err["e"]
        return result["value"]
    return asyncio.run(coro)


def get_session_id_from_baggage() -> Optional[str]:
    """
    Get the session id from the baggage.

    Returns:
        The session id.
    """
    try:
        value = baggage.get_baggage("session_id")
        return value if isinstance(value, str) else None
    except Exception:
        return None


def get_trace_id_from_span(span: trace.Span) -> str:
    """
    Get the trace id from the span.

    Args:
        span: The span to get the trace id from.

    Returns:
        The trace id.
    """
    ctx = span.get_span_context()
    return f"{ctx.trace_id:032x}"
