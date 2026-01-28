import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar

from netra.simulation.models import TaskResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


def validate_simulation_inputs(dataset_id: str, task: Callable[[Any], Any]) -> bool:
    """Validate required inputs for simulation."""
    if not dataset_id:
        logger.error("netra.simulation: dataset_id is required")
        return False
    if not callable(task):
        logger.error("netra.simulation: task must be a callable")
        return False
    return True


def run_async_safely(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine from sync code.

    If an event loop is already running, executes in a dedicated thread
    to avoid 'asyncio.run() cannot be called from a running event loop'.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_holder: Dict[str, T] = {}
        error_holder: Dict[str, Exception] = {}

        def runner() -> None:
            try:
                result_holder["value"] = asyncio.run(coro)  # type: ignore[arg-type]
            except Exception as exc:
                error_holder["exc"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "exc" in error_holder:
            raise error_holder["exc"]
        return result_holder.get("value")  # type: ignore[return-value]

    return asyncio.run(coro)  # type: ignore[arg-type]


async def execute_task(
    task: Callable[[str, Optional[str]], Any],
    message: str,
    session_id: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Execute a task function (sync or async) and extract message and session_id."""
    result = task(message, session_id)
    if asyncio.iscoroutine(result):
        result = await result

    if isinstance(result, TaskResult):
        return result.message, result.session_id
    if isinstance(result, dict):
        return result.get("message", ""), result.get("session_id", session_id)

    raise ValueError(f"Task must return TaskResult or dict, got {type(result).__name__}")
