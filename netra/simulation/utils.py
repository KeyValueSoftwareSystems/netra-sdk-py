"""Utility functions for the simulation module."""

import asyncio
import logging
import threading
from typing import Awaitable, Optional, Tuple, TypeVar

from netra.simulation.models import TaskResult
from netra.simulation.task import BaseTask

logger = logging.getLogger(__name__)

T = TypeVar("T")


def format_trace_id(trace_id: int) -> str:
    """Format the trace ID as a 32-digit hexadecimal string.

    Args:
        trace_id: The integer trace ID to format.

    Returns:
        The formatted trace ID as a hexadecimal string.
    """
    return f"{trace_id:032x}"


def validate_simulation_inputs(
    dataset_id: str,
    task: BaseTask,
) -> bool:
    """Validate required inputs for simulation.

    Args:
        dataset_id: The dataset identifier to validate.
        task: The BaseTask instance to validate.

    Returns:
        True if inputs are valid, False otherwise.
    """
    if not dataset_id:
        logger.error("netra.simulation: dataset_id is required")
        return False
    if not isinstance(task, BaseTask):
        logger.error("netra.simulation: task must be a BaseTask instance")
        return False
    return True


def run_async_safely(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync code.

    If an event loop is already running, executes in a dedicated thread
    to avoid 'asyncio.run() cannot be called from a running event loop'.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine execution.

    Raises:
        Exception: Re-raises any exception from the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_holder: dict[str, T] = {}
        error_holder: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result_holder["value"] = asyncio.run(coro)  # type: ignore[arg-type]
            except BaseException as exc:
                error_holder["exc"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "exc" in error_holder:
            raise error_holder["exc"]
        return result_holder.get("value")  # type: ignore[return-value]

    return asyncio.run(coro)  # type: ignore[arg-type]


async def execute_task(
    task: BaseTask,
    message: str,
    session_id: Optional[str],
) -> Tuple[str, Optional[str]]:
    """Execute a task's run method (sync or async) and extract message and session_id.

    Args:
        task: The BaseTask instance to execute.
        message: The input message to pass to the task.
        session_id: The current session identifier.

    Returns:
        A tuple of (response_message, session_id).

    Raises:
        ValueError: If the task returns an unsupported type.
    """
    result = task.run(message=message, session_id=session_id)
    if asyncio.iscoroutine(result):
        result = await result

    if isinstance(result, TaskResult):
        return result.message, result.session_id

    raise ValueError(f"Task must return TaskResult, got {type(result).__name__}")
