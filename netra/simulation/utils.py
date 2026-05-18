"""Utility functions for the simulation module."""

import asyncio
import base64
import inspect
import logging
import os
import threading
from typing import Awaitable, Optional, Tuple, TypeVar

import httpx

from netra.simulation.models import FileData, ProcessedFile, TaskResult
from netra.simulation.task import BaseTask

logger = logging.getLogger(__name__)

T = TypeVar("T")

_LOG_PREFIX = "netra.simulation"
_DEFAULT_FILE_DOWNLOAD_TIMEOUT = 30.0


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


def _get_file_download_timeout() -> float:
    """Get file download timeout from environment or use default.

    Returns:
        The timeout value in seconds.
    """
    timeout_str = os.getenv("NETRA_SIMULATION_FILE_DOWNLOAD_TIMEOUT")
    if not timeout_str:
        return _DEFAULT_FILE_DOWNLOAD_TIMEOUT
    try:
        return float(timeout_str)
    except ValueError:
        logger.warning(
            "%s: Invalid file download timeout '%s', using default %.1f",
            _LOG_PREFIX,
            timeout_str,
            _DEFAULT_FILE_DOWNLOAD_TIMEOUT,
        )
        return _DEFAULT_FILE_DOWNLOAD_TIMEOUT


def process_files(files: list[FileData]) -> list[ProcessedFile]:
    """Download files from pre-signed URLs and base64-encode their content.

    Each file is downloaded individually.  If any file fails to download, the
    entire batch is aborted with a ``RuntimeError`` so that file-aware tasks
    never receive a partial file list.

    Args:
        files: List of FileData objects containing download URLs.

    Returns:
        List of ProcessedFile objects with base64-encoded data.

    Raises:
        RuntimeError: If a file download or encoding fails.
    """
    if not files:
        return []

    timeout = _get_file_download_timeout()
    processed: list[ProcessedFile] = []

    for file_data in files:
        try:
            response = httpx.get(file_data.download_url, timeout=timeout)
            response.raise_for_status()
            encoded = base64.b64encode(response.content).decode("ascii")
            processed.append(
                ProcessedFile(
                    file_name=file_data.file_name,
                    content_type=file_data.content_type,
                    description=file_data.description,
                    data=encoded,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download file '{file_data.file_name}': {exc}") from exc

    return processed


def _task_accepts_files(task: BaseTask) -> bool:
    """Check whether the task's run() method accepts a 'files' parameter.

    Used for backward compatibility so that existing BaseTask subclasses that
    do not declare the files parameter are not broken.

    Args:
        task: The BaseTask instance to inspect.

    Returns:
        True if the run() method has a 'files' parameter or **kwargs.
    """
    try:
        sig = inspect.signature(task.run)
        params = sig.parameters
        if "files" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False


async def execute_task(
    task: BaseTask,
    message: str,
    session_id: Optional[str],
    raw_files: Optional[list[FileData]] = None,
) -> Tuple[str, Optional[str]]:
    """Execute a task's run method (sync or async) and extract message and session_id.

    Files are only downloaded and base64-encoded when the task's run() method
    actually accepts a ``files`` parameter, avoiding unnecessary network I/O
    for legacy tasks.

    Args:
        task: The BaseTask instance to execute.
        message: The input message to pass to the task.
        session_id: The current session identifier.
        raw_files: Raw file metadata from the backend. Downloads are deferred
            until we confirm the task can accept them.

    Returns:
        A tuple of (response_message, session_id).

    Raises:
        ValueError: If the task returns an unsupported type.
    """
    kwargs: dict[str, object] = {"message": message, "session_id": session_id}
    if raw_files and _task_accepts_files(task):
        kwargs["files"] = process_files(raw_files)

    result = task.run(**kwargs)  # type: ignore[arg-type]
    if asyncio.iscoroutine(result):
        result = await result

    if isinstance(result, TaskResult):
        return result.message, result.session_id

    raise ValueError(f"Task must return TaskResult, got {type(result).__name__}")
