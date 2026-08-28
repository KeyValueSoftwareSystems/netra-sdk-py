"""Utility functions for the simulation module."""

import asyncio
import base64
import concurrent.futures
import inspect
import logging
import os
from typing import Any, Optional

import httpx

from netra.simulation.constants import (
    DEFAULT_FILE_DOWNLOAD_TIMEOUT,
    ENV_FILE_DOWNLOAD_TIMEOUT,
    LOG_PREFIX,
    MAX_FILE_DOWNLOAD_WORKERS,
)
from netra.simulation.models import FileData, ProcessedFile, TaskResult
from netra.simulation.task import BaseTask
from netra.utils import run_async_safely as run_async_safely  # re-exported for backwards compatibility

logger = logging.getLogger(__name__)


def parse_env_float(env_var: str, default: float) -> float:
    """Read an environment variable and parse it as a float.

    Args:
        env_var: Name of the environment variable.
        default: Value to return when the variable is unset or invalid.

    Returns:
        The parsed float, or *default* on failure.
    """
    raw = os.getenv(env_var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s: Invalid value '%s' for %s, using default %.1f",
            LOG_PREFIX,
            raw,
            env_var,
            default,
        )
        return default


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


def _download_single_file(file_data: FileData, timeout: float) -> ProcessedFile:
    """Download a single file and base64-encode its content.

    Args:
        file_data: Metadata for the file to download.
        timeout: HTTP request timeout in seconds.

    Returns:
        A ProcessedFile with the base64-encoded content.

    Raises:
        RuntimeError: If the download or encoding fails.
    """
    try:
        response = httpx.get(file_data.download_url, timeout=timeout)
        response.raise_for_status()
        encoded = base64.b64encode(response.content).decode("ascii")
        return ProcessedFile(
            file_name=file_data.file_name,
            content_type=file_data.content_type,
            description=file_data.description,
            data=encoded,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download file '{file_data.file_name}': {exc}") from exc


def process_files(files: list[FileData]) -> list[ProcessedFile]:
    """Download files from pre-signed URLs and base64-encode their content.

    Downloads run concurrently via a thread pool.  If any file fails, the
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

    timeout = parse_env_float(ENV_FILE_DOWNLOAD_TIMEOUT, DEFAULT_FILE_DOWNLOAD_TIMEOUT)

    if len(files) == 1:
        return [_download_single_file(files[0], timeout)]

    max_workers = min(MAX_FILE_DOWNLOAD_WORKERS, len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_single_file, fd, timeout) for fd in files]
        return [f.result() for f in futures]


async def execute_task(
    task: BaseTask,
    message: str,
    session_id: Optional[str],
    raw_files: Optional[list[FileData]] = None,
    setup_context: Optional[dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    """Execute a task's run method (sync or async) and extract message and session_id.

    ``setup_context`` is forwarded only when the task's ``run`` method accepts
    it as a parameter, keeping backwards compatibility with existing tasks.

    Args:
        task: The BaseTask instance to execute.
        message: The input message to pass to the task.
        session_id: The current session identifier.
        raw_files: Raw file metadata from the backend.
        setup_context: Optional dict from ``before_all`` / ``before`` hooks.

    Returns:
        A tuple of (response_message, session_id).

    Raises:
        ValueError: If the task returns an unsupported type.
    """
    processed_files = process_files(raw_files) if raw_files else None

    # Forward setup_context only if the task's run() declares it
    sig = inspect.signature(task.run)
    if "setup_context" in sig.parameters:
        result = task.run(
            message=message,
            session_id=session_id,
            files=processed_files,
            setup_context=setup_context,
        )
    else:
        result = task.run(message=message, session_id=session_id, files=processed_files)

    if asyncio.iscoroutine(result):
        result = await result

    if isinstance(result, TaskResult):
        return result.message, result.session_id

    raise ValueError(f"Task must return TaskResult, got {type(result).__name__}")
