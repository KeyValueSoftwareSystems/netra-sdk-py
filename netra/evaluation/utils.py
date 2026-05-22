"""Utility functions for the evaluation module."""

import asyncio
import logging
import os
import threading
from typing import Any, Awaitable, Callable, Optional, TypeVar

from opentelemetry import baggage
from opentelemetry import context as otel_context

from netra.evaluation.constants import LOG_PREFIX
from netra.evaluation.models import DatasetRecord, EvaluatorConfig, EvaluatorContext, ItemContext

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


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


def get_session_id_from_baggage() -> Optional[str]:
    """Get the session ID from the OpenTelemetry baggage.

    Returns:
        The session ID if found, None otherwise.
    """
    ctx = otel_context.get_current()
    session_id = baggage.get_baggage("session_id", ctx)
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def format_trace_id(trace_id: int) -> str:
    """Format the trace ID as a 32-digit hexadecimal string.

    Args:
        trace_id: The integer trace ID to format.

    Returns:
        The formatted trace ID.
    """
    return f"{trace_id:032x}"


def format_span_id(span_id: int) -> str:
    """Format the span ID as a 16-digit hexadecimal string.

    Args:
        span_id: The integer span ID to format.

    Returns:
        The formatted span ID.
    """
    return f"{span_id:016x}"


def run_async_safely(coro: Awaitable[_T]) -> _T:
    """Run an async coroutine from synchronous code.

    When called from a context that already has a running event loop (e.g. a
    Jupyter notebook, or an async framework like FastAPI), ``asyncio.run()``
    would raise.  In that case we spin up a **new daemon thread** with its own
    event loop so the caller's loop is never blocked or re-entered.

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
        result_holder: dict[str, _T] = {}
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


def extract_evaluator_config(evaluator: Any) -> Optional[EvaluatorConfig]:
    """Extract evaluator configuration from an evaluator object.

    Args:
        evaluator: The evaluator object.

    Returns:
        The evaluator configuration if found, None otherwise.
    """
    if not hasattr(evaluator, "config"):
        return None
    config = evaluator.config
    if not isinstance(config, EvaluatorConfig):
        return None
    return config


async def execute_task(task: Callable[[Any], Any], item_input: Any) -> tuple[Any, str]:
    """Execute a task function (sync or async) and return (output, status).

    Args:
        task: The task function to execute.
        item_input: The input to the task function.

    Returns:
        A tuple of (task_output, status_string).
    """
    try:
        result = task(item_input)
        if asyncio.iscoroutine(result):
            result = await result
        return result, "completed"
    except Exception as exc:
        return str(exc), "failed"


async def run_single_evaluator(
    evaluator: Any,
    item_input: Any,
    task_output: Any,
    expected_output: Any,
    metadata: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Run a single evaluator and return normalized result.

    Args:
        evaluator: The evaluator object.
        item_input: The input to the task function.
        task_output: The output of the task function.
        expected_output: The expected output of the task function.
        metadata: Optional metadata to be passed to the evaluator.

    Returns:
        The normalized result dict if successful, None otherwise.
    """
    if not hasattr(evaluator, "evaluate"):
        return None

    expected_name = None
    config = extract_evaluator_config(evaluator)
    if config:
        expected_name = config.name

    context = EvaluatorContext(
        input=item_input,
        task_output=task_output,
        expected_output=expected_output,
        metadata=metadata,
    )
    result = evaluator.evaluate(context)
    if asyncio.iscoroutine(result):
        result = await result

    result_payload = {
        "evaluatorName": result.evaluator_name,
        "result": result.result,
        "isPassed": result.is_passed,
        "reason": result.reason,
    }

    if expected_name and result_payload.get("evaluatorName") != expected_name:
        return None

    return result_payload


def build_item_payload(
    ctx: ItemContext,
    status: str,
    include_output: bool = False,
) -> dict[str, Any]:
    """Build a payload dict for posting item status.

    Args:
        ctx: The item context.
        status: The status of the item (e.g. "completed", "failed").
        include_output: Whether to include the task output in the payload.

    Returns:
        The payload dict ready for HTTP submission.
    """
    payload: dict[str, Any] = {
        "traceId": ctx.trace_id,
        "sessionId": ctx.session_id,
    }

    if ctx.dataset_item_id:
        payload["datasetItemId"] = ctx.dataset_item_id
    else:
        payload["input"] = ctx.item_input
        payload["expectedOutput"] = ctx.expected_output
        if ctx.metadata:
            payload["metadata"] = ctx.metadata

    if status == "failed":
        payload["status"] = "failed"
        return payload

    if include_output:
        payload["taskOutput"] = ctx.task_output

    return payload


def validate_run_inputs(
    name: str,
    dataset: Any,
    task: Callable[[Any], Any],
) -> None:
    """Validate required inputs for run_test_suite.

    Args:
        name: The name of the run.
        dataset: The dataset to be used for the test suite.
        task: The task to be executed for each item in the dataset.

    Raises:
        ValueError: If any required input is missing or invalid.
    """
    if not name:
        raise ValueError(f"{LOG_PREFIX}: run name is required")
    if not dataset:
        raise ValueError(f"{LOG_PREFIX}: dataset is required")
    if task is None:
        raise ValueError(f"{LOG_PREFIX}: task function is required")


def extract_dataset_id(items: list[Any]) -> Optional[str]:
    """Extract dataset_id from items if they are DatasetRecords.

    Args:
        items: List of items.

    Returns:
        The dataset_id if found, None otherwise.
    """
    if items and isinstance(items[0], DatasetRecord):
        dataset_id: str = items[0].dataset_id
        return dataset_id
    return None


def build_evaluators_config(
    evaluators: Optional[list[Any]],
) -> list[EvaluatorConfig]:
    """Build evaluator configurations from evaluator objects.

    Args:
        evaluators: List of evaluators.

    Returns:
        List of evaluator configurations.
    """
    configs: list[EvaluatorConfig] = []
    if not evaluators:
        return configs

    for evaluator in evaluators:
        config = extract_evaluator_config(evaluator)
        if not config:
            continue
        try:
            configs.append(config)
        except Exception as exc:
            logger.warning("%s: Failed to extract evaluator config: %s", LOG_PREFIX, exc)
            continue
    return configs
