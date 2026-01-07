import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from opentelemetry import baggage
from opentelemetry import context as otel_context

from netra.evaluation.models import DatasetRecord, EvaluatorConfig, EvaluatorContext, ItemContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_session_id_from_baggage() -> Optional[str]:
    """
    Get the session ID from the OpenTelemetry baggage.

    Returns:
        session_id: The session ID if found, None otherwise.
    """
    ctx = otel_context.get_current()
    session_id = baggage.get_baggage("session_id", ctx)
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def format_trace_id(trace_id: int) -> str:
    """
    Format the trace ID as a 32-digit hexadecimal string.

    Return:
        trace_id: The formatted trace ID.
    """
    return f"{trace_id:032x}"


def format_span_id(span_id: int) -> str:
    """
    Format the span ID as a 16-digit hexadecimal string.

    Return:
        span_id: The formatted span ID.
    """
    return f"{span_id:016x}"


async def run_callable_maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Run callable
    """
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def run_async_safely(coroutine: Awaitable[T]) -> T:
    """Run an async coroutine from sync code.

    If there is already an event loop running in this thread, we execute in a
    dedicated thread to avoid 'asyncio.run() cannot be called from a running event loop'.

    Args:
        coroutine: The coroutine to run.

    Returns:
        The result of the coroutine.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_container: Dict[str, T] = {}
        error_container: Dict[str, Exception] = {}

        def _runner() -> None:
            try:
                result_container["result"] = asyncio.run(coroutine)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                error_container["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_container:
            raise error_container["error"]
        return result_container.get("result")  # type: ignore[return-value]

    return asyncio.run(coroutine)  # type: ignore[arg-type]


def extract_evaluator_config(evaluator: Any) -> Optional[EvaluatorConfig]:
    """
    Extract evaluator configuration from an evaluator object.

    Args:
        evaluator: The evaluator object.

    Returns:
        Optional[EvaluatorConfig]: The evaluator configuration if found, None otherwise.
    """
    if not hasattr(evaluator, "config"):
        return None
    config = evaluator.config
    if not isinstance(config, EvaluatorConfig):
        return None
    return config


async def execute_task(task: Callable[[Any], Any], item_input: Any) -> tuple[Any, str]:
    """
    Execute a task function (sync or async) and return (output, status).

    Args:
        task: The task function to execute.
        item_input: The input to the task function.

    Returns:
        tuple[Any, str]: The output of the task function and the status of the execution.
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
    metadata: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Run a single evaluator and return normalized result.

    Args:
        evaluator: The evaluator object.
        item_input: The input to the task function.
        task_output: The output of the task function.
        expected_output: The expected output of the task function.
        metadata: Optional metadata to be passed to the evaluator.

    Returns:
        Optional[Dict[str, Any]]: The normalized result of the evaluator if successful, None otherwise.
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

    if hasattr(result, "model_dump"):
        result_dict = result.model_dump()
    elif isinstance(result, dict):
        result_dict = result
    else:
        return None

    if expected_name and result_dict.get("evaluator_name") != expected_name:
        return None

    return result_dict  # type: ignore[no-any-return]


def build_item_payload(
    ctx: "ItemContext",
    status: str,
    include_output: bool = False,
) -> Dict[str, Any]:
    """
    Build a payload dict for posting item status.

    Args:
        ctx: The item context.
        status: The status of the item.
        include_output: Whether to include the task output in the payload.

    Returns:
        Dict[str, Any]: The payload dict.
    """
    payload: Dict[str, Any] = {
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

    if include_output:
        payload["taskOutput"] = ctx.task_output

    return payload


def validate_run_inputs(
    name: str,
    data: Any,
    task: Callable[[Any], Any],
) -> bool:
    """
    Validate required inputs for run_test_suite.

    Args:
        name: The name of the run.
        data: The dataset to be used for the test suite.
        task: The task to be executed for each item in the dataset.

    Returns:
        bool: True if all inputs are valid, False otherwise.
    """
    if not name:
        logger.error("netra.evaluation: run name is required")
        return False
    if not data:
        logger.error("netra.evaluation: data is required")
        return False
    if task is None:
        logger.error("netra.evaluation: task function is required")
        return False
    return True


def extract_dataset_id(items: List[Any]) -> Optional[str]:  # noqa: E501
    """
    Extract dataset_id from items if they are DatasetRecords.

    Args:
        items: List of items.

    Returns:
        Optional[str]: The dataset_id if found, None otherwise.
    """
    if items and isinstance(items[0], DatasetRecord):
        dataset_id: str = items[0].dataset_id
        return dataset_id
    return None


def build_evaluators_config(
    evaluators: Optional[List[Any]],
) -> List[EvaluatorConfig]:
    """
    Build evaluator configurations from evaluator objects.

    Args:
        evaluators: List of evaluators.

    Returns:
        List[EvaluatorConfig]: List of evaluator configurations.
    """
    configs: List[EvaluatorConfig] = []
    if not evaluators:
        return configs

    for evaluator in evaluators:
        config = extract_evaluator_config(evaluator)
        if not config:
            continue
        try:
            configs.append(config)
        except Exception:
            continue
    return configs
