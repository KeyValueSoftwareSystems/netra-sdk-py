"""Public API for running multi-turn conversation simulations."""

import asyncio
import concurrent.futures
import logging
import time
from typing import Any, Optional

from netra.config import Config
from netra.simulation.client import SimulationHttpClient
from netra.simulation.models import SimulationItem
from netra.simulation.task import BaseTask
from netra.simulation.utils import (
    execute_task,
    format_trace_id,
    run_async_safely,
    validate_simulation_inputs,
)
from netra.span_wrapper import SpanWrapper

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.simulation"
_SPAN_NAME = "Netra.Simulation.TestRun"


class Simulation:
    """Public API for running multi-turn conversation simulations.

    Attributes:
        _config: The Netra configuration object.
        _client: The HTTP client for simulation API calls.
    """

    __slots__ = ("_config", "_client")

    def __init__(self, config: Config) -> None:
        """Initialize the Simulation instance.

        Args:
            config: The Netra configuration object.
        """
        self._config = config
        self._client = SimulationHttpClient(config)

    def run_simulation(
        self,
        name: str,
        dataset_id: str,
        task: BaseTask,
        context: Optional[dict[str, Any]] = None,
        max_concurrency: int = 5,
    ) -> Optional[dict[str, Any]]:
        """Run a multi-turn conversation simulation.

        Args:
            name: Name of the simulation run.
            dataset_id: Identifier of the dataset to simulate.
            task: A BaseTask instance whose run() method receives (message, session_id)
                and returns TaskResult. Can be sync or async.
            context: Optional context data for the simulation.
            max_concurrency: Maximum parallel executions (default: 5).

        Returns:
            Dictionary with simulation results, or None on failure.
        """
        if not validate_simulation_inputs(dataset_id, task):
            return None

        start_time = time.time()
        run_result = self._client.create_run(
            name=name,
            dataset_id=dataset_id,
            context=context or {},
        )
        if not run_result:
            return None

        run_id = run_result.get("run_id")
        run_items = run_result.get("simulation_items")
        if not run_items:
            logger.error("%s: No items returned from create_run", _LOG_PREFIX)
            return None

        logger.info("%s: Starting simulation with %d items", _LOG_PREFIX, len(run_items))
        try:
            result = run_async_safely(
                self._run_simulation_async(run_id, run_items, task, max_concurrency)  # type:ignore[arg-type]
            )

            elapsed_time = time.time() - start_time
            logger.info("%s: Simulation completed in %.2f seconds", _LOG_PREFIX, elapsed_time)
            self._client.post_run_status(run_id, "completed")  # type:ignore[arg-type]
            return result
        except BaseException:
            logger.error("%s: Run simulation failed", _LOG_PREFIX)
            self._client.post_run_status(run_id, "failed")  # type:ignore[arg-type]
            return None

    async def _run_simulation_async(
        self,
        run_id: str,
        run_items: list[SimulationItem],
        task: BaseTask,
        max_concurrency: int,
    ) -> dict[str, Any]:
        """Async implementation of run_simulation with semaphore-based concurrency.

        Args:
            run_id: The simulation run identifier.
            run_items: List of simulation items to process.
            task: The BaseTask instance to execute (sync or async).
            max_concurrency: Maximum concurrent executions.

        Returns:
            Dictionary with simulation results.
        """

        max_workers = min(5, max_concurrency)
        results: dict[str, Any] = {
            "success": True,
            "completed": [],
            "failed": [],
            "total_items": len(run_items),
        }
        processed_count = 0
        lock = asyncio.Lock()

        loop = asyncio.get_running_loop()

        def run_item_in_thread(run_item: SimulationItem) -> dict[str, Any]:
            """
            Run a single simulation item in a thread.

            Args:
                run_item: The simulation item to run.

            Returns:
                Dictionary with simulation result.
            """
            return run_async_safely(self._execute_conversation(run_id, run_item, task))

        async def process_item(run_item: SimulationItem) -> None:
            """
            Process a single simulation item and handle its completion.

            Args:
                run_item: The simulation item to process.
            """
            nonlocal processed_count
            result = await loop.run_in_executor(executor, run_item_in_thread, run_item)
            async with lock:
                target = results["completed"] if result["success"] else results["failed"]
                target.append(result)
                processed_count += 1
                logger.info(
                    "%s: %d/%d processed (run_item_id=%s)",
                    _LOG_PREFIX,
                    processed_count,
                    len(run_items),
                    run_item.run_item_id,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [asyncio.create_task(process_item(run_item)) for run_item in run_items]
            try:
                await asyncio.gather(*tasks)
            except (asyncio.CancelledError, KeyboardInterrupt):
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                executor.shutdown(wait=False, cancel_futures=True)
        logger.info(
            "%s: Completed=%d, Failed=%d",
            _LOG_PREFIX,
            len(results["completed"]),
            len(results["failed"]),
        )
        return results

    async def _execute_conversation(
        self,
        run_id: str,
        run_item: SimulationItem,
        task: BaseTask,
    ) -> Any:
        """Execute a multi-turn conversation for a single simulation item.

        Args:
            run_id: The simulation run identifier.
            run_item: The simulation item to process.
            task: The BaseTask instance to execute (sync or async).

        Returns:
            Dictionary with execution result including success status.
        """
        run_item_id = run_item.run_item_id
        message = run_item.message
        turn_id = run_item.turn_id
        session_id: Optional[str] = None

        while True:
            try:
                with SpanWrapper(_SPAN_NAME, module_name=_LOG_PREFIX) as span:
                    trace_id = ""
                    otel_span = span.get_current_span()
                    if otel_span:
                        span_context = otel_span.get_span_context()
                        trace_id = format_trace_id(span_context.trace_id)

                    response_message, task_session_id = await execute_task(task, message, session_id)
                    if task_session_id:
                        session_id = task_session_id

                response = self._client.trigger_conversation(
                    message=response_message,
                    turn_id=turn_id,
                    session_id=session_id or "",
                    trace_id=trace_id,
                )

                if response is None:
                    error_msg = "Failed to get conversation response"
                    return {
                        "run_item_id": run_item_id,
                        "success": False,
                        "error": error_msg,
                        "turn_id": turn_id,
                    }

                if response.decision == "stop":
                    logger.info(
                        "%s: Completed run_item_id=%s reason=%s",
                        _LOG_PREFIX,
                        run_item_id,
                        response.reason,
                    )
                    return {
                        "run_item_id": run_item_id,
                        "success": True,
                        "final_turn_id": turn_id,
                    }

                message = response.next_user_message  # type:ignore[assignment]
                turn_id = response.next_turn_id  # type:ignore[assignment]

            except Exception as exc:
                error_msg = str(exc)
                logger.error(
                    "%s: Task failed run_item_id=%s, turn_id=%s: %s",
                    _LOG_PREFIX,
                    run_item_id,
                    turn_id,
                    error_msg,
                )
                self._client.report_failure(run_id=run_id, run_item_id=run_item_id, error=error_msg)
                return {
                    "run_item_id": run_item_id,
                    "success": False,
                    "error": error_msg,
                    "turn_id": turn_id,
                }
