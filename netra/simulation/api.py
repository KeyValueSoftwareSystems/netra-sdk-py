import asyncio
import concurrent.futures
import logging
import time
from typing import Any, Optional

from netra.config import Config
from netra.simulation.client import SimulationHttpClient
from netra.simulation.constants import DEFAULT_MAX_TURNS, LOG_PREFIX, SPAN_NAME
from netra.simulation.models import ConversationStatus, FileData, SimulationItem
from netra.simulation.task import BaseTask
from netra.simulation.utils import (
    execute_task,
    format_trace_id,
    run_async_safely,
    validate_simulation_inputs,
)
from netra.span_wrapper import SpanWrapper

logger = logging.getLogger(__name__)


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

    def close(self) -> None:
        """Release resources held by the simulation client."""
        self._client.close()

    def run_simulation(
        self,
        name: str,
        dataset_id: str,
        task: BaseTask,
        context: Optional[dict[str, Any]] = None,
        max_concurrency: int = 5,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> Optional[dict[str, Any]]:
        """Run a multi-turn conversation simulation.

        Args:
            name: Name of the simulation run.
            dataset_id: Identifier of the dataset to simulate.
            task: A BaseTask instance whose run() method receives (message, session_id, files)
                and returns TaskResult. Can be sync or async.
            context: Optional context data for the simulation.
            max_concurrency: Maximum parallel executions (default: 5).
            max_turns: Maximum conversation turns per item before aborting (default: 50).

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
        simulation_items = run_result.get("simulation_items")
        if not simulation_items:
            logger.error("%s: No items returned from create_run", LOG_PREFIX)
            return None

        logger.info("%s: Starting simulation with %d items", LOG_PREFIX, len(simulation_items))
        try:
            result = run_async_safely(
                self._run_simulation_async(
                    run_id, simulation_items, task, max_concurrency, max_turns  # type:ignore[arg-type]
                )
            )

            elapsed_time = time.time() - start_time
            logger.info("%s: Simulation completed in %.2f seconds", LOG_PREFIX, elapsed_time)
            self._client.post_run_status(run_id, "completed")  # type:ignore[arg-type]
            return result
        except Exception:
            logger.exception("%s: Run simulation failed", LOG_PREFIX)
            self._client.post_run_status(run_id, "failed")  # type:ignore[arg-type]
            return None

    async def _run_simulation_async(
        self,
        run_id: str,
        simulation_items: list[SimulationItem],
        task: BaseTask,
        max_concurrency: int,
        max_turns: int,
    ) -> dict[str, Any]:
        """Orchestrate concurrent simulation execution.

        Each simulation item is dispatched to a thread via ``run_in_executor``.
        Inside each thread, ``run_async_safely`` creates a **new** event loop
        so that async user tasks (``BaseTask.run``) work correctly without
        nesting into the orchestrator's loop.  This two-level design lets us
        honour ``max_concurrency`` while supporting both sync and async tasks
        transparently.

        Args:
            run_id: The simulation run identifier.
            simulation_items: List of simulation items to process.
            task: The BaseTask instance to execute (sync or async).
            max_concurrency: Maximum concurrent executions.
            max_turns: Maximum conversation turns per item.

        Returns:
            Dictionary with simulation results.
        """
        results: dict[str, Any] = {
            "success": True,
            "completed": [],
            "failed": [],
            "total_items": len(simulation_items),
        }
        processed_count = 0
        lock = asyncio.Lock()

        loop = asyncio.get_running_loop()

        def run_item_in_thread(run_item: SimulationItem) -> dict[str, Any]:
            """Run a single simulation item in a dedicated thread/event-loop.

            Args:
                run_item: The simulation item to run.

            Returns:
                Dictionary with simulation result.
            """
            return run_async_safely(self._execute_conversation(run_id, run_item, task, max_turns))

        async def process_item(run_item: SimulationItem) -> None:
            """Process a single simulation item and record its outcome.

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
                    LOG_PREFIX,
                    processed_count,
                    len(simulation_items),
                    run_item.run_item_id,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            tasks = [asyncio.create_task(process_item(item)) for item in simulation_items]
            try:
                await asyncio.gather(*tasks)
            except (asyncio.CancelledError, KeyboardInterrupt):
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                executor.shutdown(wait=False, cancel_futures=True)
        logger.info(
            "%s: Completed=%d, Failed=%d",
            LOG_PREFIX,
            len(results["completed"]),
            len(results["failed"]),
        )
        return results

    async def _execute_conversation(
        self,
        run_id: str,
        run_item: SimulationItem,
        task: BaseTask,
        max_turns: int,
    ) -> dict[str, Any]:
        """Execute a multi-turn conversation for a single simulation item.

        Args:
            run_id: The simulation run identifier.
            run_item: The simulation item to process.
            task: The BaseTask instance to execute (sync or async).
            max_turns: Safety limit on the number of conversation turns.

        Returns:
            Dictionary with execution result including success status.
        """
        run_item_id = run_item.run_item_id
        message = run_item.message
        turn_id = run_item.turn_id
        raw_files: list[FileData] = run_item.files
        session_id: Optional[str] = None

        for turn_number in range(1, max_turns + 1):
            try:
                with SpanWrapper(SPAN_NAME, module_name=LOG_PREFIX) as span:
                    trace_id = ""
                    otel_span = span.get_current_span()
                    if otel_span:
                        span_context = otel_span.get_span_context()
                        trace_id = format_trace_id(span_context.trace_id)

                    response_message, task_session_id = await execute_task(
                        task, message, session_id, raw_files=raw_files
                    )
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

                if response.decision == ConversationStatus.STOP:
                    logger.info(
                        "%s: Completed run_item_id=%s reason=%s",
                        LOG_PREFIX,
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
                raw_files = response.next_files

            except Exception as exc:
                error_msg = str(exc)
                logger.exception(
                    "%s: Task failed run_item_id=%s, turn_id=%s: %s",
                    LOG_PREFIX,
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

        error_msg = f"Exceeded maximum turns ({max_turns}) for run_item_id={run_item_id}"
        logger.error("%s: %s", LOG_PREFIX, error_msg)
        self._client.report_failure(run_id=run_id, run_item_id=run_item_id, error=error_msg)
        return {
            "run_item_id": run_item_id,
            "success": False,
            "error": error_msg,
            "turn_id": turn_id,
        }
