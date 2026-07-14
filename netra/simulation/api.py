import asyncio
import concurrent.futures
import logging
import time
from typing import Any, Optional

from netra.config import Config
from netra.simulation.client import SimulationHttpClient
from netra.simulation.constants import DEFAULT_MAX_TURNS, LOG_PREFIX, SPAN_NAME
from netra.simulation.hooks import (
    SimulationHooks,
    run_after,
    run_after_all,
    run_before,
    run_before_all,
)
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
        hooks: Optional[SimulationHooks] = None,
    ) -> Optional[dict[str, Any]]:
        """Run a multi-turn conversation simulation.

        Args:
            name: Name of the simulation run.
            dataset_id: Identifier of the dataset to simulate.
            task: A BaseTask instance whose run() method receives
                (message, session_id, files, setup_context) and returns
                TaskResult. Can be sync or async.
            context: Optional context data for the simulation.
            max_concurrency: Maximum parallel executions (default: 5).
            max_turns: Maximum conversation turns per item before aborting
                (default: 50).
            hooks: Optional :class:`~netra.simulation.hooks.SimulationHooks`
                with ``before_all``, ``before``, ``after``, and ``after_all``
                callables. Scripts live entirely on the SDK side; only
                lightweight metadata is forwarded to the backend for UI display.

                - ``before_all`` runs once before any scenario. If it raises,
                  the entire run is marked failed and no scenarios execute.
                - ``before`` runs before each scenario. If it raises, that
                  scenario is marked ``prescript_failed`` and skipped; other
                  scenarios continue.
                - ``after`` / ``after_all`` failures are logged but do not
                  affect scenario or run status.

        Returns:
            Dictionary with simulation results, or None on failure.
        """
        if not validate_simulation_inputs(dataset_id, task):
            return None

        hooks_meta = hooks.describe() if hooks else None

        start_time = time.time()
        run_result = self._client.create_run(
            name=name,
            dataset_id=dataset_id,
            context=context or {},
            hooks_meta=hooks_meta,
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
                    run_id, simulation_items, task, max_concurrency, max_turns, hooks  # type:ignore[arg-type]
                )
            )

            elapsed_time = time.time() - start_time
            logger.info("%s: Simulation completed in %.2f seconds", LOG_PREFIX, elapsed_time)
            self._client.post_run_status(run_id, "completed")  # type:ignore[arg-type]
            return result
        except Exception:
            logger.error("%s: Run simulation failed", LOG_PREFIX, exc_info=True)
            self._client.post_run_status(run_id, "failed")  # type:ignore[arg-type]
            return None

    async def _run_simulation_async(
        self,
        run_id: str,
        simulation_items: list[SimulationItem],
        task: BaseTask,
        max_concurrency: int,
        max_turns: int,
        hooks: Optional[SimulationHooks],
    ) -> dict[str, Any]:
        """Orchestrate concurrent simulation execution.

        Each simulation item is dispatched to a thread via ``run_in_executor``.
        Inside each thread, ``run_async_safely`` creates a **new** event loop
        so that async user tasks (``BaseTask.run``) work correctly without
        nesting into the orchestrator's loop.  This two-level design lets us
        honour ``max_concurrency`` while supporting both sync and async tasks
        transparently.

        Executes ``before_all`` first (if configured). If it fails the entire
        run is aborted.  Individual scenarios run concurrently via a thread
        pool; each thread gets its own event loop so sync and async tasks work
        without loop nesting.

        Args:
            run_id: The simulation run identifier.
            simulation_items: List of simulation items to process.
            task: The BaseTask instance to execute (sync or async).
            max_concurrency: Maximum concurrent executions.
            max_turns: Maximum conversation turns per item.
            hooks: Optional lifecycle hooks.

        Returns:
            Dictionary with simulation results.
        """
        results: dict[str, Any] = {
            "success": True,
            "completed": [],
            "failed": [],
            "total_items": len(simulation_items),
        }

        # --- before_all ---
        shared_context: Optional[dict[str, Any]] = None
        if hooks and hooks.before_all is not None:
            try:
                shared_context = await run_before_all(hooks)
            except Exception as exc:
                logger.error(
                    "%s: before_all hook failed: %s — aborting run",
                    LOG_PREFIX,
                    exc,
                    exc_info=True,
                )
                # Mark every item as prescript_failed and abort
                for item in simulation_items:
                    self._client.report_failure(
                        run_id=run_id,
                        run_item_id=item.run_item_id,
                        error=f"before_all hook failed: {exc}",
                        status="prescript_failed",
                    )
                results["success"] = False
                results["failed"] = [
                    {
                        "run_item_id": item.run_item_id,
                        "success": False,
                        "error": f"before_all hook failed: {exc}",
                    }
                    for item in simulation_items
                ]
                return results

        processed_count = 0
        lock = asyncio.Lock()
        loop = asyncio.get_running_loop()

        def run_item_in_thread(run_item: SimulationItem) -> dict[str, Any]:
            """Run a single simulation item in a dedicated thread/event-loop.

            Args:
                run_item: The simulation item to run.

            Returns: Dictionary with simulation result.
            """
            return run_async_safely(
                self._execute_conversation(run_id, run_item, task, max_turns, hooks, shared_context)
            )

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

        # --- after_all ---
        await run_after_all(hooks, results, shared_context)

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
        hooks: Optional[SimulationHooks],
        shared_context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a multi-turn conversation for a single simulation item.

        Runs the ``before`` hook before starting the conversation loop and
        the ``after`` hook when the conversation ends (success or failure).

        Args:
            run_id: The simulation run identifier.
            run_item: The simulation item to process.
            task: The BaseTask instance to execute (sync or async).
            max_turns: Safety limit on the number of conversation turns.
            hooks: Optional lifecycle hooks.
            shared_context: Context dict returned by the ``before_all`` hook.

        Returns:
            Dictionary with execution result including success status.
        """
        run_item_id = run_item.run_item_id
        dataset_item_id = run_item.dataset_item_id
        message = run_item.message
        turn_id = run_item.turn_id
        raw_files: list[FileData] = run_item.files
        session_id: Optional[str] = None

        # --- before ---
        setup_context: Optional[dict[str, Any]] = None
        has_before_hooks = hooks and hooks.before and dataset_item_id in hooks.before
        if has_before_hooks:
            try:
                setup_context = await run_before(hooks, dataset_item_id, shared_context)
            except Exception as exc:
                error_msg = f"before hook failed: {exc}"
                logger.error(
                    "%s: %s for run_item_id=%s",
                    LOG_PREFIX,
                    error_msg,
                    run_item_id,
                    exc_info=True,
                )
                self._client.report_failure(
                    run_id=run_id,
                    run_item_id=run_item_id,
                    error=error_msg,
                    status="prescript_failed",
                )
                item_result = {
                    "run_item_id": run_item_id,
                    "success": False,
                    "error": error_msg,
                }
                await run_after(hooks, dataset_item_id, item_result, shared_context)
                return item_result
        else:
            setup_context = shared_context

        for turn_number in range(1, max_turns + 1):
            try:
                with SpanWrapper(SPAN_NAME, module_name=LOG_PREFIX) as span:
                    trace_id = ""
                    otel_span = span.get_current_span()
                    if otel_span:
                        span_context = otel_span.get_span_context()
                        trace_id = format_trace_id(span_context.trace_id)

                    response_message, task_session_id = await execute_task(
                        task,
                        message,
                        session_id,
                        raw_files=raw_files,
                        setup_context=setup_context,
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
                    item_result = {
                        "run_item_id": run_item_id,
                        "success": False,
                        "error": error_msg,
                        "turn_id": turn_id,
                    }
                    await run_after(hooks, dataset_item_id, item_result, shared_context)
                    return item_result

                if response.decision == ConversationStatus.STOP:
                    logger.info(
                        "%s: Completed run_item_id=%s reason=%s",
                        LOG_PREFIX,
                        run_item_id,
                        response.reason,
                    )
                    item_result = {
                        "run_item_id": run_item_id,
                        "success": True,
                        "final_turn_id": turn_id,
                    }
                    await run_after(hooks, dataset_item_id, item_result, shared_context)
                    return item_result

                message = response.next_user_message  # type:ignore[assignment]
                turn_id = response.next_turn_id  # type:ignore[assignment]
                raw_files = response.next_files

            except Exception as exc:
                error_msg = str(exc)
                logger.error(
                    "%s: Task failed run_item_id=%s, turn_id=%s: %s",
                    LOG_PREFIX,
                    run_item_id,
                    turn_id,
                    error_msg,
                )
                self._client.report_failure(run_id=run_id, run_item_id=run_item_id, error=error_msg)
                item_result = {
                    "run_item_id": run_item_id,
                    "success": False,
                    "error": error_msg,
                    "turn_id": turn_id,
                }
                await run_after(hooks, dataset_item_id, item_result, shared_context)
                return item_result

        error_msg = f"Exceeded maximum turns ({max_turns}) for run_item_id={run_item_id}"
        logger.error("%s: %s", LOG_PREFIX, error_msg)
        self._client.report_failure(run_id=run_id, run_item_id=run_item_id, error=error_msg)
        item_result = {
            "run_item_id": run_item_id,
            "success": False,
            "error": error_msg,
            "turn_id": turn_id,
        }
        await run_after(hooks, dataset_item_id, item_result, shared_context)
        return item_result
