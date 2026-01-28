import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from netra.config import Config
from netra.simulation.client import SimulationHttpClient
from netra.simulation.models import ConversationStatus, SimulationItem
from netra.simulation.utils import execute_task, run_async_safely, validate_simulation_inputs

logger = logging.getLogger(__name__)


class Simulation:
    """Public API for running multi-turn conversation simulations."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = SimulationHttpClient(config)

    def run_simulation(
        self,
        dataset_id: str,
        task: Callable[[str, Optional[str]], Any],
        context: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Run a multi-turn conversation simulation.

        Args:
            dataset_id: Identifier of the dataset to simulate.
            task: Callable that receives (message, session_id) and returns TaskResult.
                  Can be sync or async.
            context: Optional context data for the simulation.
            max_concurrency: Maximum parallel executions (default: 5).

        Returns:
            Dictionary with simulation results, or None on failure.
        """
        if not validate_simulation_inputs(dataset_id, task):
            return None

        items = self._client.create_run(dataset_id=dataset_id, context=context or {})
        if not items:
            logger.error("netra.simulation: No items returned from create_run")
            return None

        logger.info("netra.simulation: Starting simulation with %d items", len(items))
        return run_async_safely(self._run_simulation_async(items, task, max_concurrency))

    async def _run_simulation_async(
        self,
        items: List[SimulationItem],
        task: Callable[[str, Optional[str]], Any],
        max_concurrency: int,
    ) -> Dict[str, Any]:
        """
        Async implementation of run_simulation with semaphore-based concurrency.
        
        items: List of simulation items to process.
        task: The task function to execute (sync or async).
        max_concurrency: Maximum concurrent executions.

        Returns:
            Dictionary with simulation results.
        """
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        results: Dict[str, Any] = {
            "success": True,
            "completed": [],
            "failed": [],
            "total_items": len(items),
        }
        processed_count = 0
        lock = asyncio.Lock()

        async def process_item(item: SimulationItem) -> None:
            nonlocal processed_count
            async with semaphore:
                result = await self._execute_conversation(item, task)
                async with lock:
                    target = results["completed"] if result["success"] else results["failed"]
                    target.append(result)
                    processed_count += 1
                    logger.info(
                        "netra.simulation: %d/%d processed (item_id=%s)",
                        processed_count,
                        len(items),
                        item.item_id,
                    )

        await asyncio.gather(*[process_item(item) for item in items])
        logger.info(
            "netra.simulation: Completed=%d, Failed=%d",
            len(results["completed"]),
            len(results["failed"]),
        )
        return results

    async def _execute_conversation(
        self,
        item: SimulationItem,
        task: Callable[[str, Optional[str]], Any],
    ) -> Dict[str, Any]:
        """
        Execute a multi-turn conversation for a single simulation item.
        
        item: The simulation item to process.
        task: The task function to execute (sync or async).

        Returns:
            Dictionary with execution result.
        """
        item_id = item.item_id
        message = item.message
        turn = item.turn
        session_id: Optional[str] = None

        logger.info("netra.simulation: Starting item_id=%s", item_id)

        while True:
            try:
                response_message, session_id = await execute_task(task, message, session_id)
            except Exception as exc:
                error_msg = str(exc)
                logger.error("netra.simulation: Task failed item_id=%s, turn=%d: %s", item_id, turn, error_msg)
                self._client.report_failure(item_id=item_id, turn=turn, error=error_msg)
                return {"item_id": item_id, "success": False, "error": error_msg, "turn": turn}

            response = self._client.trigger_conversation(
                item_id=item_id,
                message=response_message,
                turn=turn,
                session_id=session_id or "",
            )

            if response is None:
                error_msg = "Failed to get conversation response"
                logger.error("netra.simulation: %s for item_id=%s", error_msg, item_id)
                return {"item_id": item_id, "success": False, "error": error_msg, "turn": turn}

            if response.status == ConversationStatus.STOP:
                logger.info("netra.simulation: Completed item_id=%s at turn=%d", item_id, turn)
                return {"item_id": item_id, "success": True, "final_turn": turn}

            message = response.message
            turn = response.turn
            logger.debug("netra.simulation: Continuing item_id=%s, turn=%d", item_id, turn)