"""
Base task class for Netra simulation framework.

This module provides the abstract base class that all custom tasks
should inherit from when implementing simulation tasks for run_simulation().
"""

from abc import ABC, abstractmethod
from typing import Awaitable, Optional

from netra.simulation.models import TaskResult


class BaseTask(ABC):
    """
    Abstract base class for all simulation tasks.

    Subclasses must:
        - Implement run(): Executes the task logic and returns a TaskResult.

    The run method receives a message and optional session_id, and must return
    a TaskResult containing the response message and session_id.

    Example:
        class MyTask(BaseTask):
            def run(self, message: str, session_id: Optional[str] = None) -> TaskResult:
                # Call your LLM or agent here
                response = my_agent.chat(message, session_id=session_id)
                return TaskResult(
                    message=response.text,
                    session_id=response.session_id or session_id or "default",
                )

        # Usage:
        result = Netra.simulation.run_simulation(
            dataset_id="my-dataset-id",
            task=MyTask(),
        )

    Async Example:
        class MyAsyncTask(BaseTask):
            async def run(self, message: str, session_id: Optional[str] = None) -> TaskResult:
                # Call your async LLM or agent here
                response = await my_async_agent.chat(message, session_id=session_id)
                return TaskResult(
                    message=response.text,
                    session_id=response.session_id or session_id or "default",
                )
    """

    @abstractmethod
    def run(self, message: str, session_id: Optional[str] = None) -> TaskResult | Awaitable[TaskResult]:
        """
        Execute the task logic.

        This method can be sync or async. If async, the framework will
        await the coroutine automatically.

        Args:
            message: The input message from the simulation.
            session_id: Optional session identifier for conversation continuity.
                        Will be None for the first turn of a conversation.

        Returns:
            TaskResult: The task result containing:
                - message (str): The response message from the task.
                - session_id (str): The session identifier for conversation continuity.
        """
