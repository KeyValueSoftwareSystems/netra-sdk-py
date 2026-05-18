"""
Base task class for Netra simulation framework.

This module provides the abstract base class that all custom tasks
should inherit from when implementing simulation tasks for run_simulation().
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Optional

from netra.simulation.models import TaskResult


class BaseTask(ABC):
    """
    Abstract base class for all simulation tasks.

    Subclasses must:
        - Implement run(): Executes the task logic and returns a TaskResult.

    The base contract requires ``message`` and ``session_id``.  Subclasses that
    need file attachments can add a ``files`` keyword argument — the framework
    detects it via introspection and will pass downloaded files automatically.

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

    Example with file uploads:
        class MyFileTask(BaseTask):
            def run(
                self,
                message: str,
                session_id: Optional[str] = None,
                files: Optional[list[ProcessedFile]] = None,
            ) -> TaskResult:
                # Access base64-encoded file data
                if files:
                    for f in files:
                        print(f.file_name, f.content_type, len(f.data))
                response = my_agent.chat(message, session_id=session_id, files=files)
                return TaskResult(
                    message=response.text,
                    session_id=response.session_id or session_id or "default",
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
    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TaskResult | Awaitable[TaskResult]:
        """
        Execute the task logic.

        This method can be sync or async. If async, the framework will
        await the coroutine automatically.

        The base signature requires only ``message`` and ``session_id``.
        Subclasses that handle file attachments should declare an additional
        ``files: Optional[list[ProcessedFile]] = None`` parameter — the
        framework will supply it automatically when the dataset item includes
        file attachments.

        Args:
            message: The input message from the simulation.
            session_id: Optional session identifier for conversation continuity.
                        Will be None for the first turn of a conversation.
            **kwargs: Reserved for forward-compatible extensions (e.g. ``files``).

        Returns:
            TaskResult: The task result containing:
                - message (str): The response message from the task.
                - session_id (str): The session identifier for conversation continuity.
        """
