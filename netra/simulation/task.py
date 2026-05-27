"""
Base task class for Netra simulation framework.

This module provides the abstract base class that all custom tasks
should inherit from when implementing simulation tasks for run_simulation().
"""

from abc import ABC, abstractmethod
from typing import Awaitable, Optional

from netra.simulation.models import ProcessedFile, TaskResult


class BaseTask(ABC):
    """
    Abstract base class for all simulation tasks.

    Subclasses must:
        - Implement run(): Executes the task logic and returns a TaskResult.

    The framework always passes ``message``, ``session_id``, and ``files``
    to ``run()``.  Tasks that don't need file attachments can simply ignore
    the ``files`` parameter.

    Example:
        class MyTask(BaseTask):
            def run(
                self,
                message: str,
                session_id: Optional[str] = None,
                files: Optional[list[ProcessedFile]] = None,
            ) -> TaskResult:
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
            async def run(
                self,
                message: str,
                session_id: Optional[str] = None,
                files: Optional[list[ProcessedFile]] = None,
            ) -> TaskResult:
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
        files: Optional[list[ProcessedFile]] = None,
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
            files: Optional list of base64-encoded file attachments from the
                   dataset item.  Will be None when no files are attached.

        Returns:
            TaskResult: The task result containing:
                - message (str): The response message from the task.
                - session_id (str): The session identifier for conversation continuity.
        """
