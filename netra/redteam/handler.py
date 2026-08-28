"""The user-supplied agent callback for ``run_red_team()``.

A handler is a plain function, called once per turn — no class to extend.

Example:
    def my_handler(prompt: str, session_id: str, turn_index: int) -> str:
        return my_agent.chat(prompt, session_id=session_id)

    Netra.red_team.run_red_team(config_id="...", handler=my_handler)

Async handlers work the same way. To override the session id, return
``{"message": "...", "session_id": "..."}`` instead of a plain string.
"""

import asyncio
from typing import Any, Awaitable, Callable, Union

RedTeamAgentResponse = Union[str, dict[str, Any]]
RedTeamAgentHandler = Callable[[str, str, int], Union[RedTeamAgentResponse, Awaitable[RedTeamAgentResponse]]]


async def execute_handler(
    handler: RedTeamAgentHandler,
    prompt: str,
    session_id: str,
    turn_index: int,
) -> tuple[str, str]:
    """Call the user's handler for one turn and normalize its return value.

    Returns:
        A tuple of ``(output_message, session_id)``.

    Raises:
        TypeError: If the return value isn't a string or a dict with a
            string ``"message"`` key.
    """
    result = handler(prompt, session_id, turn_index)
    if asyncio.iscoroutine(result):
        result = await result

    if isinstance(result, str):
        return result, session_id

    if isinstance(result, dict):
        message = result.get("message")
        if isinstance(message, str):
            override_session_id = result.get("session_id")
            return message, override_session_id if isinstance(override_session_id, str) else session_id

    raise TypeError(f"red_team handler must return str | {{'message': str, ...}}, got {type(result).__name__}")
