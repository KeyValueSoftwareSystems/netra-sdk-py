"""Data models for the simulation module."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConversationStatus(Enum):
    """Status indicating whether to continue or stop the conversation."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(slots=True, frozen=True)
class SimulationItem:
    """Represents a single item in a simulation run.

    Attributes:
        run_item_id: Unique identifier for the run item.
        message: The user message content.
        turn_id: Identifier for the conversation turn.
    """

    run_item_id: str
    message: str
    turn_id: str


@dataclass(slots=True)
class ConversationResponse:
    """Response from the conversation trigger API.

    Attributes:
        decision: The decision to continue or stop the conversation.
        reason: Optional reason for stopping the conversation.
        next_turn_id: Identifier for the next turn if continuing.
        next_user_message: The next user message if continuing.
        next_run_item_id: Identifier for the next run item if continuing.
    """

    decision: str
    reason: Optional[str] = None
    next_turn_id: Optional[str] = None
    next_user_message: Optional[str] = None
    next_run_item_id: Optional[str] = None


@dataclass(slots=True, frozen=True)
class TaskResult:
    """Result returned from the user's task function.

    Attributes:
        message: The response message from the task.
        session_id: The session identifier for conversation continuity.
    """

    message: str
    session_id: str
