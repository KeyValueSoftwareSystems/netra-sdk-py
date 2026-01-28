from dataclasses import dataclass
from enum import Enum


class ConversationStatus(Enum):
    """Status indicating whether to continue or stop the conversation."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass
class SimulationItem:
    """Represents a single item in a simulation run."""

    item_id: str
    message: str
    turn: int


@dataclass
class ConversationResponse:
    """Response from the conversation trigger API."""

    message: str
    turn: int
    status: ConversationStatus


@dataclass
class TaskResult:
    """Result returned from the user's task function."""

    message: str
    session_id: str
