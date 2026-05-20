"""Data models for the simulation module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConversationStatus(Enum):
    """Status indicating whether to continue or stop the conversation."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(slots=True, frozen=True)
class FileData:
    """Raw file metadata received from the backend.

    Attributes:
        file_name: Name of the file.
        content_type: MIME type of the file content.
        description: Optional description of the file.
        download_url: Pre-signed URL to download the file.
    """

    file_name: str
    content_type: str
    description: Optional[str]
    download_url: str


@dataclass(slots=True, frozen=True)
class ProcessedFile:
    """File after download and base64 encoding, delivered to the user task.

    Attributes:
        file_name: Name of the file.
        content_type: MIME type of the file content.
        description: Optional description of the file.
        data: Base64-encoded file content.
    """

    file_name: str
    content_type: str
    description: Optional[str]
    data: str


@dataclass(slots=True, frozen=True)
class SimulationItem:
    """Represents a single item in a simulation run.

    Attributes:
        run_item_id: Unique identifier for the run item.
        message: The user message content.
        turn_id: Identifier for the conversation turn.
        files: File metadata attached to this item.
    """

    run_item_id: str
    message: str
    turn_id: str
    files: list[FileData] = field(default_factory=list)


@dataclass(slots=True)
class ConversationResponse:
    """Response from the conversation trigger API.

    Attributes:
        decision: Whether to continue or stop the conversation.
        reason: Optional reason for stopping the conversation.
        next_turn_id: Identifier for the next turn if continuing.
        next_user_message: The next user message if continuing.
        next_files: File metadata for the next turn if continuing.
    """

    decision: ConversationStatus
    reason: Optional[str] = None
    next_turn_id: Optional[str] = None
    next_user_message: Optional[str] = None
    next_files: list[FileData] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class TaskResult:
    """Result returned from the user's task function.

    Attributes:
        message: The response message from the task.
        session_id: The session identifier for conversation continuity.
    """

    message: str
    session_id: str
