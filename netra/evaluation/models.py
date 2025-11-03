from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


@dataclass
class DatasetItem:
    id: str
    input: Any
    dataset_id: str


@dataclass
class Dataset:
    dataset_id: str
    items: List[DatasetItem]


@dataclass
class Run:
    id: str
    dataset_id: str
    name: Optional[str]
    test_entries: List[DatasetItem]


class EntryStatus(Enum):
    AGENT_TRIGGERED = "agent_triggered"
    AGENT_COMPLETED = "agent_completed"
    FAILED = "failed"


class RunStatus(Enum):
    COMPLETED = "completed"
