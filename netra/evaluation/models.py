from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatasetItem(BaseModel):  # type:ignore[misc]
    id: str
    input: str
    dataset_id: str
    expected_output: Optional[Any] = None


class DatasetEntry(BaseModel):  # type:ignore[misc]
    input: str
    expected_output: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    policy_ids: Optional[List[str]] = None


class Dataset(BaseModel):  # type:ignore[misc]
    dataset_id: str
    items: List[DatasetItem]


class Run(BaseModel):  # type:ignore[misc]
    id: str
    dataset_id: str
    name: Optional[str]
    test_entries: List[DatasetItem]


class EvaluationScore(BaseModel):  # type:ignore[misc]
    metric_type: str
    score: float


class EntryStatus(Enum):
    AGENT_TRIGGERED = "agent_triggered"
    AGENT_COMPLETED = "agent_completed"
    FAILED = "failed"


class RunStatus(Enum):
    COMPLETED = "completed"
