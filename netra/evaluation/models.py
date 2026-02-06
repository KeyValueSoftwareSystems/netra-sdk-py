import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateDatasetResponse(BaseModel):  # type:ignore[misc]
    project_id: str
    organization_id: str
    name: str
    tags: Optional[List[str]] = []
    created_by: str
    updated_by: str
    updated_at: str
    id: str
    created_at: str
    deleted_at: Optional[str] = None


class AddDatasetItemResponse(BaseModel):  # type:ignore[misc]
    dataset_id: str
    project_id: str
    organization_id: str
    source: str
    input: Any
    expected_output: Optional[Any] = None
    is_active: bool
    tags: Optional[List[str]] = []
    created_by: str
    updated_by: str
    updated_at: str
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    id: str
    created_at: str
    deleted_at: Optional[str] = None


class DatasetRecord(BaseModel):  # type:ignore[misc]
    id: str
    input: Any
    dataset_id: str
    expected_output: Optional[Any] = None


class GetDatasetItemsResponse(BaseModel):  # type:ignore[misc]
    items: List[DatasetRecord]


class DatasetItem(BaseModel):  # type:ignore[misc]
    input: Any
    expected_output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class ScoreType(str, Enum):
    BOOLEAN = "boolean"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class EvaluatorConfig(BaseModel):  # type:ignore[misc]
    name: str
    label: str
    score_type: ScoreType = Field(alias="scoreType")

    model_config = {
        "populate_by_name": True,
    }


class EvaluatorContext(BaseModel):  # type:ignore[misc]
    input: Any
    task_output: Any
    expected_output: Any = None
    metadata: Optional[Dict[str, Any]] = None


class EvaluatorOutput(BaseModel):  # type:ignore[misc]
    evaluator_name: str
    result: Any
    is_passed: bool
    reason: Optional[str] = None


class Dataset(BaseModel):  # type:ignore[misc]
    items: List[DatasetItem] | List[DatasetRecord]


@dataclass
class ItemContext:
    """Context for a single dataset item during test suite execution."""

    index: int
    item_input: Any
    expected_output: Any = None
    metadata: Optional[Dict[str, Any]] = None
    dataset_item_id: Optional[str] = None
    trace_id: str = ""
    span_id: str = ""
    session_id: Optional[str] = None
    test_run_item_id: Optional[str] = None
    task_output: Any = None
    status: str = "pending"


@dataclass
class RunContext:
    """Shared context for a test suite run."""

    run_id: str
    run_name: str
    evaluators: Optional[List[Any]] = None
    poller: Optional[Any] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    bg_eval_tasks: List[asyncio.Task[None]] = field(default_factory=list)


class LocalDataset(BaseModel):  # type:ignore[misc]
    """Local dataset class for running test suite locally."""

    items: List[DatasetItem]


class TurnType(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass
class ItemProcessingResult:
    """Result of processing a single dataset item."""

    item_entry: Dict[str, Any]
    should_run_evaluators: bool
    ctx: ItemContext
    status: str
