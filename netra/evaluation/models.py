"""Data models for the evaluation module."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateDatasetResponse(BaseModel):  # type:ignore[misc]
    """Response from creating a dataset."""

    project_id: str
    organization_id: str
    name: str
    tags: Optional[list[str]] = Field(default_factory=list)
    created_by: str
    updated_by: str
    updated_at: str
    id: str
    created_at: str
    deleted_at: Optional[str] = None


class AddDatasetItemResponse(BaseModel):  # type:ignore[misc]
    """Response from adding a dataset item."""

    dataset_id: str
    project_id: str
    organization_id: str
    source: str
    input: Any
    expected_output: Optional[Any] = None
    is_active: bool
    tags: Optional[list[str]] = Field(default_factory=list)
    created_by: str
    updated_by: str
    updated_at: str
    source_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    id: str
    created_at: str
    deleted_at: Optional[str] = None


class DatasetRecord(BaseModel):  # type:ignore[misc]
    """A single record fetched from a remote dataset."""

    id: str
    input: Any
    dataset_id: str
    expected_output: Optional[Any] = None


class GetDatasetItemsResponse(BaseModel):  # type:ignore[misc]
    """Response from fetching dataset items."""

    items: list[DatasetRecord]


class DatasetItem(BaseModel):  # type:ignore[misc]
    """A single dataset item provided by the user."""

    input: Any
    expected_output: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None


class ScoreType(str, Enum):
    """Supported evaluator score types."""

    BOOLEAN = "boolean"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class EvaluatorConfig(BaseModel):  # type:ignore[misc]
    """Configuration for a single evaluator."""

    name: str
    label: str
    score_type: ScoreType = Field(alias="scoreType")

    model_config = {
        "populate_by_name": True,
    }


class EvaluatorContext(BaseModel):  # type:ignore[misc]
    """Context passed to an evaluator's evaluate() method."""

    input: Any
    task_output: Any
    expected_output: Any = None
    metadata: Optional[dict[str, Any]] = None


class EvaluatorOutput(BaseModel):  # type:ignore[misc]
    """Result returned from an evaluator's evaluate() method."""

    evaluator_name: str
    result: Any
    is_passed: bool
    reason: Optional[str] = None


class Dataset(BaseModel):  # type:ignore[misc]
    """Container for dataset items used by run_test_suite."""

    items: list[DatasetItem] | list[DatasetRecord]


@dataclass(slots=True)
class ItemContext:
    """Context for a single dataset item during test suite execution."""

    index: int
    item_input: Any
    expected_output: Any = None
    metadata: Optional[dict[str, Any]] = None
    dataset_item_id: Optional[str] = None
    trace_id: str = ""
    span_id: str = ""
    session_id: Optional[str] = None
    test_run_item_id: Optional[str] = None
    task_output: Any = None
    status: str = "pending"


class LocalDataset(BaseModel):  # type:ignore[misc]
    """Local dataset class for running test suite locally."""

    items: list[DatasetItem]


class TurnType(str, Enum):
    """Turn type for a dataset."""

    SINGLE = "single"
    MULTI = "multi"


@dataclass(slots=True)
class ItemProcessingResult:
    """Result of processing a single dataset item."""

    item_entry: dict[str, Any]
    should_run_evaluators: bool
    ctx: ItemContext
    status: str
