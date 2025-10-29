from dataclasses import dataclass
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
