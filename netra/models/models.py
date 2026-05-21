from typing import List, Optional

from pydantic import BaseModel


class ModelPrice(BaseModel):  # type:ignore[misc]
    usage_type: str
    min_units: float
    max_units: Optional[float] = None
    price: Optional[float] = None
    unit_value: float


class Model(BaseModel):  # type:ignore[misc]
    id: str
    name: str
    match_pattern: str
    project_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None
    prices: List[ModelPrice] = []
