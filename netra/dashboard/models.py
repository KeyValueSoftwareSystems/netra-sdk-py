from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class Scope(str, Enum):
    """Scope of data to query."""

    SPANS = "Spans"
    TRACES = "Traces"


class ChartType(str, Enum):
    """Type of chart visualization."""

    LINE_TIME_SERIES = "Line Time Series"
    BAR_TIME_SERIES = "Bar Time Series"
    HORIZONTAL_BAR = "Horizontal Bar"
    VERTICAL_BAR = "Vertical Bar"
    PIE = "Pie"
    NUMBER = "Number"


class Measure(str, Enum):
    """Metric to measure."""

    LATENCY = "Latency"
    ERROR_RATE = "Error Rate"
    PII_COUNT = "PII Count"
    REQUEST_COUNT = "Request Count"
    TOTAL_COST = "Total Cost"
    VIOLATIONS = "Violations"
    TOTAL_TOKENS = "Total Tokens"


class Aggregation(str, Enum):
    """Aggregation method for metrics."""

    AVERAGE = "Average"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    MEDIAN = "Median"
    PERCENTAGE = "Percentage"
    TOTAL_COUNT = "Total Count"


class GroupBy(str, Enum):
    """Time grouping granularity."""

    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"


class DimensionField(str, Enum):
    """Dimension fields for grouping results."""

    ENVIRONMENT = "environment"
    USER_ID = "user_id"
    SERVICE = "service"
    MODEL_NAME = "model_name"


class Operator(str, Enum):
    """Filter operators for query conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL_TO = "greater_equal_to"
    LESS_EQUAL_TO = "less_equal_to"
    ANY_OF = "any_of"
    NONE_OF = "none_of"


class Type(str, Enum):
    """Data types for filter conditions."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY_OPTIONS = "arrayOptions"
    OBJECT = "object"


class FilterField(str, Enum):
    """
    Filter fields for dashboard queries.

    Note:
        - Use MODEL_NAME for Spans scope
        - Use MODELS for Traces scope
        - For metadata filters, use metadata_field() helper function
    """

    TOTAL_COST = "total_cost"
    SERVICE = "service"
    TENANT_ID = "tenant_id"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    ENVIRONMENT = "environment"
    LATENCY = "latency"
    MODEL_NAME = "model_name"
    MODELS = "models"


def metadata_field(key: str) -> str:
    """
    Create a metadata filter field.

    Args:
        key: The metadata key to filter on.

    Returns:
        The formatted metadata field string (e.g., "metadata['key']").

    Example:
        >>> metadata_field("customer_tier")
        "metadata['customer_tier']"
    """
    return f"metadata['{key}']"


class Filter(BaseModel):  # type:ignore[misc]
    """
    Filter condition for dashboard queries.

    Attributes:
        field: Filter field - use FilterField enum or metadata_field() helper.
        operator: Filter operator from FilterOperator enum.
        type: Data type from FilterType enum.
        value: The value to filter by.
        key: Required for FilterType.OBJECT filters.
    """

    field: FilterField
    operator: Operator
    type: Type
    value: Any
    key: Optional[str] = None


class Metrics(BaseModel):  # type:ignore[misc]
    """
    Metrics configuration for dashboard queries.

    Attributes:
        measure: The metric to measure (e.g., Metric.LATENCY).
        aggregation: The aggregation method (e.g., Aggregation.AVERAGE).
    """

    measure: Measure
    aggregation: Aggregation


class Dimension(BaseModel):  # type:ignore[misc]
    """
    Dimension configuration for dashboard queries.

    Attributes:
        field: The dimension field to group results by.
    """

    field: DimensionField


class FilterConfig(BaseModel):  # type:ignore[misc]
    """
    Filter configuration for dashboard queries.

    Attributes:
        start_time: Start time in ISO 8601 UTC format (YYYY-MM-DDTHH:mm:ss.SSSZ).
        end_time: End time in ISO 8601 UTC format (YYYY-MM-DDTHH:mm:ss.SSSZ).
        group_by: Time grouping granularity.
        filters: Optional list of filter conditions.
    """

    start_time: str
    end_time: str
    group_by: GroupBy
    filters: Optional[List[Filter]] = None


class TimeRange(BaseModel):  # type:ignore[misc]
    """Time range information in the response."""

    start_time: str
    end_time: str


class TimeSeriesDataPoint(BaseModel):  # type:ignore[misc]
    """Data point for time series without dimension."""

    date: str
    value: float


class Value(BaseModel):  # type:ignore[misc]
    """Value for a specific dimension."""

    dimension: str
    value: float


class TimeSeriesWithDimension(BaseModel):  # type:ignore[misc]
    """Time series data point with dimension values."""

    date: str
    values: List[Value]


class TimeSeriesResponse(BaseModel):  # type:ignore[misc]
    """Response for time series with dimension."""

    time_series: List[TimeSeriesWithDimension]
    dimensions: List[str]


class CategoricalDataPoint(BaseModel):  # type:ignore[misc]
    """Data point for categorical charts (Pie/Bar)."""

    dimension: str
    value: float


class NumberResponse(BaseModel):  # type:ignore[misc]
    """Response for number chart."""

    value: float


Data = Union[
    List[TimeSeriesDataPoint],
    TimeSeriesResponse,
    List[CategoricalDataPoint],
    NumberResponse,
    Dict[str, Any],
]


class QueryResponse(BaseModel):  # type:ignore[misc]
    """Response wrapper for dashboard queries."""

    time_range: TimeRange
    data: Data
