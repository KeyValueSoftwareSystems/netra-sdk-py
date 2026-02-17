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
    AUDIO_DURATION = "Audio Duration"
    CHARACTER_COUNT = "Character Count"


class Aggregation(str, Enum):
    """Aggregation method for metrics."""

    AVERAGE = "Average"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    MEDIAN = "Median (p50)"
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
    METADATA = "metadata"


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


class SessionFilterField(str, Enum):
    """Filter fields for session stats queries."""

    TENANT_ID = "tenant_id"
    ENVIRONMENT = "environment"
    SERVICE = "service"


class SessionFilterOperator(str, Enum):
    """Filter operators for session stats queries."""

    ANY_OF = "any_of"


class SessionFilterType(str, Enum):
    """Data types for session stats filter conditions."""

    ARRAY = "arrayOptions"


class SortField(str, Enum):
    """Sort fields for session stats queries."""

    SESSION_ID = "session_id"
    START_TIME = "start_time"
    TOTAL_REQUESTS = "totalRequests"
    TOTAL_COST = "totalCost"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class SessionFilter(BaseModel):  # type:ignore[misc]
    """
    Filter condition for session stats queries.

    Attributes:
        field: Filter field from SessionFilterField enum.
        operator: Filter operator from SessionFilterOperator enum.
        type: Data type from SessionFilterType enum.
        value: The list of values to filter by.
    """

    field: SessionFilterField
    operator: SessionFilterOperator
    type: SessionFilterType
    value: List[str]


class SessionStatsData(BaseModel):  # type:ignore[misc]
    """Individual session statistics data."""

    session_id: str
    start_time: str
    total_requests: int
    total_cost: float
    session_duration: str


class SessionStatsResult(BaseModel):  # type:ignore[misc]
    """Response wrapper for session stats queries."""

    data: List[Dict[str, Any]]
    has_next_page: bool
    next_cursor: Optional[str] = None


class SessionFilterConfig(BaseModel):  # type:ignore[misc]
    start_time: str
    end_time: str
    filters: Optional[List[SessionFilter]] = None
