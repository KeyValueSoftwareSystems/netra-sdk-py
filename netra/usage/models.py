from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SessionUsageData(BaseModel):  # type:ignore[misc]
    session_id: str
    token_count: int
    request_count: int
    total_cost: float


class TenantUsageData(BaseModel):  # type:ignore[misc]
    tenant_id: str
    organisation_id: str
    token_count: int
    request_count: int
    session_count: int
    total_cost: float


class TraceSummary(BaseModel):  # type:ignore[misc]
    id: str
    name: str
    kind: str
    latency_ms: int
    start_time: str
    end_time: str
    cursor: str
    organisation_id: str
    project_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: Optional[str] = None
    models: Optional[List[str]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens_cost: Optional[float] = None
    completion_tokens_cost: Optional[float] = None
    cached_tokens_cost: Optional[float] = None
    total_cost: Optional[float] = None
    has_pii: Optional[bool] = None
    pii_entities: Optional[List[Any]] = None
    has_violation: Optional[bool] = None
    violations: Optional[List[Any]] = None
    has_error: Optional[bool] = None
    service: Optional[str] = None


class TracesPage(BaseModel):  # type:ignore[misc]
    traces: List[TraceSummary]
    has_next_page: bool
    next_cursor: Optional[str] = None


class TraceSpan(BaseModel):  # type:ignore[misc]
    id: str
    trace_id: str
    name: str
    kind: str
    parent_span_id: Optional[str] = None
    latency_ms: Optional[int] = None
    start_time_ms: Optional[str] = None
    end_time_ms: Optional[str] = None
    organisation_id: str
    project_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: Optional[str] = None
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    prompt_tokens_cost: Optional[float] = None
    completion_tokens_cost: Optional[float] = None
    cached_tokens_cost: Optional[float] = None
    total_cost: Optional[float] = None
    has_pii: Optional[bool] = None
    pii_entities: Optional[List[Any]] = None
    pii_actions: Optional[Dict[str, Any]] = None
    has_violation: Optional[bool] = None
    violations: Optional[List[Any]] = None
    violation_actions: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    attributes: Optional[str] = None
    events: Optional[str] = None
    links: Optional[str] = None
    resources: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    has_error: Optional[bool] = None
    created_at: str
    cursor: str


class SpansPage(BaseModel):  # type:ignore[misc]
    spans: List[TraceSpan]
    has_next_page: bool
    next_cursor: Optional[str] = None
