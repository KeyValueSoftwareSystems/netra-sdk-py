from pydantic import BaseModel


class SessionUsageData(BaseModel):  # type:ignore[misc]
    session_id: str
    token_count: int
    request_count: int


class TenantUsageData(BaseModel):  # type:ignore[misc]
    tenant_id: str
    token_count: int
    request_count: int
    session_count: int
