import logging
from typing import Any

from netra.config import Config
from netra.usage.client import _UsageHttpClient
from netra.usage.models import SessionUsageData, TenantUsageData

logger = logging.getLogger(__name__)


class Usage:
    """Public entry-point exposed as Netra.usage"""

    def __init__(self, cfg: Config) -> None:
        self._config = cfg
        self._client = _UsageHttpClient(cfg)

    def get_session_usage(self, session_id: str) -> SessionUsageData | Any:
        if not session_id:
            logger.error("netra.usage: session_id is required to fetch session usage")
            return None
        result = self._client.get_session_usage(session_id)
        session_id = result.get("session_id", "")
        if not session_id:
            return None
        token_count = result.get("tokenCount", 0)
        request_count = result.get("requests", 0)
        return SessionUsageData(session_id=session_id, token_count=token_count, request_count=request_count)

    def get_tenant_usage(self, tenant_id: str) -> TenantUsageData | Any:
        if not tenant_id:
            logger.error("netra.usage: tenant_id is required to fetch tenant usage")
            return None
        result = self._client.get_tenant_usage(tenant_id)
        tenant_id = result.get("tenant_id", "")
        if not tenant_id:
            return None
        token_count = int(result.get("tokenCount", 0))
        request_count = int(result.get("requests", 0))
        session_count = int(result.get("sessions", 0))
        return TenantUsageData(
            tenant_id=tenant_id, token_count=token_count, request_count=request_count, session_count=session_count
        )
