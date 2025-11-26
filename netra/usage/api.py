import logging
from typing import Any, Optional

from netra.config import Config
from netra.usage.client import UsageHttpClient
from netra.usage.models import SessionUsageData, TenantUsageData

logger = logging.getLogger(__name__)


class Usage:
    """Public entry-point exposed as Netra.usage"""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the usage client.

        Args:
            cfg: Configuration object with usage settings
        """
        self._config = cfg
        self._client = UsageHttpClient(cfg)

    def get_session_usage(
        self,
        session_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> SessionUsageData | Any:
        """
        Get session usage data.

        Args:
            session_id: Session identifier
            start_time: Start time for the usage data (in ISO 8601 UTC format)
            end_time: End time for the usage data (in ISO 8601 UTC format)

        Returns:
            SessionUsageData: Session usage data
        """
        if not session_id:
            logger.error("netra.usage: session_id is required to fetch session usage")
            return None
        result = self._client.get_session_usage(session_id, start_time=start_time, end_time=end_time)
        session_id = result.get("session_id", "")
        if not session_id:
            return None
        token_count_raw = result.get("tokenCount", 0)
        request_count_raw = result.get("requests", 0)
        total_cost_raw = result.get("totalCost", 0.0)
        token_count = int(token_count_raw)
        request_count = int(request_count_raw)
        total_cost = float(total_cost_raw)

        return SessionUsageData(
            session_id=session_id,
            token_count=token_count,
            request_count=request_count,
            total_cost=total_cost,
        )

    def get_tenant_usage(
        self,
        tenant_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> TenantUsageData | Any:
        """
        Get tenant usage data.

        Args:
            tenant_id: Tenant identifier
            start_time: Start time for the usage data (in ISO 8601 UTC format)
            end_time: End time for the usage data (in ISO 8601 UTC format)

        Returns:
            TenantUsageData: Tenant usage data
        """
        if not tenant_id:
            logger.error("netra.usage: tenant_id is required to fetch tenant usage")
            return None
        result = self._client.get_tenant_usage(tenant_id, start_time=start_time, end_time=end_time)
        tenant_id = result.get("tenant_id", "")
        if not tenant_id:
            return None
        organisation_id = result.get("organisation_id")
        token_count_raw = result.get("tokenCount", 0)
        request_count_raw = result.get("requests", 0)
        session_count_raw = result.get("sessions", 0)
        total_cost_raw = result.get("totalCost", 0.0)
        token_count = int(token_count_raw)
        request_count = int(request_count_raw)
        session_count = int(session_count_raw)
        total_cost = float(total_cost_raw)

        return TenantUsageData(
            tenant_id=tenant_id,
            organisation_id=organisation_id,
            token_count=token_count,
            request_count=request_count,
            session_count=session_count,
            total_cost=total_cost,
        )
