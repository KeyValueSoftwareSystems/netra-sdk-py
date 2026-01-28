import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from netra.config import Config
from netra.simulation.models import ConversationResponse, ConversationStatus, SimulationItem

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 10.0


class SimulationHttpClient:
    """Internal HTTP client for simulation API endpoints."""

    def __init__(self, config: Config) -> None:
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """Create and configure the HTTP client."""
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.simulation: NETRA_OTLP_ENDPOINT is required")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.simulation: Failed to create HTTP client: %s", exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Extract base URL, removing telemetry suffix if present."""
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """Build request headers from configuration."""
        headers: Dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    def _get_timeout(self) -> float:
        """Get timeout from environment or use default."""
        timeout_str = os.getenv("NETRA_SIMULATION_TIMEOUT")
        if not timeout_str:
            return DEFAULT_TIMEOUT
        try:
            return float(timeout_str)
        except ValueError:
            logger.warning(
                "netra.simulation: Invalid timeout '%s', using default %.1f",
                timeout_str,
                DEFAULT_TIMEOUT,
            )
            return DEFAULT_TIMEOUT

    def create_run(
        self,
        dataset_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SimulationItem]:
        """Create a new simulation run for the specified dataset."""
        if not self._client:
            logger.error("netra.simulation: Client not initialized")
            return []

        # TODO: Replace mock with actual API call
        logger.info("netra.simulation: [MOCK] Creating run for dataset_id=%s", dataset_id)
        return [
            SimulationItem(item_id="item_001", message="Hello, how can I help you?", turn=1),
            SimulationItem(item_id="item_002", message="What is the weather today?", turn=1),
            SimulationItem(item_id="item_003", message="Tell me a joke", turn=1),
        ]

    def trigger_conversation(
        self,
        item_id: str,
        message: str,
        turn: int,
        session_id: str,
    ) -> Optional[ConversationResponse]:
        """Send a conversation turn to the backend and get the next response."""
        if not self._client:
            logger.error("netra.simulation: Client not initialized")
            return None

        # TODO: Replace mock with actual API call
        logger.info("netra.simulation: [MOCK] Conversation item_id=%s, turn=%d", item_id, turn)

        if turn >= 3:
            return ConversationResponse(message="", turn=turn, status=ConversationStatus.STOP)

        return ConversationResponse(
            message=f"Mock response for turn {turn + 1}",
            turn=turn + 1,
            status=ConversationStatus.CONTINUE,
        )

    def report_failure(self, item_id: str, turn: int, error: str) -> bool:
        """Report a task execution failure to the backend."""
        if not self._client:
            logger.error("netra.simulation: Client not initialized")
            return False

        # TODO: Replace mock with actual API call
        logger.info("netra.simulation: [MOCK] Failure item_id=%s, turn=%d, error=%s", item_id, turn, error)
        return True
