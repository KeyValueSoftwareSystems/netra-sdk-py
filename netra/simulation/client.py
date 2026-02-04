"""HTTP client for simulation API endpoints."""

import logging
import os
from typing import Any, Optional

import httpx

from netra.config import Config
from netra.simulation.models import ConversationResponse, SimulationItem

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0
_LOG_PREFIX = "netra.simulation"


class SimulationHttpClient:
    """Internal HTTP client for simulation API endpoints.

    Attributes:
        _client: The underlying httpx client instance.
    """

    __slots__ = ("_client",)

    def __init__(self, config: Config) -> None:
        """Initialize the simulation HTTP client.

        Args:
            config: The Netra configuration object.
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """Create and configure the HTTP client.

        Args:
            config: The Netra configuration object.

        Returns:
            Configured httpx client or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("%s: NETRA_OTLP_ENDPOINT is required", _LOG_PREFIX)
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("%s: Failed to create HTTP client: %s", _LOG_PREFIX, exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Extract base URL, removing telemetry suffix if present.

        Args:
            endpoint: The raw endpoint URL.

        Returns:
            The cleaned base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> dict[str, str]:
        """Build request headers from configuration.

        Args:
            config: The Netra configuration object.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    def _get_timeout(self) -> float:
        """Get timeout from environment or use default.

        Returns:
            The timeout value in seconds.
        """
        timeout_str = os.getenv("NETRA_SIMULATION_TIMEOUT")
        if not timeout_str:
            return _DEFAULT_TIMEOUT
        try:
            return float(timeout_str)
        except ValueError:
            logger.warning(
                "%s: Invalid timeout '%s', using default %.1f",
                _LOG_PREFIX,
                timeout_str,
                _DEFAULT_TIMEOUT,
            )
            return _DEFAULT_TIMEOUT

    def create_run(
        self,
        name: str,
        dataset_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a new simulation run for the specified dataset.

        Args:
            name: Name of the simulation run.
            dataset_id: Identifier of the dataset to simulate.
            context: Optional context data for the simulation.

        Returns:
            Dictionary containing run_id and simulation_items, or None on failure.
        """
        if not self._client:
            logger.error("%s: Client not initialized", _LOG_PREFIX)
            return None

        response: Optional[httpx.Response] = None
        try:
            url = "/evaluations/test_run/multi-turn"
            payload: dict[str, Any] = {
                "name": name,
                "datasetId": dataset_id,
                "context": context or {},
            }
            response = self._client.post(url, json=payload, timeout=500)
            response.raise_for_status()
            data = response.json()

            response_data = data.get("data", {})
            user_messages = response_data.get("userMessages", [])
            if not user_messages:
                logger.warning("%s: No user messages returned from create_run", _LOG_PREFIX)
                return None

            run_id = response_data.get("id", "")
            simulation_items = [
                SimulationItem(
                    run_item_id=msg.get("testRunItemId", ""),
                    message=msg.get("userMessage", ""),
                    turn_id=msg.get("turnId", ""),
                )
                for msg in user_messages
            ]
            return {
                "run_id": run_id,
                "simulation_items": simulation_items,
            }

        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to create simulation run: %s", _LOG_PREFIX, error_msg)
            return None

    def trigger_conversation(
        self,
        message: str,
        turn_id: str,
        session_id: str,
        trace_id: str,
    ) -> Optional[ConversationResponse]:
        """Send a conversation turn to the backend and get the next response.

        Args:
            message: Agent response message.
            turn_id: Turn identifier.
            session_id: Session identifier.
            trace_id: Trace identifier.

        Returns:
            ConversationResponse with next turn info, or None on failure.
        """
        if not self._client:
            logger.error("%s: Client not initialized", _LOG_PREFIX)
            return None

        response: Optional[httpx.Response] = None
        try:
            url = "/evaluations/turn/agent-response"
            payload: dict[str, Any] = {
                "turnId": turn_id,
                "agentResponse": {"message": message},
                "sessionId": session_id,
                "traceId": trace_id,
            }

            response = self._client.post(url, json=payload, timeout=500)
            response.raise_for_status()
            data = response.json()

            response_data = data.get("data", {})
            decision = response_data.get("decision", "continue")

            if decision == "stop":
                return ConversationResponse(
                    decision=decision,
                    reason=response_data.get("reason", ""),
                )

            user_messages = response_data.get("userMessages", [])
            if not user_messages:
                logger.warning("%s: No user messages in continue response", _LOG_PREFIX)
                return None

            next_msg = next(iter(user_messages))
            return ConversationResponse(
                decision=decision,
                next_turn_id=next_msg.get("turnId", ""),
                next_user_message=next_msg.get("userMessage", ""),
                next_run_item_id=next_msg.get("testRunItemId", ""),
            )

        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to trigger conversation: %s", _LOG_PREFIX, error_msg)
            return None

    def report_failure(self, run_id: str, run_item_id: str, error: str) -> None:
        """Report a task execution failure to the backend.

        Args:
            run_id: Identifier of the run.
            run_item_id: Identifier of the run item.
            error: Error message describing the failure.
        """
        if not self._client:
            logger.error("%s: Client not initialized", _LOG_PREFIX)
            return

        response: Optional[httpx.Response] = None
        try:
            url = f"/evaluations/run/{run_id}/item/{run_item_id}/status"
            payload: dict[str, Any] = {"status": "failed", "failureReason": error}
            response = self._client.patch(url, json=payload)
            response.raise_for_status()
            logger.info("%s: Reported failure - %s", _LOG_PREFIX, error)
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to report failure: %s", _LOG_PREFIX, error_msg)

    def post_run_status(self, run_id: str, status: str) -> Any:
        """Submit the run status.

        Args:
            run_id: The id of the run to update.
            status: The status of the run.

        Returns:
            Backend JSON response containing confirmation, or error dict.
        """
        if not self._client:
            logger.error("%s: Client not initialized; cannot post run status", _LOG_PREFIX)
            return {"success": False}

        response: Optional[httpx.Response] = None
        try:
            url = f"/evaluations/run/{run_id}/status"
            payload: dict[str, Any] = {"status": status}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Completed test run successfully", _LOG_PREFIX)
                return data.get("data", {})
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to post run status for run '%s': %s", _LOG_PREFIX, run_id, error_msg)
            return {"success": False}

    def _extract_error_message(
        self,
        response: Optional[httpx.Response],
        exc: Exception,
    ) -> Any:
        """Extract error message from response or exception.

        Args:
            response: The HTTP response object, if available.
            exc: The exception that was raised.

        Returns:
            A descriptive error message string.
        """
        if response is not None:
            try:
                response_json = response.json()
                error_data = response_json.get("error", {})
                if isinstance(error_data, dict):
                    return error_data.get("message", str(exc))
            except Exception:
                pass
        return str(exc)
