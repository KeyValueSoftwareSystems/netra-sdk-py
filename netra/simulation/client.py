"""HTTP client for simulation API endpoints."""

import logging
from typing import Any, Dict, Optional

from netra.client import BaseNetraClient
from netra.config import Config
from netra.simulation.models import ConversationResponse, SimulationItem

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.simulation"


class SimulationHttpClient(BaseNetraClient):
    """Internal HTTP client for simulation API endpoints."""

    def __init__(self, config: Config) -> None:
        """Initialize the simulation HTTP client.

        Args:
            config: The Netra configuration object.
        """
        super().__init__(
            config,
            log_prefix=_LOG_PREFIX,
            timeout_env_var="NETRA_SIMULATION_TIMEOUT",
            default_timeout=500.0,
        )

    def create_run(
        self,
        name: str,
        dataset_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
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

        try:
            url = "/evaluations/test_run/multi-turn"
            payload: Dict[str, Any] = {
                "name": name,
                "datasetId": dataset_id,
                "context": context or {},
            }
            response = self._client.post(url, json=payload)
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
            error_msg = self._extract_error_message(exc)
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

        try:
            url = "/evaluations/turn/agent-response"
            payload: Dict[str, Any] = {
                "turnId": turn_id,
                "agentResponse": {"message": message},
                "sessionId": session_id,
                "traceId": trace_id,
            }

            response = self._client.post(url, json=payload)
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
            error_msg = self._extract_error_message(exc)
            logger.error("%s: Failed to trigger conversation: %s", _LOG_PREFIX, error_msg)
            raise

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

        try:
            url = f"/evaluations/run/{run_id}/item/{run_item_id}/status"
            payload: Dict[str, Any] = {"status": "failed", "failureReason": error}
            self._client.patch(url, json=payload).raise_for_status()
            logger.info("%s: Reported failure - %s", _LOG_PREFIX, error)
        except Exception as exc:
            logger.error("%s: Failed to report failure: %s", _LOG_PREFIX, self._extract_error_message(exc))

    def post_run_status(self, run_id: str, status: str) -> Any:
        """Submit the run status.

        Args:
            run_id: The id of the run to update.
            status: The status of the run.

        Returns:
            Backend JSON response containing confirmation, or None on failure.
        """
        if not self._client:
            logger.error("%s: Client not initialized; cannot post run status", _LOG_PREFIX)
            return None

        try:
            url = f"/evaluations/run/{run_id}/status"
            payload: Dict[str, Any] = {"status": status}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Test run status %s", _LOG_PREFIX, status)
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error(
                "%s: Failed to post run status for run '%s': %s",
                _LOG_PREFIX,
                run_id,
                self._extract_error_message(exc),
            )
            return None
