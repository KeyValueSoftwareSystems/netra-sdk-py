import logging
from typing import Any, Optional

import httpx

from netra.config import Config
from netra.simulation.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    TELEMETRY_SUFFIX,
    URL_AGENT_RESPONSE,
    URL_CREATE_RUN,
    URL_RUN_ITEM_STATUS,
    URL_RUN_STATUS,
)
from netra.simulation.models import ConversationResponse, ConversationStatus, FileData, SimulationItem
from netra.simulation.utils import parse_env_float

logger = logging.getLogger(__name__)


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

    def close(self) -> None:
        """Close the underlying HTTP client and release connection resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                logger.debug("%s: Error closing HTTP client", LOG_PREFIX, exc_info=True)
            finally:
                self._client = None

    def _ensure_client(self) -> Optional[httpx.Client]:
        """Return the underlying client, logging an error if it is not initialized.

        Returns:
            The httpx client, or None if not available.
        """
        if not self._client:
            logger.error("%s: Client not initialized", LOG_PREFIX)
        return self._client

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """Create and configure the HTTP client.

        Args:
            config: The Netra configuration object.

        Returns:
            Configured httpx client or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("%s: NETRA_OTLP_ENDPOINT is required", LOG_PREFIX)
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = parse_env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT)

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("%s: Failed to create HTTP client: %s", LOG_PREFIX, exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Extract base URL, removing telemetry suffix if present.

        Args:
            endpoint: The raw endpoint URL.

        Returns:
            The cleaned base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith(TELEMETRY_SUFFIX):
            base_url = base_url[: -len(TELEMETRY_SUFFIX)]
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

    def create_run(
        self,
        name: str,
        dataset_id: str,
        context: Optional[dict[str, Any]] = None,
        hooks_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a new simulation run for the specified dataset.

        Args:
            name: Name of the simulation run.
            dataset_id: Identifier of the dataset to simulate.
            context: Optional context data for the simulation.
            hooks_meta: Optional metadata describing configured hooks, stored
                by the backend for UI display purposes.

        Returns:
            Dictionary containing run_id and simulation_items, or None on failure.
        """
        if not self._ensure_client():
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_CREATE_RUN
            payload: dict[str, Any] = {
                "name": name,
                "datasetId": dataset_id,
                "context": context or {},
            }
            if hooks_meta:
                payload["hooksMeta"] = hooks_meta
            response = self._client.post(url, json=payload)  # type:ignore[union-attr]
            response.raise_for_status()
            data = response.json()

            response_data = data.get("data", {})
            user_messages = response_data.get("userMessages", [])
            if not user_messages:
                logger.warning("%s: No user messages returned from create_run", LOG_PREFIX)
                return None

            run_id = response_data.get("id", "")
            simulation_items = [
                SimulationItem(
                    run_item_id=msg.get("testRunItemId", ""),
                    dataset_item_id=msg.get("datasetItemId", ""),
                    message=msg.get("userMessage", ""),
                    turn_id=msg.get("turnId", ""),
                    files=self._parse_files(msg.get("attachments")),
                )
                for msg in user_messages
            ]
            return {
                "run_id": run_id,
                "simulation_items": simulation_items,
            }

        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to create simulation run: %s", LOG_PREFIX, error_msg)
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
        if not self._ensure_client():
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_AGENT_RESPONSE
            payload: dict[str, Any] = {
                "turnId": turn_id,
                "agentResponse": {"message": message},
                "sessionId": session_id,
                "traceId": trace_id,
            }

            response = self._client.post(url, json=payload)  # type:ignore[union-attr]
            response.raise_for_status()
            data = response.json()

            response_data = data.get("data", {})
            raw_decision = response_data.get("decision", "continue")
            decision = ConversationStatus(raw_decision)

            if decision == ConversationStatus.STOP:
                return ConversationResponse(
                    decision=decision,
                    reason=response_data.get("reason", ""),
                )

            user_messages = response_data.get("userMessages", [])
            if not user_messages:
                logger.warning("%s: No user messages in continue response", LOG_PREFIX)
                return None

            next_msg = next(iter(user_messages))
            return ConversationResponse(
                decision=decision,
                next_turn_id=next_msg.get("turnId", ""),
                next_user_message=next_msg.get("userMessage", ""),
                next_files=self._parse_files(next_msg.get("attachments")),
            )

        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to trigger conversation: %s", LOG_PREFIX, error_msg)
            raise

    def report_failure(
        self,
        run_id: str,
        run_item_id: str,
        error: str,
        status: str = "failed",
    ) -> None:
        """Report a task execution failure to the backend.

        Args:
            run_id: Identifier of the run.
            run_item_id: Identifier of the run item.
            error: Error message describing the failure.
            status: The run status to set on the item. Use ``"prescript_failed"``
                when the failure originated from a ``before_all`` or ``before``
                hook. Defaults to ``"failed"``.
        """
        if not self._ensure_client():
            return

        response: Optional[httpx.Response] = None
        try:
            url = URL_RUN_ITEM_STATUS.format(run_id=run_id, run_item_id=run_item_id)
            payload: dict[str, Any] = {"status": status, "failureReason": error}
            response = self._client.patch(url, json=payload)  # type:ignore[union-attr]
            response.raise_for_status()
            logger.info("%s: Reported failure - %s", LOG_PREFIX, error)
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to report failure: %s", LOG_PREFIX, error_msg)

    def post_run_status(self, run_id: str, status: str) -> Any:
        """Submit the run status.

        Args:
            run_id: The id of the run to update.
            status: The status of the run.

        Returns:
            Backend JSON response containing confirmation, or error dict.
        """
        if not self._ensure_client():
            return {"success": False}

        response: Optional[httpx.Response] = None
        try:
            url = URL_RUN_STATUS.format(run_id=run_id)
            payload: dict[str, Any] = {"status": status}
            response = self._client.post(url, json=payload)  # type:ignore[union-attr]
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Test run status %s", LOG_PREFIX, status)
                return data.get("data", {})
            return data
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.error("%s: Failed to post run status for run '%s': %s", LOG_PREFIX, run_id, error_msg)
            return {"success": False}

    @staticmethod
    def _parse_files(raw_files: list[dict[str, str]] | None) -> list[FileData]:
        """Parse raw file entries from the backend response into FileData objects.

        Args:
            raw_files: List of file dictionaries from the JSON response, or None.

        Returns:
            List of FileData objects. Malformed entries are skipped.
        """
        if not raw_files or not isinstance(raw_files, list):
            return []

        parsed: list[FileData] = []
        for entry in raw_files:
            if not isinstance(entry, dict):
                continue
            file_name = entry.get("fileName", "")
            download_url = entry.get("downloadUrl", "")
            if not file_name or not download_url:
                logger.warning(
                    "%s: Skipping file entry with missing fileName or downloadUrl",
                    LOG_PREFIX,
                )
                continue
            parsed.append(
                FileData(
                    file_name=file_name,
                    content_type=entry.get("contentType", ""),
                    description=entry.get("description"),
                    download_url=download_url,
                )
            )
        return parsed

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
                logger.debug("%s: Could not parse error from response body", LOG_PREFIX, exc_info=True)
        return str(exc)
