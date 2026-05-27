import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from netra.config import Config
from netra.evaluation.constants import (
    DEFAULT_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    TELEMETRY_SUFFIX,
    URL_CREATE_DATASET,
    URL_CREATE_RUN,
    URL_DATASET_ITEMS,
    URL_GET_DATASET,
    URL_GET_RUN,
    URL_LOCAL_EVALUATIONS,
    URL_RUN_ITEM,
    URL_RUN_STATUS,
    URL_SPAN,
)
from netra.evaluation.models import DatasetItem, TurnType
from netra.evaluation.utils import parse_env_float

logger = logging.getLogger(__name__)


class EvaluationHttpClient:
    """Internal HTTP client for Evaluation APIs.

    Attributes:
        _client: The underlying httpx client instance.
    """

    __slots__ = ("_client",)

    def __init__(self, config: Config) -> None:
        """Initialize HTTP client for evaluation endpoints.

        Args:
            config: The configuration object.
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
        """Create an HTTP client for evaluation endpoints.

        Args:
            config: The configuration object.

        Returns:
            An HTTP client for evaluation endpoints, or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("%s: NETRA_OTLP_ENDPOINT is required for evaluation APIs", LOG_PREFIX)
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = parse_env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT)

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception:
            logger.exception("%s: Failed to initialize evaluation HTTP client", LOG_PREFIX)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """Extract base URL, removing telemetry suffix if present.

        Args:
            endpoint: The endpoint to resolve.

        Returns:
            The resolved base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith(TELEMETRY_SUFFIX):
            base_url = base_url[: -len(TELEMETRY_SUFFIX)]
        return base_url

    def _build_headers(self, config: Config) -> dict[str, str]:
        """Build request headers from configuration.

        Args:
            config: The configuration object.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    def _extract_error_message(
        self,
        response: Optional[httpx.Response],
        exc: Exception,
    ) -> str:
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
                    msg = error_data.get("message")
                    if isinstance(msg, str):
                        return msg
            except Exception:
                logger.debug("%s: Could not parse error from response body", LOG_PREFIX, exc_info=True)
        return str(exc)

    def _post_data(
        self,
        url: str,
        payload: dict[str, Any],
        error_context: str,
    ) -> Optional[dict[str, Any]]:
        """Send a POST request and return the unwrapped ``data`` envelope.

        Args:
            url: The endpoint URL.
            payload: The JSON payload to send.
            error_context: Description used in the error log message.

        Returns:
            The ``data`` dict from the response envelope, or None on failure.
        """
        client = self._ensure_client()
        if not client:
            return None

        response: Optional[httpx.Response] = None
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict) and "data" in body:
                result = body["data"]
                return result if isinstance(result, dict) else None
            return None
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: %s: %s", LOG_PREFIX, error_context, error_msg)
            return None

    def _get_data(
        self,
        url: str,
        error_context: str,
    ) -> Optional[Any]:
        """Send a GET request and return the unwrapped ``data`` envelope.

        Args:
            url: The endpoint URL.
            error_context: Description used in the error log message.

        Returns:
            The ``data`` value from the response envelope, or None on failure.
        """
        client = self._ensure_client()
        if not client:
            return None

        response: Optional[httpx.Response] = None
        try:
            response = client.get(url)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict) and "data" in body:
                return body["data"]
            return None
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: %s: %s", LOG_PREFIX, error_context, error_msg)
            return None

    def create_dataset(
        self,
        name: Optional[str],
        tags: Optional[list[str]] = None,
        turn_type: TurnType = TurnType.SINGLE,
    ) -> Optional[dict[str, Any]]:
        """Create an empty dataset.

        Args:
            name: The name of the dataset.
            tags: Optional list of tags to associate with the dataset.
            turn_type: The turn type of the dataset. Defaults to "single".

        Returns:
            A backend JSON response containing dataset info on success, or None on failure.
        """
        payload: dict[str, Any] = {"name": name, "tags": tags if tags else [], "turnType": turn_type.value}
        result = self._post_data(URL_CREATE_DATASET, payload, "Failed to create dataset")
        if result is not None:
            logger.info("%s: Dataset created successfully", LOG_PREFIX)
        return result

    def add_dataset_item(self, dataset_id: str, item: DatasetItem) -> Optional[dict[str, Any]]:
        """Add a single item to an existing dataset.

        Args:
            dataset_id: The id of the dataset to which the item will be added.
            item: The dataset item to add.

        Returns:
            A backend JSON response on success, or None on failure.
        """
        url = URL_DATASET_ITEMS.format(dataset_id=dataset_id)
        item_payload: dict[str, Any] = {
            "input": item.input if item.input else None,
            "expectedOutput": item.expected_output if item.expected_output else None,
            "tags": item.tags if item.tags else None,
            "metadata": item.metadata if item.metadata else None,
        }
        result = self._post_data(url, item_payload, f"Failed to add item to dataset '{dataset_id}'")
        if result is not None:
            logger.info("%s: Dataset item added successfully", LOG_PREFIX)
        return result

    def get_dataset(self, dataset_id: str) -> Optional[list[dict[str, Any]]]:
        """Fetch dataset items for a dataset id.

        Args:
            dataset_id: The id of the dataset to fetch.

        Returns:
            A list of dataset item dicts, or None on failure.
        """
        client = self._ensure_client()
        if not client:
            return None

        response: Optional[httpx.Response] = None
        try:
            url = URL_GET_DATASET.format(dataset_id=dataset_id)
            response = client.get(url)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict) and "data" in body:
                logger.info("%s: Dataset fetched successfully", LOG_PREFIX)
                data = body["data"]
                return data if isinstance(data, list) else None
            return None
        except Exception as exc:
            error_msg = self._extract_error_message(response, exc)
            logger.exception("%s: Failed to fetch dataset '%s': %s", LOG_PREFIX, dataset_id, error_msg)
            return None

    def create_run(
        self,
        name: str,
        dataset_id: Optional[str] = None,
        evaluators_config: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[dict[str, Any]]:
        """Create a new run based on the provided name, dataset_id, and evaluators_config.

        Args:
            name: The name of the run.
            dataset_id: The id of the dataset to which the run will be associated.
            evaluators_config: Optional list of evaluators to be used for the run.

        Returns:
            A backend JSON response containing run info, or None on failure.
        """
        payload: dict[str, Any] = {
            "name": name,
            "datasetId": dataset_id if dataset_id else None,
            "localEvaluators": evaluators_config if evaluators_config else [],
        }
        return self._post_data(URL_CREATE_RUN, payload, f"Failed to create run '{name}'")

    def post_run_item(self, run_id: str, payload: dict[str, Any]) -> Optional[str]:
        """Submit a new run item to the backend.

        Args:
            run_id: The id of the run to which the item will be added.
            payload: The run item to add.

        Returns:
            The run item id on success, or None on failure.
        """
        url = URL_RUN_ITEM.format(run_id=run_id)
        result = self._post_data(url, payload, f"Failed to post run item for run '{run_id}'")
        if result is not None:
            run_item = result.get("item", {})
            if isinstance(run_item, dict):
                item_id = run_item.get("id")
                return str(item_id) if item_id is not None else None
        return None

    def submit_local_evaluations(
        self,
        run_id: str,
        test_run_item_id: str,
        evaluator_results: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Submit local evaluations result.

        Args:
            run_id: The id of the run.
            test_run_item_id: The id of the test run item.
            evaluator_results: The evaluator results to submit.

        Returns:
            A backend JSON response containing confirmation, or None on failure.
        """
        url = URL_LOCAL_EVALUATIONS.format(run_id=run_id, test_run_item_id=test_run_item_id)
        request_payload: dict[str, Any] = {"evaluatorResults": evaluator_results}
        return self._post_data(
            url,
            request_payload,
            f"Failed to submit local evaluations for run '{run_id}', item '{test_run_item_id}'",
        )

    def post_run_status(self, run_id: str, status: str) -> Optional[dict[str, Any]]:
        """Submit the run status.

        Args:
            run_id: The id of the run to update.
            status: The status of the run.

        Returns:
            A backend JSON response containing confirmation, or None on failure.
        """
        payload: dict[str, Any] = {"status": status}
        url = URL_RUN_STATUS.format(run_id=run_id)
        result = self._post_data(url, payload, f"Failed to post run status for run '{run_id}'")
        if result is not None:
            logger.info("%s: Test run status updated to '%s'", LOG_PREFIX, status)
        return result

    def get_run_results(self, run_id: str) -> Optional[dict[str, Any]]:
        """Fetch test run results by run ID.

        Args:
            run_id: The id of the run to fetch.

        Returns:
            A JSON response containing run results, or None on failure.
        """
        url = URL_GET_RUN.format(run_id=run_id)
        result = self._get_data(url, f"Failed to fetch run results for run '{run_id}'")
        if result is not None:
            logger.info("%s: Run fetched successfully", LOG_PREFIX)
        return result if isinstance(result, dict) else None

    def get_span_by_id(self, span_id: str) -> Optional[dict[str, Any]]:
        """Check if a span exists in the backend.

        Args:
            span_id: The span ID to check.

        Returns:
            The span data if found, None otherwise.
        """
        if not self._ensure_client():
            return None

        try:
            url = URL_SPAN.format(span_id=span_id)
            response = self._client.get(url)  # type:ignore[union-attr]
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                data = body.get("data", body)
                return data if isinstance(data, dict) else None
            return None
        except Exception:
            return None

    async def wait_for_span_ingestion(
        self,
        span_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        initial_delay_seconds: float = 0.5,
    ) -> bool:
        """Wait until a span is available in the backend.

        Polls the GET /spans/:id endpoint to verify span availability
        before running evaluators.

        Args:
            span_id: The span ID to poll for.
            timeout_seconds: Maximum time to wait for span ingestion.
            poll_interval_seconds: Time between polling attempts.
            initial_delay_seconds: Initial delay before first poll attempt.

        Returns:
            True if span was found within timeout, False otherwise.
        """
        if not span_id:
            return False

        await asyncio.sleep(initial_delay_seconds)

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            span_data = await asyncio.to_thread(self.get_span_by_id, span_id)
            if span_data is not None:
                return True
            await asyncio.sleep(poll_interval_seconds)

        return False
