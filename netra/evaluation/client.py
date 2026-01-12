import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from netra.config import Config
from netra.evaluation.models import DatasetItem

logger = logging.getLogger(__name__)


class EvaluationHttpClient:
    """
    Internal HTTP client for Evaluation APIs.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize HTTP client for evaluation endpoints.

        Args:
            config: The configuration object.
        """
        self._client: Optional[httpx.Client] = self._create_client(config)

    def _create_client(self, config: Config) -> Optional[httpx.Client]:
        """
        Create an HTTP client for evaluation endpoints.

        Args:
            config: The configuration object.

        Returns:
            An HTTP client for evaluation endpoints, or None if creation fails.
        """
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.evaluation: NETRA_OTLP_ENDPOINT is required for evaluation APIs")
            return None

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = self._get_timeout()

        try:
            return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.evaluation: Failed to initialize evaluation HTTP client: %s", exc)
            return None

    def _resolve_base_url(self, endpoint: str) -> str:
        """
        Resolve base URL from endpoint.

        Args:
            endpoint: The endpoint to resolve.

        Returns:
            The resolved base URL.
        """
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]
        return base_url

    def _build_headers(self, config: Config) -> Dict[str, str]:
        """
        Build Headers for Evaluation Client

        Args:
            config: The configuration object.

        Returns:
            The headers for evaluation client.
        """
        headers: Dict[str, str] = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def _get_timeout(self) -> float:
        """
        Get timeout for evaluation client.

        Returns:
            The timeout for evaluation client.
        """
        timeout_env = os.getenv("NETRA_EVALUATION_TIMEOUT")
        if not timeout_env:
            return 10.0
        try:
            return float(timeout_env)
        except ValueError:
            logger.warning(
                "netra.evaluation: Invalid NETRA_EVALUATION_TIMEOUT value '%s', using default 10.0",
                timeout_env,
            )
            return 10.0

    def create_dataset(self, name: Optional[str], tags: Optional[List[str]] = None) -> Any:
        """
        Create an empty dataset

        Args:
            name: The name of the dataset.
            tags: Optional list of tags to associate with the dataset.

        Returns:
            A backend JSON response containing dataset info (id, name, tags, etc.) on success,
            or None if creation fails.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot create dataset")
            return None
        try:
            url = "/evaluations/dataset"
            payload: Dict[str, Any] = {
                "name": name,
                "tags": tags if tags else [],
            }
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("netra.evaluation: Dataset created successfully")
                return data.get("data", {})
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to create dataset: %s", response_json.get("error").get("message", "")
            )
            return None

    def add_dataset_item(self, dataset_id: str, item: DatasetItem) -> Any:
        """
        Add a single item to an existing dataset and return backend data (e.g., new item id).

        Args:
            dataset_id: The id of the dataset to which the item will be added.
            item_payload: The dataset item to add.

        Returns:
            A backend JSON response on success or {"success": False} on error.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot add item to dataset")
            return {"success": False}
        try:
            url = f"/evaluations/dataset/{dataset_id}/items"
            item_payload: Dict[str, Any] = {
                "input": item.input if item.input else None,
                "expectedOutput": item.expected_output if item.expected_output else None,
                "tags": item.tags if item.tags else None,
                "metadata": item.metadata if item.metadata else None,
            }
            response = self._client.post(url, json=item_payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("netra.evaluation: Dataset item added successfully")
                return data.get("data", {})
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to add item to dataset '%s': %s",
                dataset_id,
                response_json.get("error").get("message", ""),
            )
            return None

    def get_dataset(self, dataset_id: str) -> Any:
        """
        Fetch dataset items for a dataset id.

        Args:
            dataset_id: The id of the dataset to fetch.

        Returns:
            A list of dataset items.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot fetch dataset")
            return {"success": False}
        try:
            url = f"/evaluations/dataset/{dataset_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("netra.evaluation: Dataset fetched successfully")
                return data.get("data", [])
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to fetch dataset '%s': %s",
                dataset_id,
                response_json.get("error").get("message", ""),
            )
            return None

    def create_run(
        self,
        name: str,
        dataset_id: Optional[str] = None,
        evaluators_config: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Create a new run based on the provided name, dataset_id, and evaluators_config.

        Args:
            name: The name of the run.
            dataset_id: The id of the dataset to which the run will be associated.
            evaluators_config: Optional list of evaluators to be used for the run.

        Returns:
            A backend JSON response containing run_id
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot create run")
            return {"success": False}
        try:
            url = f"/evaluations/test_run"
            payload: Dict[str, Any] = {
                "name": name,
                "datasetId": dataset_id if dataset_id else None,
                "localEvaluators": evaluators_config if evaluators_config else [],
            }
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to create run '%s': %s", name, response_json.get("error").get("message", "")
            )
            return {"success": False}

    def post_run_item(self, run_id: str, payload: Dict[str, Any]) -> Any:
        """
        Submit a new run item to the backend.

        Args:
            run_id: The id of the run to which the item will be added.
            payload: The run item to add.

        Returns:
            A backend JSON response on success or {"success": False} on error.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot post run item")
            return {"success": False}
        try:
            url = f"/evaluations/run/{run_id}/item"
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                run_item = data.get("data", {}).get("item")
                run_item_id = run_item.get("id")
                return run_item_id
            return data
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to post run item for run '%s': %s",
                run_id,
                response_json.get("error").get("message", ""),
            )
            return {"success": False}

    def submit_local_evaluations(
        self, run_id: str, test_run_item_id: str, evaluator_results: List[Dict[str, Any]]
    ) -> Any:
        """
        Submit local evaluations result

        Args:
            run_id: The id of the run to which the item will be added.
            test_run_item_id: The id of the test run item.
            evaluator_results: The evaluator results to submit.

        Returns:
            A backend JSON response containing confirmation of the submission.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot submit local evaluations")
            return {"success": False}
        try:
            url = f"/evaluations/run/{run_id}/item/{test_run_item_id}/local-evaluations"
            payload: Dict[str, Any] = {"evaluatorResults": evaluator_results}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to submit local evaluations for run '%s', item '%s': %s",
                run_id,
                test_run_item_id,
                response_json.get("error").get("message", ""),
            )
            return {"success": False}

    def post_run_status(self, run_id: str, status: str) -> Any:
        """
        Submit the run status

        Args:
             run_id: The id of the run to which the item will be added.
             status: The status of the run.

         Returns:
             A backend JSON response containing confirmation of the submission.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot post run status")
            return {"success": False}
        try:
            url = f"/evaluations/run/{run_id}/status"
            payload: Dict[str, Any] = {"status": status}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("netra.evaluation: Completed test run successfully")
                return data.get("data", {})
            return data
        except Exception:
            response_json = response.json()
            logger.error(
                "netra.evaluation: Failed to post run status for run '%s': %s",
                run_id,
                response_json.get("error").get("message", ""),
            )
            return {"success": False}

    def get_span_by_id(self, span_id: str) -> Any:
        """
        Check if a span exists in the backend.

        Args:
            span_id: The span ID to check.

        Returns:
            The span data if found, None otherwise.
        """
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot get span")
            return None
        try:
            url = f"sdk/traces/spans/{span_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("data", data)
            return data
        except Exception:
            return None

    async def wait_for_span_ingestion(
        self,
        span_id: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        initial_delay_seconds: float = 0.5,
    ) -> bool:
        """
        Wait until a span is available in the backend.

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
