import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from netra.client import BaseNetraClient
from netra.config import Config
from netra.evaluation.models import DatasetItem, TurnType

logger = logging.getLogger(__name__)

_LOG_PREFIX = "netra.evaluation"


class EvaluationHttpClient(BaseNetraClient):
    """Internal HTTP client for Evaluation APIs."""

    def __init__(self, config: Config) -> None:
        """
        Initialize HTTP client for evaluation endpoints.

        Args:
            config: The configuration object.
        """
        super().__init__(
            config,
            log_prefix=_LOG_PREFIX,
            timeout_env_var="NETRA_EVALUATION_TIMEOUT",
        )

    def create_dataset(
        self, name: Optional[str], tags: Optional[List[str]] = None, turn_type: TurnType = TurnType.SINGLE
    ) -> Any:
        """
        Create an empty dataset.

        Args:
            name: The name of the dataset.
            tags: Optional list of tags to associate with the dataset.
            turn_type: The turn type of the dataset, either "single" or "multi". Defaults to "single".

        Returns:
            A backend JSON response containing dataset info on success, or None if creation fails.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot create dataset", _LOG_PREFIX)
            return None
        try:
            url = "/evaluations/dataset"
            payload: Dict[str, Any] = {"name": name, "tags": tags if tags else [], "turnType": turn_type.value}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Dataset created successfully", _LOG_PREFIX)
                return data.get("data", {})
        except Exception as exc:
            logger.error("%s: Failed to create dataset: %s", _LOG_PREFIX, self._extract_error_message(exc))
            return None

    def add_dataset_item(self, dataset_id: str, item: DatasetItem) -> Any:
        """
        Add a single item to an existing dataset.

        Args:
            dataset_id: The id of the dataset to which the item will be added.
            item: The dataset item to add.

        Returns:
            A backend JSON response on success or None on error.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot add item to dataset", _LOG_PREFIX)
            return None
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
                logger.info("%s: Dataset item added successfully", _LOG_PREFIX)
                return data.get("data", {})
        except Exception as exc:
            logger.error(
                "%s: Failed to add item to dataset '%s': %s",
                _LOG_PREFIX,
                dataset_id,
                self._extract_error_message(exc),
            )
            return None

    def get_dataset(self, dataset_id: str) -> Any:
        """
        Fetch dataset items for a dataset id.

        Args:
            dataset_id: The id of the dataset to fetch.

        Returns:
            A list of dataset items, or None on error.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot fetch dataset", _LOG_PREFIX)
            return None
        try:
            url = f"/evaluations/dataset/{dataset_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Dataset fetched successfully", _LOG_PREFIX)
                return data.get("data", [])
        except Exception as exc:
            logger.error(
                "%s: Failed to fetch dataset '%s': %s",
                _LOG_PREFIX,
                dataset_id,
                self._extract_error_message(exc),
            )
            return None

    def create_run(
        self,
        name: str,
        dataset_id: Optional[str] = None,
        evaluators_config: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Create a new evaluation run.

        Args:
            name: The name of the run.
            dataset_id: The id of the dataset to which the run will be associated.
            evaluators_config: Optional list of evaluators to be used for the run.

        Returns:
            A backend JSON response containing run_id, or None on failure.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot create run", _LOG_PREFIX)
            return None
        try:
            url = "/evaluations/test_run"
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
            return data
        except Exception as exc:
            logger.error("%s: Failed to create run '%s': %s", _LOG_PREFIX, name, self._extract_error_message(exc))
            return None

    def post_run_item(self, run_id: str, payload: Dict[str, Any]) -> Any:
        """
        Submit a new run item to the backend.

        Args:
            run_id: The id of the run to which the item will be added.
            payload: The run item to add.

        Returns:
            The run item id on success, or None on failure.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot post run item", _LOG_PREFIX)
            return None
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
        except Exception as exc:
            logger.error(
                "%s: Failed to post run item for run '%s': %s",
                _LOG_PREFIX,
                run_id,
                self._extract_error_message(exc),
            )
            return None

    def submit_local_evaluations(
        self, run_id: str, test_run_item_id: str, evaluator_results: List[Dict[str, Any]]
    ) -> Any:
        """
        Submit local evaluations result.

        Args:
            run_id: The id of the run.
            test_run_item_id: The id of the test run item.
            evaluator_results: The evaluator results to submit.

        Returns:
            A backend JSON response containing confirmation of the submission.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot submit local evaluations", _LOG_PREFIX)
            return None
        try:
            url = f"/evaluations/run/{run_id}/item/{test_run_item_id}/local-evaluations"
            payload: Dict[str, Any] = {"evaluatorResults": evaluator_results}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
            return data
        except Exception as exc:
            logger.error(
                "%s: Failed to submit local evaluations for run '%s', item '%s': %s",
                _LOG_PREFIX,
                run_id,
                test_run_item_id,
                self._extract_error_message(exc),
            )
            return None

    def post_run_status(self, run_id: str, status: str) -> Any:
        """
        Submit the run status.

        Args:
            run_id: The id of the run.
            status: The status of the run.

        Returns:
            A backend JSON response containing confirmation of the submission.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot post run status", _LOG_PREFIX)
            return None
        try:
            url = f"/evaluations/run/{run_id}/status"
            payload: Dict[str, Any] = {"status": status}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                logger.info("%s: Completed test run successfully", _LOG_PREFIX)
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

    def get_span_by_id(self, span_id: str) -> Any:
        """
        Check if a span exists in the backend.

        Args:
            span_id: The span ID to check.

        Returns:
            The span data if found, None otherwise.
        """
        if not self._client:
            logger.error("%s: Client is not initialized; cannot get span", _LOG_PREFIX)
            return None
        try:
            url = f"/sdk/traces/spans/{span_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data.get("data", data)
            return data
        except Exception as exc:
            logger.error("%s: Failed to get span '%s': %s", _LOG_PREFIX, span_id, self._extract_error_message(exc))
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
