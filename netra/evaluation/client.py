import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from netra.config import Config
from netra.evaluation.models import EntryStatus, EvaluationScore, RunStatus

logger = logging.getLogger(__name__)


class _EvaluationHttpClient:
    def __init__(self, config: Config) -> None:
        """Initialize HTTP client for evaluation endpoints.

        If NETRA_OTLP_ENDPOINT is not provided, the client will be disabled but
        methods will log errors and return safe defaults instead of raising.
        """
        self._client: Optional[httpx.Client] = None
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            logger.error("netra.evaluation: NETRA_OTLP_ENDPOINT is required for evaluation APIs")
            return

        base_url = endpoint.rstrip("/")
        # Normalize base if user pointed to OTLP endpoints
        if base_url.endswith("/telemetry"):
            base_url = base_url[: -len("/telemetry")]

        headers = dict(config.headers or {})
        api_key = config.api_key
        if api_key:
            headers["x-api-key"] = api_key
        timeout_env = os.getenv("NETRA_EVALUATION_TIMEOUT")
        try:
            timeout = float(timeout_env) if timeout_env else 10.0
        except ValueError:
            logger.warning(
                "netra.evaluation: Invalid NETRA_EVALUATION_TIMEOUT value '%s', using default 10.0", timeout_env
            )
            timeout = 10.0
        try:
            self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.error("netra.evaluation: Failed to initialize evaluation HTTP client: %s", exc)
            self._client = None

    def get_dataset(self, dataset_id: str) -> Any:
        """Fetch dataset items for a dataset id.

        Returns an empty list on error and logs the error.
        """
        if not self._client:
            logger.error(
                "netra.evaluation: Evaluation client is not initialized; cannot fetch dataset '%s'", dataset_id
            )
            return []
        try:
            url = f"/evaluations/dataset/{dataset_id}"
            response = self._client.get(url)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", [])
        except Exception as exc:
            logger.error("netra.evaluation: Failed to fetch dataset '%s': %s", dataset_id, exc)
        return []

    def create_run(self, dataset_id: str, name: str) -> Any:
        """Create a run for a dataset.

        Returns a backend JSON response on success or {"success": False} on error.
        """
        if not self._client:
            logger.error(
                "netra.evaluation: Evaluation client is not initialized; cannot create run for dataset '%s'", dataset_id
            )
            return {"success": False}
        try:
            url = f"/evaluations/run/dataset/{dataset_id}"
            payload = {"name": name}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
        except Exception as exc:
            logger.error("netra.evaluation: Failed to create run for dataset '%s': %s", dataset_id, exc)
            return {"success": False}
        return {"success": False}

    def create_dataset(
        self, name: Optional[str], tags: Optional[List[str]] = None, policy_ids: Optional[List[str]] = None
    ) -> Any:
        """Create an empty dataset and return backend data (expects an id)."""
        if not self._client:
            logger.error("netra.evaluation: Evaluation client is not initialized; cannot create dataset")
            return {"success": False}
        try:
            url = "/evaluations/dataset"
            payload: Dict[str, Any] = {
                "name": name,
                "tags": tags if tags else [],
                "policyIds": policy_ids if policy_ids else [],
            }
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
        except Exception as exc:
            logger.error("netra.evaluation: Failed to create dataset: %s", exc)
            return {"success": False}
        return {"success": False}

    def add_dataset_entry(self, dataset_id: str, item_payload: Dict[str, Any]) -> Any:
        """Add a single item to an existing dataset and return backend data (e.g., new item id)."""
        if not self._client:
            logger.error(
                "netra.evaluation: Evaluation client is not initialized; cannot add item to dataset '%s'",
                dataset_id,
            )
            return {"success": False}
        try:
            url = f"/evaluations/dataset/{dataset_id}/items"
            response = self._client.post(url, json=item_payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data.get("data", {})
        except Exception as exc:
            logger.error("netra.evaluation: Failed to add item to dataset '%s': %s", dataset_id, exc)
            return {"success": False}
        return {"success": False}

    def post_entry_status(
        self,
        run_id: str,
        test_id: str,
        status: EntryStatus,
        trace_id: Optional[str],
        session_id: Optional[str],
        score: Optional[List[EvaluationScore]] = None,
    ) -> None:
        """Post per-entry status. Logs errors and returns None on failure."""
        if not self._client:
            logger.error(
                "netra.evaluation: Evaluation client is not initialized; cannot post status '%s' for run '%s' test '%s'",
                status.value,
                run_id,
                test_id,
            )
            return
        try:
            url = f"/evaluations/run/{run_id}/test/{test_id}"
            payload: Dict[str, Any] = {
                "status": status.value,
                "traceId": trace_id,
                "sessionId": session_id if session_id else None,
            }
            # Include score objects when provided
            if score is not None:
                try:
                    normalized_score: List[Dict[str, Any]] = []
                    for item in score:
                        if item is None:
                            continue
                        metric_type = item.metric_type
                        metric_score = item.score
                        if metric_type is not None and metric_score is not None:
                            normalized_score.append({"metric": metric_type, "score": metric_score})
                    if normalized_score:
                        payload["metrics"] = normalized_score
                    else:
                        payload["metrics"] = []
                except Exception as norm_exc:
                    logger.debug("netra.evaluation: Failed to normalize score payload: %s", norm_exc, exc_info=True)
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except Exception as exc:
            logger.error(
                "netra.evaluation: Failed to post status '%s' for run '%s' test '%s': %s",
                status.value,
                run_id,
                test_id,
                exc,
            )

    def post_run_status(self, run_id: str, status: RunStatus) -> None:
        """Post final run status, e.g., {"status": "completed"}. Logs errors on failure."""
        if not self._client:
            logger.error(
                "netra.evaluation: Evaluation client is not initialized; cannot post run status '%s' for run '%s'",
                status.value,
                run_id,
            )
            return
        try:
            url = f"/evaluations/run/{run_id}/status"
            payload: Dict[str, Any] = {"status": status.value}
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except Exception as exc:
            logger.error("netra.evaluation: Failed to post run status '%s' for run '%s': %s", status.value, run_id, exc)
