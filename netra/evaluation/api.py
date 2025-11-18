import asyncio
import inspect
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from netra.config import Config
from netra.evaluation.client import _EvaluationHttpClient
from netra.evaluation.context import RunEntryContext
from netra.evaluation.models import Dataset, DatasetEntry, DatasetItem, EntryStatus, EvaluationScore, Run, RunStatus
from netra.evaluation.utils import get_session_id_from_baggage, run_async_safely

logger = logging.getLogger(__name__)


class Evaluation:
    """Public entry-point exposed as Netra.evaluation"""

    def __init__(self, cfg: Config) -> None:
        """Initialize the evaluation client."""
        self._config = cfg
        self._client = _EvaluationHttpClient(cfg)

    def create_dataset(
        self, name: str, tags: Optional[List[str]] = None, policy_ids: Optional[List[str]] = None
    ) -> Any:
        """Create an empty dataset and return its id on success, else None."""
        if not name:
            logger.error("netra.evaluation: Failed to create dataset via API: name is required")
            return None
        resp = self._client.create_dataset(name=name, tags=tags, policy_ids=policy_ids)
        dataset_id = resp.get("id")
        if not dataset_id:
            logger.error("netra.evaluation: Failed to create dataset")
            return None
        logger.info("netra.evaluation: Created dataset '%s'", name)
        return dataset_id

    def add_dataset_entry(
        self,
        dataset_id: str,
        item: DatasetEntry,
    ) -> Optional[str]:
        """Add a single item to an existing dataset and return the new item id if available."""
        try:
            item_payload: Dict[str, Any] = {}
            if item.input is None:
                logger.warning("netra.evaluation: Skipping dataset item without required 'input': %s", item)
                return None
            item_payload["input"] = item.input
            if item.expected_output is not None:
                item_payload["expectedOutput"] = item.expected_output
            if item.tags is not None:
                item_payload["tags"] = list(item.tags)
            if item.policy_ids is not None:
                item_payload["policyIds"] = list(item.policy_ids)
        except Exception as e:
            logger.error("netra.evaluation: Failed to normalize dataset item for add: %s", e)
            return None

        resp = self._client.add_dataset_entry(dataset_id, item_payload)
        item_id = resp.get("id")
        logger.info("netra.evaluation: Added dataset item '%s' to dataset", item_id)
        return str(item_id) if item_id else None

    def get_dataset(self, dataset_id: str) -> Dataset:
        """Get a dataset by ID."""
        response = self._client.get_dataset(dataset_id)
        items: List[DatasetItem] = []
        for item in response:
            item_id = item.get("id")
            item_input = item.get("input")
            item_dataset_id = item.get("datasetId")
            if item_id is None or item_dataset_id is None or item_input is None:
                logger.warning("netra.evaluation: Skipping dataset item with missing required fields: %s", item)
                continue
            try:
                items.append(
                    DatasetItem(
                        id=item_id,
                        input=item_input,
                        dataset_id=item_dataset_id,
                        expected_output=item.get("expectedOutput"),
                    )
                )
            except Exception as exc:
                logger.error("netra.evaluation: Failed to parse dataset item: %s", exc)
        logger.info("netra.evaluation: Fetched dataset '%s' successfully", dataset_id)
        return Dataset(dataset_id=dataset_id, items=items)

    def create_run(self, dataset: Dataset, name: Optional[str] = None) -> Optional[Run]:
        """Create a new run for the evaluation."""
        run_name = name or f"run-{datetime.now().isoformat()}"
        response = self._client.create_run(dataset_id=dataset.dataset_id, name=run_name)
        run_id = response.get("id")
        if not run_id:
            logger.error("netra.evaluation: Failed to create run for dataset '%s'", dataset.dataset_id)
            return None
        logger.info("netra.evaluation: Created run '%s'", run_name)
        return Run(id=str(run_id), dataset_id=dataset.dataset_id, name=run_name, test_entries=list(dataset.items))

    def run_entry(self, run: Run, entry: DatasetItem) -> RunEntryContext:
        """Start a new run entry."""
        return RunEntryContext(self._client, self._config, run, entry)

    def record(self, ctx: RunEntryContext, score: Optional[List[EvaluationScore]] = None) -> None:
        """Record completion status for a run entry, optionally including score list."""
        try:
            session_id = get_session_id_from_baggage()
            self._client.post_entry_status(
                ctx.run.id,
                ctx.entry.id,
                status=EntryStatus.AGENT_COMPLETED,
                trace_id=ctx.trace_id,
                session_id=session_id,
                score=score,
            )
            logger.info("netra.evaluation: Run entry '%s' for agent completed successfully", ctx.run.id)
        except Exception as exc:
            logger.error("netra.evaluation: Failed to POST agent_completed: %s", exc)

    def run_test_suite(
        self,
        name: str,
        data: Dataset,
        task: Callable[[Any], Any],
        evaluators: Optional[List[Callable[..., Any]]] = None,
        max_concurrency: int = 50,
    ) -> Dict[str, Any]:
        return run_async_safely(
            self._run_test_suite_async(
                name=name, data=data, task=task, evaluators=evaluators, max_concurrency=max_concurrency
            )
        )

    async def _run_test_suite_async(
        self,
        name: str,
        data: Dataset,
        task: Callable[[Any], Any],
        evaluators: Optional[List[Callable[..., Any]]],
        max_concurrency: int,
    ) -> Dict[str, Any]:
        run = self.create_run(data, name)
        if run is None:
            return {"success": False, "error": "failed_to_create_run"}

        semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _run_task(input_value: Any) -> Any:
            result = task(input_value)
            if inspect.isawaitable(result):
                return await result
            return result

        async def _process_entry(entry: DatasetItem) -> Dict[str, Any]:
            async with semaphore:
                output: Any = None
                status: str = "completed"
                error: Optional[str] = None
                with self.run_entry(run, entry) as ctx:
                    try:
                        output = await _run_task(entry.input)
                        collected_scores: List[EvaluationScore] = []

                        # 1) Scores from task output (back-compat)
                        if isinstance(output, dict):
                            scores_from_task = output.get("scores") or output.get("score")

                            def _normalize_scores(obj: Any) -> List[EvaluationScore]:
                                norm: List[EvaluationScore] = []
                                if obj is None:
                                    return norm
                                if isinstance(obj, EvaluationScore):
                                    norm.append(obj)
                                elif isinstance(obj, list):
                                    for it in obj:
                                        if isinstance(it, EvaluationScore):
                                            norm.append(it)
                                        elif isinstance(it, dict):
                                            mt = it.get("metric_type")
                                            sc = it.get("score")
                                            if mt is not None and sc is not None:
                                                try:
                                                    norm.append(EvaluationScore(metric_type=mt, score=float(sc)))
                                                except Exception:
                                                    pass
                                elif isinstance(obj, dict):
                                    mt = obj.get("metric_type")
                                    sc = obj.get("score")
                                    if mt is not None and sc is not None:
                                        try:
                                            norm.append(EvaluationScore(metric_type=mt, score=float(sc)))
                                        except Exception:
                                            pass
                                return norm

                            collected_scores.extend(_normalize_scores(scores_from_task))

                        # 2) Scores from evaluators (sync/async)
                        if evaluators:
                            for evaluator in evaluators:
                                try:
                                    result = evaluator(
                                        input=entry.input,
                                        output=output,
                                        expected_output=entry.expected_output,
                                    )
                                    if inspect.isawaitable(result):
                                        result = await result

                                    # Reuse normalizer
                                    if "_normalize_scores" in locals():
                                        collected_scores.extend(_normalize_scores(result))
                                    else:
                                        # Fallback minimal normalization
                                        if isinstance(result, EvaluationScore):
                                            collected_scores.append(result)
                                        elif isinstance(result, list):
                                            collected_scores.extend(
                                                [r for r in result if isinstance(r, EvaluationScore)]
                                            )
                                except Exception as ev_exc:  # noqa: BLE001
                                    logger.debug("netra.evaluation: evaluator error: %s", ev_exc, exc_info=True)

                        # Record completion with any collected scores
                        self.record(ctx, score=collected_scores if collected_scores else None)
                    except Exception as exc:  # noqa: BLE001
                        status = "failed"
                        error = repr(exc)
                        try:
                            session_id = get_session_id_from_baggage()
                            self._client.post_entry_status(
                                run.id,
                                entry.id,
                                status=EntryStatus.FAILED,
                                trace_id=ctx.trace_id,
                                session_id=session_id,
                            )
                        except Exception as post_exc:  # noqa: BLE001
                            logger.debug(
                                "netra.evaluation: Failed to POST explicit failed status: %s", post_exc, exc_info=True
                            )
                return {
                    "id": entry.id,
                    "input": entry.input,
                    "output": output,
                    "status": status,
                    "error": error,
                }

        tasks = [asyncio.create_task(_process_entry(entry)) for entry in run.test_entries]
        results = await asyncio.gather(*tasks)

        try:
            self._client.post_run_status(run.id, status=RunStatus.COMPLETED)
            logger.info("netra.evaluation: Test suite completed successfully")
        except Exception as exc:  # noqa: BLE001
            logger.error("netra.evaluation: Failed to POST final run status: %s", exc)

        return {"success": True, "run_id": run.id, "results": results}
