import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from netra.config import Config
from netra.evaluation.client import EvaluationHttpClient
from netra.evaluation.models import (
    AddDatasetItemResponse,
    CreateDatasetResponse,
    Dataset,
    DatasetItem,
    DatasetRecord,
    EvaluatorConfig,
    GetDatasetItemsResponse,
    ItemContext,
)
from netra.evaluation.utils import (
    build_evaluators_config,
    build_item_payload,
    execute_task,
    extract_dataset_id,
    format_span_id,
    format_trace_id,
    get_session_id_from_baggage,
    run_async_safely,
    run_single_evaluator,
    validate_run_inputs,
)
from netra.span_wrapper import SpanWrapper

logger = logging.getLogger(__name__)


class Evaluation:
    """Public entry-point exposed as Netra.evaluation"""

    def __init__(self, config: Config) -> None:
        """
        Initialize the evaluation client.

        Args:
            config: The configuration object.
        """
        self._config = config
        self._client = EvaluationHttpClient(config)

    def create_dataset(self, name: str, tags: Optional[List[str]] = None) -> Any:
        """
        Create an empty dataset and return its id on success, else None.

        Args:
            name: The name of the dataset.
            tags: Optional list of tags to associate with the dataset.

        Returns:
            A backend JSON response containing dataset info (id, name, tags, etc.) on success,
        """
        if not name:
            logger.error("netra.evaluation: Failed to create dataset: dataset name is required")
            return None
        response = self._client.create_dataset(name=name, tags=tags)

        if not response:
            return None

        return CreateDatasetResponse(
            project_id=response.get("projectId", ""),
            organization_id=response.get("organizationId", ""),
            name=response.get("name", ""),
            tags=response.get("tags", []),
            created_by=response.get("createdBy", ""),
            updated_by=response.get("updatedBy", ""),
            updated_at=response.get("updatedAt", ""),
            id=response.get("id", ""),
            created_at=response.get("createdAt", ""),
            deleted_at=response.get("deletedAt", None),
        )

    def add_dataset_item(
        self,
        dataset_id: str,
        item: DatasetItem,
    ) -> Any:
        """
        Add a single item to an existing dataset

        Args:
            dataset_id: The id of the dataset to which the item will be added.
            item: The dataset item to add.

        Returns:
            A backend JSON response containing dataset item info (id, input, expected_output, etc.) on success
        """

        if not item.input:
            logger.error("netra.evaluation: Skipping dataset item without required 'input'")
            return None
        response = self._client.add_dataset_item(dataset_id=dataset_id, item=item)

        return AddDatasetItemResponse(
            dataset_id=response.get("datasetId", ""),
            project_id=response.get("projectId", ""),
            organization_id=response.get("organizationId", ""),
            source=response.get("source", ""),
            input=response.get("input", ""),
            expected_output=response.get("expectedOutput", ""),
            is_active=True,
            tags=response.get("tags", []),
            created_by=response.get("createdBy", ""),
            updated_by=response.get("updatedBy", ""),
            updated_at=response.get("updatedAt", ""),
            source_id=response.get("sourceId", None),
            metadata=response.get("metadata", None),
            id=response.get("id", ""),
            created_at=response.get("createdAt", ""),
            deleted_at=response.get("deletedAt", None),
        )

    def get_dataset(self, dataset_id: str) -> Any:
        """
        Get a dataset by ID.

        Args:
            dataset_id: The id of the dataset to retrieve.

        Returns:
            A backend JSON response containing dataset info (id, input, expected_output etc.) on success,
        """
        if not dataset_id:
            logger.error("netra.evaluation: Failed to get dataset: dataset id is required")
            return None
        response = self._client.get_dataset(dataset_id)
        if not response:
            return None
        dataset_items: List[DatasetItem] = []
        for item in response:
            item_id = item.get("id")
            item_input = item.get("input")
            item_dataset_id = item.get("datasetId")
            item_expected_output = item.get("expectedOutput", "")
            if item_id is None or item_dataset_id is None or item_input is None:
                logger.warning("netra.evaluation: Skipping dataset item with missing required fields: %s", item)
                continue
            try:
                dataset_items.append(
                    DatasetRecord(
                        id=item_id,
                        input=item_input,
                        dataset_id=item_dataset_id,
                        expected_output=item_expected_output,
                    )
                )
            except Exception as exc:
                logger.error("netra.evaluation: Failed to parse dataset item: %s", exc)
        return GetDatasetItemsResponse(items=dataset_items)

    def create_run(
        self,
        name: str,
        dataset_id: Optional[str] = None,
        evaluators_config: Optional[List[EvaluatorConfig]] = None,
    ) -> Any:
        """
        Create a new run for the given dataset and evaluators

        Args:
            name: The name of the run.
            dataset_id: The id of the dataset to which the run will be associated.
            evaluators_config: Optional list of evaluators to be used for the run.

        Returns:
            run_id: The id of the created run.
        """

        if not name:
            logger.error("netra.evaluation: Failed to create run: run name is required")
            return None

        evaluators_config_dicts: Optional[List[Dict[str, Any]]] = None
        if evaluators_config:
            evaluators_config_dicts = [e.model_dump(by_alias=True) for e in evaluators_config]
        response = self._client.create_run(name=name, dataset_id=dataset_id, evaluators_config=evaluators_config_dicts)
        run_id = response.get("id", None)
        return run_id

    def run_test_suite(
        self,
        name: str,
        data: Dataset,
        task: Callable[[Any], Any],
        evaluators: Optional[List[Any]] = None,
        max_concurrency: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        Netra evaluation function to initiate a test suite.

        Args:
            name: The name of the run.
            data: The dataset to be used for the test suite.
            task: The task to be executed for each item in the dataset.
            evaluators: Optional list of evaluators to be used for the test suite.
            max_concurrency: The maximum number of concurrent tasks to be executed.

        Returns:
            A dictionary containing the run id and the results of the test suite.
        """
        result = run_async_safely(self._run_test_suite_async(name, data, task, evaluators, max_concurrency))
        return result

    async def _run_test_suite_async(
        self,
        name: str,
        data: Dataset,
        task: Callable[[Any], Any],
        evaluators: Optional[List[Any]],
        max_concurrency: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Async implementation of run_test_suite.

        Args:
            name: The name of the run.
            data: The dataset to be used for the test suite.
            task: The task to be executed for each item in the dataset.
            evaluators: Optional list of evaluators to be used for the test suite.
            max_concurrency: The maximum number of concurrent tasks to be executed.

        Returns:
            items: The results of the test suite.
        """
        if not validate_run_inputs(name, data, task):
            return None

        items = list(data.items)
        dataset_id = extract_dataset_id(items)
        evaluators_config = build_evaluators_config(evaluators)

        run_id = self.create_run(
            name=name,
            dataset_id=dataset_id,
            evaluators_config=evaluators_config,
        )
        if not run_id:
            logger.error("netra.evaluation: Failed to create run")
            return None
        logger.info("netra.evaluation: Run created with id: %s", run_id)

        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        results: List[Dict[str, Any]] = []
        bg_eval_tasks: List[asyncio.Task[None]] = []

        async def process_item(idx: int, item: Any) -> None:
            async with semaphore:
                ctx = self._create_item_context(idx, item)
                await self._execute_item_pipeline(
                    run_id=run_id,
                    run_name=name,
                    ctx=ctx,
                    task=task,
                    evaluators=evaluators,
                    results=results,
                    bg_eval_tasks=bg_eval_tasks,
                )

        await asyncio.gather(*[process_item(i, item) for i, item in enumerate(items)])

        if bg_eval_tasks:
            await asyncio.gather(*bg_eval_tasks, return_exceptions=True)

        self._client.post_run_status(run_id, "completed")
        return {"runId": run_id, "items": results}

    def _create_item_context(self, idx: int, item: Any) -> ItemContext:
        """
        Create an ItemContext from a dataset item.

        Args:
            idx: The index of the item.
            item: The dataset item.

        Returns:
            ItemContext: The created ItemContext.
        """
        if isinstance(item, DatasetRecord):
            return ItemContext(
                index=idx,
                item_input=item.input,
                expected_output=item.expected_output,
                dataset_item_id=item.id,
            )
        return ItemContext(
            index=idx,
            item_input=item.input,
            expected_output=getattr(item, "expected_output", None),
            metadata=getattr(item, "metadata", None),
        )

    async def _execute_item_pipeline(
        self,
        run_id: str,
        run_name: str,
        ctx: ItemContext,
        task: Callable[[Any], Any],
        evaluators: Optional[List[Any]],
        results: List[Dict[str, Any]],
        bg_eval_tasks: List[asyncio.Task[None]],
    ) -> None:
        """
        Execute the full pipeline for a single item.

        Args:
            run_id: The run ID.
            run_name: The name of the run.
            ctx: The item context.
            task: The task function to execute.
            evaluators: Optional list of evaluators.
            results: List to append results to.
            bg_eval_tasks: List to append background evaluation tasks to.
        """
        span_name = f"evaluation.{run_name}.{ctx.index}"

        with SpanWrapper(span_name, module_name="netra.evaluation") as span:
            otel_span = span.get_current_span()
            if otel_span:
                span_context = otel_span.get_span_context()
                ctx.trace_id = format_trace_id(span_context.trace_id)
                ctx.span_id = format_span_id(span_context.span_id)
            ctx.session_id = get_session_id_from_baggage()

            ctx.task_output, ctx.status = await execute_task(task, ctx.item_input)
            ctx.test_run_item_id = self._post_completed_status(run_id, ctx)

            if evaluators and ctx.status == "completed":
                eval_task = asyncio.create_task(self._run_evaluators_for_item(run_id, ctx, evaluators))
                bg_eval_tasks.append(eval_task)

            results.append(
                {
                    "index": ctx.index,
                    "status": ctx.status,
                    "traceId": ctx.trace_id,
                    "spanId": ctx.span_id,
                    "testRunItemId": ctx.test_run_item_id,
                }
            )

    def _post_triggered_status(self, run_id: str, ctx: ItemContext) -> str:
        """
        Post agent_triggered status and return test_run_item_id.

        Args:
            run_id: The run ID.
            ctx: The item context.

        Returns:
            str: The test_run_item_id.
        """
        payload = build_item_payload(ctx, status="agent_triggered")
        response = self._client.post_run_item(run_id, payload)

        if isinstance(response, dict):
            item_id = response.get("id") or response.get("testRunItemId")
            if item_id:
                return str(item_id)
        return f"local-{ctx.index}"

    def _post_completed_status(self, run_id: str, ctx: ItemContext) -> Any:
        """
        Post completed/failed status with task output.

        Args:
            run_id: The run ID.
            ctx: The item context.
        """
        payload = build_item_payload(ctx, status=ctx.status, include_output=True)
        run_item_id = self._client.post_run_item(run_id, payload)
        return run_item_id

    async def _run_evaluators_for_item(
        self,
        run_id: str,
        ctx: ItemContext,
        evaluators: List[Any],
    ) -> None:
        """
        Run all evaluators for a single item after span ingestion.

        Args:
            run_id: The run ID.
            ctx: The item context.
            evaluators: List of evaluators.
        """
        await self._client.wait_for_span_ingestion(ctx.span_id)

        evaluator_results: List[Dict[str, Any]] = []
        for evaluator in evaluators:
            try:
                result = await run_single_evaluator(
                    evaluator=evaluator,
                    item_input=ctx.item_input,
                    task_output=ctx.task_output,
                    expected_output=ctx.expected_output,
                    metadata=ctx.metadata,
                )
                if result:
                    evaluator_results.append(result)
            except Exception:
                continue

        if evaluator_results and ctx.test_run_item_id:
            self._client.submit_local_evaluations(run_id, ctx.test_run_item_id, evaluator_results)
