import logging
from datetime import datetime
from typing import List, Optional

from netra.config import Config

from .client import _EvaluationHttpClient
from .context import RunEntryContext
from .models import Dataset, DatasetItem, Run

logger = logging.getLogger(__name__)


class Evaluation:
    """Public entry-point exposed as Netra.evaluation"""

    def __init__(self, cfg: Config) -> None:
        """Initialize the evaluation client."""
        self._config = cfg
        self._client = _EvaluationHttpClient(cfg)

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
                    )
                )
            except Exception as exc:
                logger.error("netra.evaluation: Failed to parse dataset item: %s", exc)
        return Dataset(dataset_id=dataset_id, items=items)

    def create_run(self, dataset: Dataset, name: Optional[str] = None) -> Optional[Run]:
        """Create a new run for the evaluation."""
        run_name = name or f"run-{datetime.now().isoformat()}"
        response = self._client.create_run(dataset_id=dataset.dataset_id, name=run_name)
        run_id = response.get("id")
        if not run_id:
            logger.error("netra.evaluation: Failed to create run for dataset '%s'", dataset.dataset_id)
            return None
        return Run(id=str(run_id), dataset_id=dataset.dataset_id, name=run_name, test_entries=list(dataset.items))

    def run_entry(self, run: Run, entry: DatasetItem) -> RunEntryContext:
        """Start a new run entry."""
        return RunEntryContext(self._client, self._config, run, entry)

    def record(self, ctx: RunEntryContext) -> None:
        """Record completion status for a run entry (no result payload)."""
        try:
            session_id = RunEntryContext._get_session_id_from_baggage()
            self._client.post_entry_status(
                ctx.run.id, ctx.entry.id, status="agent_completed", trace_id=ctx.trace_id, session_id=session_id
            )
        except Exception as exc:
            logger.error("netra.evaluation: Failed to POST agent_completed: %s", exc)
