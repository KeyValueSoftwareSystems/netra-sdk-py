"""
Unit tests for the netra/evaluation/ module.

Covers models, utils, client, api, and evaluator layers with mocked
HTTP interactions and async helpers.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from netra.evaluation.evaluator import BaseEvaluator
from netra.evaluation.models import (
    AddDatasetItemResponse,
    CreateDatasetResponse,
    Dataset,
    DatasetItem,
    DatasetRecord,
    EvaluatorConfig,
    EvaluatorContext,
    EvaluatorOutput,
    GetDatasetItemsResponse,
    ItemContext,
    ItemProcessingResult,
    LocalDataset,
    ScoreType,
    TurnType,
)
from netra.evaluation.utils import (
    build_evaluators_config,
    build_item_payload,
    execute_task,
    extract_dataset_id,
    format_span_id,
    format_trace_id,
    parse_env_float,
    run_async_safely,
    run_single_evaluator,
    validate_run_inputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PassEvaluator(BaseEvaluator):
    """Evaluator that always passes."""

    def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
        return EvaluatorOutput(
            evaluator_name=self.config.name,
            result=True,
            is_passed=True,
            reason="always passes",
        )


class FailEvaluator(BaseEvaluator):
    """Evaluator that always fails."""

    def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
        return EvaluatorOutput(
            evaluator_name=self.config.name,
            result=False,
            is_passed=False,
            reason="always fails",
        )


class AsyncEvaluator(BaseEvaluator):
    """Async evaluator that always passes."""

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:  # type: ignore[override]
        return EvaluatorOutput(
            evaluator_name=self.config.name,
            result=True,
            is_passed=True,
            reason="async pass",
        )


class CrashingEvaluator(BaseEvaluator):
    """Evaluator that raises an exception."""

    def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
        raise RuntimeError("evaluator exploded")


def _make_evaluator_config(name: str = "test_eval", label: str = "Test Evaluator") -> EvaluatorConfig:
    """Create a test EvaluatorConfig."""
    return EvaluatorConfig(name=name, label=label, scoreType=ScoreType.BOOLEAN)


def _make_config(
    endpoint: str = "https://api.getnetra.ai/telemetry",
    api_key: str = "key-1",
) -> MagicMock:
    """Create a mock Config."""
    cfg = MagicMock()
    cfg.otlp_endpoint = endpoint
    cfg.api_key = api_key
    cfg.headers = {}
    return cfg


# ---------------------------------------------------------------------------
# Section 1: Models
# ---------------------------------------------------------------------------


class TestScoreType:
    """Tests for ScoreType enum."""

    def test_values(self) -> None:
        assert ScoreType.BOOLEAN.value == "boolean"
        assert ScoreType.NUMERICAL.value == "numerical"
        assert ScoreType.CATEGORICAL.value == "categorical"


class TestTurnType:
    """Tests for TurnType enum."""

    def test_values(self) -> None:
        assert TurnType.SINGLE.value == "single"
        assert TurnType.MULTI.value == "multi"


class TestDatasetItem:
    """Tests for DatasetItem model."""

    def test_required_input(self) -> None:
        item = DatasetItem(input="hello")
        assert item.input == "hello"
        assert item.expected_output is None
        assert item.metadata is None
        assert item.tags is None

    def test_all_fields(self) -> None:
        item = DatasetItem(
            input="hello",
            expected_output="world",
            metadata={"key": "val"},
            tags=["tag1"],
        )
        assert item.expected_output == "world"
        assert item.metadata == {"key": "val"}
        assert item.tags == ["tag1"]


class TestDatasetRecord:
    """Tests for DatasetRecord model."""

    def test_creation(self) -> None:
        record = DatasetRecord(id="r1", input="q", dataset_id="ds1")
        assert record.id == "r1"
        assert record.expected_output is None


class TestDataset:
    """Tests for Dataset model."""

    def test_with_dataset_items(self) -> None:
        ds = Dataset(items=[DatasetItem(input="a"), DatasetItem(input="b")])
        assert len(ds.items) == 2

    def test_with_dataset_records(self) -> None:
        ds = Dataset(
            items=[
                DatasetRecord(id="r1", input="a", dataset_id="ds1"),
                DatasetRecord(id="r2", input="b", dataset_id="ds1"),
            ]
        )
        assert len(ds.items) == 2


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig model."""

    def test_alias(self) -> None:
        config = EvaluatorConfig(name="e1", label="Eval 1", scoreType=ScoreType.BOOLEAN)
        assert config.score_type == ScoreType.BOOLEAN

    def test_populate_by_name(self) -> None:
        config = EvaluatorConfig(name="e1", label="Eval 1", scoreType=ScoreType.NUMERICAL)
        assert config.score_type == ScoreType.NUMERICAL


class TestItemContext:
    """Tests for ItemContext dataclass."""

    def test_defaults(self) -> None:
        ctx = ItemContext(index=0, item_input="hello")
        assert ctx.status == "pending"
        assert ctx.trace_id == ""
        assert ctx.task_output is None

    def test_slots(self) -> None:
        ctx = ItemContext(index=0, item_input="hello")
        with pytest.raises(AttributeError):
            ctx.nonexistent = True  # type: ignore[attr-defined]


class TestItemProcessingResult:
    """Tests for ItemProcessingResult dataclass."""

    def test_creation(self) -> None:
        ctx = ItemContext(index=0, item_input="x")
        result = ItemProcessingResult(
            item_entry={"index": 0},
            should_run_evaluators=True,
            ctx=ctx,
            status="completed",
        )
        assert result.should_run_evaluators is True


class TestLocalDataset:
    """Tests for LocalDataset model."""

    def test_creation(self) -> None:
        ld = LocalDataset(items=[DatasetItem(input="x")])
        assert len(ld.items) == 1


class TestCreateDatasetResponse:
    """Tests for CreateDatasetResponse model."""

    def test_creation(self) -> None:
        resp = CreateDatasetResponse(
            project_id="p1",
            organization_id="o1",
            name="ds1",
            created_by="user",
            updated_by="user",
            updated_at="2025-01-01",
            id="id1",
            created_at="2025-01-01",
        )
        assert resp.id == "id1"
        assert resp.deleted_at is None


class TestAddDatasetItemResponse:
    """Tests for AddDatasetItemResponse model."""

    def test_creation(self) -> None:
        resp = AddDatasetItemResponse(
            dataset_id="ds1",
            project_id="p1",
            organization_id="o1",
            source="sdk",
            input="hello",
            is_active=True,
            created_by="user",
            updated_by="user",
            updated_at="2025-01-01",
            id="item1",
            created_at="2025-01-01",
        )
        assert resp.is_active is True


# ---------------------------------------------------------------------------
# Section 2: Utils
# ---------------------------------------------------------------------------


class TestParseEnvFloat:
    """Tests for parse_env_float."""

    def test_returns_default_when_unset(self) -> None:
        assert parse_env_float("_NETRA_EVAL_TEST_NONEXISTENT_", 42.0) == 42.0

    def test_parses_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_NETRA_EVAL_TEST_FLOAT_", "3.14")
        assert parse_env_float("_NETRA_EVAL_TEST_FLOAT_", 1.0) == pytest.approx(3.14)

    def test_returns_default_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_NETRA_EVAL_TEST_FLOAT_", "not-a-number")
        assert parse_env_float("_NETRA_EVAL_TEST_FLOAT_", 7.0) == 7.0


class TestFormatTraceId:
    """Tests for format_trace_id."""

    def test_zero(self) -> None:
        assert format_trace_id(0) == "0" * 32

    def test_known_value(self) -> None:
        result = format_trace_id(255)
        assert result == "0" * 30 + "ff"
        assert len(result) == 32


class TestFormatSpanId:
    """Tests for format_span_id."""

    def test_zero(self) -> None:
        assert format_span_id(0) == "0" * 16

    def test_known_value(self) -> None:
        result = format_span_id(255)
        assert result == "0" * 14 + "ff"
        assert len(result) == 16


class TestRunAsyncSafely:
    """Tests for run_async_safely."""

    def test_runs_coroutine(self) -> None:
        async def coro() -> int:
            return 42

        assert run_async_safely(coro()) == 42

    def test_propagates_exception(self) -> None:
        async def coro() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_async_safely(coro())


class TestExecuteTask:
    """Tests for execute_task."""

    def test_sync_task(self) -> None:
        def task(inp: str) -> str:
            return f"result: {inp}"

        output, status = asyncio.run(execute_task(task, "hello"))
        assert output == "result: hello"
        assert status == "completed"

    def test_async_task(self) -> None:
        async def task(inp: str) -> str:
            return f"async: {inp}"

        output, status = asyncio.run(execute_task(task, "hello"))
        assert output == "async: hello"
        assert status == "completed"

    def test_failed_task(self) -> None:
        def task(inp: str) -> str:
            raise ValueError("task error")

        output, status = asyncio.run(execute_task(task, "hello"))
        assert status == "failed"
        assert "task error" in output


class TestValidateRunInputs:
    """Tests for validate_run_inputs."""

    def test_valid(self) -> None:
        validate_run_inputs("name", Dataset(items=[DatasetItem(input="x")]), lambda x: x)

    def test_missing_name(self) -> None:
        with pytest.raises(ValueError, match="run name is required"):
            validate_run_inputs("", Dataset(items=[DatasetItem(input="x")]), lambda x: x)

    def test_missing_task(self) -> None:
        with pytest.raises(ValueError, match="task function is required"):
            validate_run_inputs("name", Dataset(items=[DatasetItem(input="x")]), None)  # type: ignore[arg-type]


class TestExtractDatasetId:
    """Tests for extract_dataset_id."""

    def test_with_dataset_records(self) -> None:
        items = [DatasetRecord(id="r1", input="a", dataset_id="ds1")]
        assert extract_dataset_id(items) == "ds1"

    def test_with_dataset_items(self) -> None:
        items = [DatasetItem(input="a")]
        assert extract_dataset_id(items) is None

    def test_empty_list(self) -> None:
        assert extract_dataset_id([]) is None


class TestBuildEvaluatorsConfig:
    """Tests for build_evaluators_config."""

    def test_none_evaluators(self) -> None:
        assert build_evaluators_config(None) == []

    def test_empty_evaluators(self) -> None:
        assert build_evaluators_config([]) == []

    def test_extracts_configs(self) -> None:
        config = _make_evaluator_config()
        evaluator = PassEvaluator(config)
        result = build_evaluators_config([evaluator])
        assert len(result) == 1
        assert result[0].name == "test_eval"

    def test_skips_evaluators_without_config(self) -> None:
        no_config = MagicMock(spec=[])
        result = build_evaluators_config([no_config])
        assert result == []


class TestBuildItemPayload:
    """Tests for build_item_payload."""

    def test_completed_with_output(self) -> None:
        ctx = ItemContext(
            index=0,
            item_input="hello",
            expected_output="world",
            trace_id="trace-1",
            session_id="session-1",
            task_output="result",
            status="completed",
        )
        payload = build_item_payload(ctx, status="completed", include_output=True)
        assert payload["traceId"] == "trace-1"
        assert payload["taskOutput"] == "result"
        assert "status" not in payload

    def test_failed_status(self) -> None:
        ctx = ItemContext(
            index=0,
            item_input="hello",
            trace_id="trace-1",
            status="failed",
        )
        payload = build_item_payload(ctx, status="failed")
        assert payload["status"] == "failed"
        assert "taskOutput" not in payload

    def test_uses_passed_status_not_ctx_status(self) -> None:
        ctx = ItemContext(
            index=0,
            item_input="hello",
            trace_id="trace-1",
            status="completed",
            task_output="result",
        )
        payload = build_item_payload(ctx, status="failed")
        assert payload["status"] == "failed"

    def test_with_dataset_item_id(self) -> None:
        ctx = ItemContext(
            index=0,
            item_input="hello",
            dataset_item_id="item-1",
            trace_id="trace-1",
        )
        payload = build_item_payload(ctx, status="completed")
        assert payload["datasetItemId"] == "item-1"
        assert "input" not in payload

    def test_without_dataset_item_id(self) -> None:
        ctx = ItemContext(
            index=0,
            item_input="hello",
            expected_output="world",
            metadata={"key": "val"},
            trace_id="trace-1",
        )
        payload = build_item_payload(ctx, status="completed")
        assert payload["input"] == "hello"
        assert payload["expectedOutput"] == "world"
        assert payload["metadata"] == {"key": "val"}


class TestRunSingleEvaluator:
    """Tests for run_single_evaluator."""

    def test_sync_evaluator(self) -> None:
        config = _make_evaluator_config()
        evaluator = PassEvaluator(config)
        result = asyncio.run(
            run_single_evaluator(
                evaluator=evaluator,
                item_input="hello",
                task_output="world",
                expected_output="world",
                metadata=None,
            )
        )
        assert result is not None
        assert result["evaluatorName"] == "test_eval"
        assert result["isPassed"] is True

    def test_async_evaluator(self) -> None:
        config = _make_evaluator_config()
        evaluator = AsyncEvaluator(config)
        result = asyncio.run(
            run_single_evaluator(
                evaluator=evaluator,
                item_input="hello",
                task_output="world",
                expected_output="world",
                metadata=None,
            )
        )
        assert result is not None
        assert result["isPassed"] is True

    def test_evaluator_without_evaluate(self) -> None:
        evaluator = MagicMock(spec=[])
        result = asyncio.run(
            run_single_evaluator(
                evaluator=evaluator,
                item_input="hello",
                task_output="world",
                expected_output="world",
                metadata=None,
            )
        )
        assert result is None

    def test_name_mismatch_returns_none(self) -> None:
        config = _make_evaluator_config(name="expected_name")

        class WrongNameEvaluator(BaseEvaluator):
            def evaluate(self, context: EvaluatorContext) -> EvaluatorOutput:
                return EvaluatorOutput(
                    evaluator_name="wrong_name",
                    result=True,
                    is_passed=True,
                )

        evaluator = WrongNameEvaluator(config)
        result = asyncio.run(
            run_single_evaluator(
                evaluator=evaluator,
                item_input="hello",
                task_output="world",
                expected_output="world",
                metadata=None,
            )
        )
        assert result is None


# ---------------------------------------------------------------------------
# Section 3: Client
# ---------------------------------------------------------------------------


class TestEvaluationHttpClient:
    """Tests for EvaluationHttpClient."""

    def test_create_client_with_valid_config(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config())
        assert client._client is not None
        client.close()

    def test_create_client_strips_telemetry_suffix(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint="https://api.getnetra.ai/telemetry"))
        assert client._client is not None
        assert "/telemetry" not in str(client._client.base_url)
        client.close()

    def test_create_client_returns_none_on_empty_endpoint(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        assert client._client is None

    def test_close_sets_client_to_none(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config())
        assert client._client is not None
        client.close()
        assert client._client is None

    def test_close_idempotent(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config())
        client.close()
        client.close()
        assert client._client is None

    def test_ensure_client_returns_none_when_not_initialized(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        assert client._ensure_client() is None

    def test_extract_error_message_from_response(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": {"message": "custom error"}}
        result = client._extract_error_message(mock_response, ValueError("fallback"))
        assert result == "custom error"

    def test_extract_error_message_fallback_on_none_response(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        result = client._extract_error_message(None, ValueError("fallback"))
        assert result == "fallback"

    def test_extract_error_message_fallback_on_missing_error_key(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"other": "data"}
        result = client._extract_error_message(mock_response, ValueError("fallback"))
        assert result == "fallback"

    def test_extract_error_message_fallback_on_json_parse_error(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError("not json")
        result = client._extract_error_message(mock_response, RuntimeError("orig"))
        assert result == "orig"

    @patch("netra.evaluation.client.httpx.Client")
    def test_create_dataset_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"id": "ds-1", "name": "test"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.create_dataset(name="test")
        assert result is not None
        assert result["id"] == "ds-1"

    @patch("netra.evaluation.client.httpx.Client")
    def test_create_dataset_returns_none_on_error(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_instance = MagicMock()
        mock_instance.post.side_effect = httpx.ConnectError("timeout")
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.create_dataset(name="test")
        assert result is None

    def test_create_dataset_returns_none_without_client(self) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        client = EvaluationHttpClient(_make_config(endpoint=""))
        result = client.create_dataset(name="test")
        assert result is None

    @patch("netra.evaluation.client.httpx.Client")
    def test_create_run_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"id": "run-1"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.create_run(name="test")
        assert result is not None
        assert result["id"] == "run-1"

    @patch("netra.evaluation.client.httpx.Client")
    def test_post_run_item_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"item": {"id": "item-1"}}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.post_run_item("run-1", {"traceId": "t1"})
        assert result == "item-1"

    @patch("netra.evaluation.client.httpx.Client")
    def test_post_run_item_returns_none_on_error(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_instance = MagicMock()
        mock_instance.post.side_effect = httpx.ConnectError("timeout")
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.post_run_item("run-1", {"traceId": "t1"})
        assert result is None

    @patch("netra.evaluation.client.httpx.Client")
    def test_get_dataset_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": [{"id": "item-1", "input": "q", "datasetId": "ds-1"}]}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.get.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.get_dataset("ds-1")
        assert result is not None
        assert len(result) == 1

    @patch("netra.evaluation.client.httpx.Client")
    def test_post_run_status_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"status": "completed"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.post_run_status("run-1", "completed")
        assert result == {"status": "completed"}

    @patch("netra.evaluation.client.httpx.Client")
    def test_post_run_status_returns_none_on_error(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_instance = MagicMock()
        mock_instance.post.side_effect = httpx.ConnectError("timeout")
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.post_run_status("run-1", "completed")
        assert result is None

    @patch("netra.evaluation.client.httpx.Client")
    def test_get_run_results_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"items": []}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.get.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.get_run_results("run-1")
        assert result is not None

    @patch("netra.evaluation.client.httpx.Client")
    def test_submit_local_evaluations_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"success": True}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.submit_local_evaluations("run-1", "item-1", [{"evaluatorName": "e1"}])
        assert result is not None

    @patch("netra.evaluation.client.httpx.Client")
    def test_add_dataset_item_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"id": "item-1", "input": "q"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        item = DatasetItem(input="hello", expected_output="world")
        result = client.add_dataset_item("ds-1", item)
        assert result is not None
        assert result["id"] == "item-1"

    @patch("netra.evaluation.client.httpx.Client")
    def test_get_span_by_id_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"spanId": "span-1"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.get.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.get_span_by_id("span-1")
        assert result is not None

    @patch("netra.evaluation.client.httpx.Client")
    def test_get_span_by_id_returns_none_on_error(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.client import EvaluationHttpClient

        mock_instance = MagicMock()
        mock_instance.get.side_effect = httpx.ConnectError("timeout")
        mock_client_cls.return_value = mock_instance

        client = EvaluationHttpClient(_make_config())
        result = client.get_span_by_id("span-1")
        assert result is None


# ---------------------------------------------------------------------------
# Section 4: API (Evaluation class)
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for the Evaluation public API."""

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_dataset_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.create_dataset.return_value = {
            "projectId": "p1",
            "organizationId": "o1",
            "name": "ds1",
            "tags": [],
            "createdBy": "user",
            "updatedBy": "user",
            "updatedAt": "2025-01-01",
            "id": "ds-1",
            "createdAt": "2025-01-01",
        }
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.create_dataset(name="ds1")
        assert isinstance(result, CreateDatasetResponse)
        assert result.id == "ds-1"

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_dataset_returns_none_on_empty_name(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        evaluation = Evaluation(_make_config())
        result = evaluation.create_dataset(name="")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_dataset_returns_none_on_client_failure(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.create_dataset.return_value = None
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.create_dataset(name="ds1")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_add_dataset_item_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.add_dataset_item.return_value = {
            "datasetId": "ds-1",
            "projectId": "p1",
            "organizationId": "o1",
            "source": "sdk",
            "input": "hello",
            "createdBy": "user",
            "updatedBy": "user",
            "updatedAt": "2025-01-01",
            "id": "item-1",
            "createdAt": "2025-01-01",
        }
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.add_dataset_item("ds-1", DatasetItem(input="hello"))
        assert isinstance(result, AddDatasetItemResponse)
        assert result.id == "item-1"

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_add_dataset_item_returns_none_on_empty_input(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        evaluation = Evaluation(_make_config())
        result = evaluation.add_dataset_item("ds-1", DatasetItem(input=""))
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_add_dataset_item_returns_none_on_client_failure(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.add_dataset_item.return_value = None
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.add_dataset_item("ds-1", DatasetItem(input="hello"))
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_get_dataset_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.get_dataset.return_value = [
            {"id": "item-1", "input": "q", "datasetId": "ds-1", "expectedOutput": "a"}
        ]
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.get_dataset("ds-1")
        assert isinstance(result, GetDatasetItemsResponse)
        assert len(result.items) == 1

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_get_dataset_returns_none_on_empty_id(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        evaluation = Evaluation(_make_config())
        result = evaluation.get_dataset("")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_get_dataset_skips_invalid_items(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.get_dataset.return_value = [
            {"id": "item-1", "input": "q", "datasetId": "ds-1"},
            {"id": None, "input": "q", "datasetId": "ds-1"},
        ]
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.get_dataset("ds-1")
        assert isinstance(result, GetDatasetItemsResponse)
        assert len(result.items) == 1

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_run_success(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.create_run.return_value = {"id": "run-1"}
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.create_run(name="test")
        assert result == "run-1"

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_run_returns_none_on_empty_name(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        evaluation = Evaluation(_make_config())
        result = evaluation.create_run(name="")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_create_run_returns_none_on_client_failure(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.create_run.return_value = None
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.create_run(name="test")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_get_run_results(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.get_run_results.return_value = {"items": []}
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        result = evaluation.get_run_results("run-1")
        assert result == {"items": []}

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_get_run_results_returns_none_on_empty_id(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        evaluation = Evaluation(_make_config())
        result = evaluation.get_run_results("")
        assert result is None

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_close_delegates_to_client(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        evaluation.close()
        mock_client.close.assert_called_once()

    @patch("netra.evaluation.api.SpanWrapper")
    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_run_test_suite_success(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        mock_client = MagicMock()
        mock_client.create_run.return_value = {"id": "run-1"}
        mock_client.post_run_item.return_value = "item-1"
        mock_client.post_run_status.return_value = None

        async def mock_wait(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_client.wait_for_span_ingestion = mock_wait
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        dataset = Dataset(items=[DatasetItem(input="hello")])
        result = evaluation.run_test_suite(
            name="test",
            data=dataset,
            task=lambda x: f"result: {x}",
        )

        assert result is not None
        assert result["runId"] == "run-1"
        assert len(result["items"]) == 1

    @patch("netra.evaluation.api.SpanWrapper")
    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_run_test_suite_marks_failed_on_exception(
        self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock
    ) -> None:
        from netra.evaluation.api import Evaluation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        mock_client = MagicMock()
        mock_client.create_run.return_value = {"id": "run-1"}
        mock_client.post_run_item.side_effect = RuntimeError("backend down")
        mock_client.post_run_status.return_value = None
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        dataset = Dataset(items=[DatasetItem(input="hello")])

        with pytest.raises(RuntimeError, match="backend down"):
            evaluation.run_test_suite(
                name="test",
                data=dataset,
                task=lambda x: f"result: {x}",
            )

        mock_client.post_run_status.assert_called_with("run-1", "failed")

    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_run_test_suite_returns_none_when_create_run_fails(self, mock_client_cls: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_client = MagicMock()
        mock_client.create_run.return_value = None
        mock_client_cls.return_value = mock_client

        evaluation = Evaluation(_make_config())
        dataset = Dataset(items=[DatasetItem(input="hello")])
        result = evaluation.run_test_suite(
            name="test",
            data=dataset,
            task=lambda x: x,
        )
        assert result is None

    @patch("netra.evaluation.api.SpanWrapper")
    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_run_test_suite_with_evaluators(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_otel_span = MagicMock()
        mock_otel_span.get_span_context.return_value = MagicMock(trace_id=123, span_id=456)
        mock_span.get_current_span.return_value = mock_otel_span
        mock_span_wrapper.return_value = mock_span

        async def mock_wait(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_client = MagicMock()
        mock_client.create_run.return_value = {"id": "run-1"}
        mock_client.post_run_item.return_value = "item-1"
        mock_client.post_run_status.return_value = None
        mock_client.wait_for_span_ingestion = mock_wait
        mock_client.submit_local_evaluations.return_value = None
        mock_client_cls.return_value = mock_client

        config = _make_evaluator_config()
        evaluator = PassEvaluator(config)

        evaluation = Evaluation(_make_config())
        dataset = Dataset(items=[DatasetItem(input="hello")])
        result = evaluation.run_test_suite(
            name="test",
            data=dataset,
            task=lambda x: f"result: {x}",
            evaluators=[evaluator],
        )

        assert result is not None
        assert result["runId"] == "run-1"
        mock_client.submit_local_evaluations.assert_called_once()

    @patch("netra.evaluation.api.SpanWrapper")
    @patch("netra.evaluation.api.EvaluationHttpClient")
    def test_run_test_suite_handles_failed_task(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.evaluation.api import Evaluation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        mock_client = MagicMock()
        mock_client.create_run.return_value = {"id": "run-1"}
        mock_client.post_run_item.return_value = "item-1"
        mock_client.post_run_status.return_value = None
        mock_client_cls.return_value = mock_client

        def failing_task(inp: Any) -> None:
            raise ValueError("task failed")

        evaluation = Evaluation(_make_config())
        dataset = Dataset(items=[DatasetItem(input="hello")])
        result = evaluation.run_test_suite(
            name="test",
            data=dataset,
            task=failing_task,
        )

        assert result is not None
        assert result["items"][0]["status"] == "failed"


# ---------------------------------------------------------------------------
# Section 5: Evaluator
# ---------------------------------------------------------------------------


class TestBaseEvaluator:
    """Tests for BaseEvaluator abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEvaluator(_make_evaluator_config())  # type: ignore[abstract]

    def test_sync_subclass(self) -> None:
        config = _make_evaluator_config()
        evaluator = PassEvaluator(config)
        context = EvaluatorContext(input="x", task_output="y")
        result = evaluator.evaluate(context)
        assert isinstance(result, EvaluatorOutput)
        assert result.is_passed is True

    def test_fail_evaluator(self) -> None:
        config = _make_evaluator_config()
        evaluator = FailEvaluator(config)
        context = EvaluatorContext(input="x", task_output="y")
        result = evaluator.evaluate(context)
        assert result.is_passed is False

    def test_config_accessible(self) -> None:
        config = _make_evaluator_config(name="my_eval")
        evaluator = PassEvaluator(config)
        assert evaluator.config.name == "my_eval"
