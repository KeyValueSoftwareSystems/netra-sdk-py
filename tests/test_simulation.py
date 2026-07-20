"""
Unit tests for the netra/simulation/ module.

Covers models, utils, client, api, and task layers with mocked
HTTP interactions and async helpers.
"""

import asyncio
import base64
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from netra.simulation.models import (
    ConversationResponse,
    ConversationStatus,
    FileData,
    ProcessedFile,
    SimulationItem,
    TaskResult,
)
from netra.simulation.task import BaseTask
from netra.simulation.utils import (
    execute_task,
    format_trace_id,
    parse_env_float,
    process_files,
    run_async_safely,
    validate_simulation_inputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SyncTask(BaseTask):
    """Synchronous task that echoes the message."""

    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[ProcessedFile]] = None,
    ) -> TaskResult:
        return TaskResult(message=f"echo: {message}", session_id=session_id or "sid-1")


class AsyncTask(BaseTask):
    """Asynchronous task that echoes the message."""

    async def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[ProcessedFile]] = None,
    ) -> TaskResult:
        return TaskResult(message=f"async-echo: {message}", session_id=session_id or "sid-async")


class FileAwareTask(BaseTask):
    """Task that uses the files parameter."""

    def run(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[ProcessedFile]] = None,
    ) -> TaskResult:
        count = len(files) if files else 0
        return TaskResult(message=f"files={count}", session_id=session_id or "sid-files")


# ---------------------------------------------------------------------------
# Section 1: Models
# ---------------------------------------------------------------------------


class TestConversationStatus:
    """Tests for ConversationStatus enum."""

    def test_continue_value(self) -> None:
        assert ConversationStatus.CONTINUE.value == "continue"

    def test_stop_value(self) -> None:
        assert ConversationStatus.STOP.value == "stop"

    def test_from_string(self) -> None:
        assert ConversationStatus("continue") == ConversationStatus.CONTINUE
        assert ConversationStatus("stop") == ConversationStatus.STOP


class TestFileData:
    """Tests for the FileData frozen dataclass."""

    def test_creation(self) -> None:
        fd = FileData(file_name="a.txt", content_type="text/plain", description="desc", download_url="https://x")
        assert fd.file_name == "a.txt"
        assert fd.content_type == "text/plain"
        assert fd.description == "desc"
        assert fd.download_url == "https://x"

    def test_frozen(self) -> None:
        fd = FileData(file_name="a.txt", content_type="text/plain", description=None, download_url="https://x")
        with pytest.raises(AttributeError):
            fd.file_name = "b.txt"  # type: ignore[misc]


class TestProcessedFile:
    """Tests for the ProcessedFile frozen dataclass."""

    def test_creation(self) -> None:
        pf = ProcessedFile(file_name="a.txt", content_type="text/plain", description=None, data="AAAA")
        assert pf.data == "AAAA"

    def test_frozen(self) -> None:
        pf = ProcessedFile(file_name="a.txt", content_type="text/plain", description=None, data="AAAA")
        with pytest.raises(AttributeError):
            pf.data = "BBBB"  # type: ignore[misc]


class TestSimulationItem:
    """Tests for the SimulationItem frozen dataclass."""

    def test_defaults(self) -> None:
        item = SimulationItem(run_item_id="r1", dataset_item_id="d1", message="hi", turn_id="t1")
        assert item.files == []

    def test_with_files(self) -> None:
        fd = FileData(file_name="a.txt", content_type="text/plain", description=None, download_url="https://x")
        item = SimulationItem(run_item_id="r1", dataset_item_id="d1", message="hi", turn_id="t1", files=[fd])
        assert len(item.files) == 1


class TestConversationResponse:
    """Tests for the ConversationResponse dataclass."""

    def test_stop_decision(self) -> None:
        resp = ConversationResponse(decision=ConversationStatus.STOP, reason="done")
        assert resp.decision == ConversationStatus.STOP
        assert resp.reason == "done"

    def test_continue_decision_defaults(self) -> None:
        resp = ConversationResponse(decision=ConversationStatus.CONTINUE)
        assert resp.next_turn_id is None
        assert resp.next_user_message is None
        assert resp.next_files == []


class TestTaskResult:
    """Tests for the TaskResult frozen dataclass."""

    def test_creation(self) -> None:
        tr = TaskResult(message="hello", session_id="s1")
        assert tr.message == "hello"
        assert tr.session_id == "s1"

    def test_frozen(self) -> None:
        tr = TaskResult(message="hello", session_id="s1")
        with pytest.raises(AttributeError):
            tr.message = "bye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 2: Utils
# ---------------------------------------------------------------------------


class TestParseEnvFloat:
    """Tests for parse_env_float."""

    def test_returns_default_when_unset(self) -> None:
        assert parse_env_float("_NETRA_TEST_NONEXISTENT_VAR_", 42.0) == 42.0

    def test_parses_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_NETRA_TEST_FLOAT_", "3.14")
        assert parse_env_float("_NETRA_TEST_FLOAT_", 1.0) == pytest.approx(3.14)

    def test_returns_default_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("_NETRA_TEST_FLOAT_", "not-a-number")
        assert parse_env_float("_NETRA_TEST_FLOAT_", 7.0) == 7.0


class TestFormatTraceId:
    """Tests for format_trace_id."""

    def test_zero(self) -> None:
        assert format_trace_id(0) == "0" * 32

    def test_known_value(self) -> None:
        result = format_trace_id(255)
        assert result == "0" * 30 + "ff"
        assert len(result) == 32


class TestValidateSimulationInputs:
    """Tests for validate_simulation_inputs."""

    def test_valid(self) -> None:
        assert validate_simulation_inputs("ds-1", SyncTask()) is True

    def test_empty_dataset_id(self) -> None:
        assert validate_simulation_inputs("", SyncTask()) is False

    def test_wrong_task_type(self) -> None:
        assert validate_simulation_inputs("ds-1", "not a task") is False  # type: ignore[arg-type]


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


class TestProcessFiles:
    """Tests for process_files."""

    def test_empty_list(self) -> None:
        assert process_files([]) == []

    @patch("netra.simulation.utils.httpx.get")
    def test_downloads_and_encodes(self, mock_get: MagicMock) -> None:
        raw_content = b"hello world"
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = raw_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        fd = FileData(file_name="a.txt", content_type="text/plain", description=None, download_url="https://x/a.txt")
        result = process_files([fd])

        assert len(result) == 1
        assert result[0].file_name == "a.txt"
        assert result[0].data == base64.b64encode(raw_content).decode("ascii")

    @patch("netra.simulation.utils.httpx.get")
    def test_raises_on_download_failure(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("connection refused")

        fd = FileData(file_name="a.txt", content_type="text/plain", description=None, download_url="https://x/a.txt")
        with pytest.raises(RuntimeError, match="Failed to download file 'a.txt'"):
            process_files([fd])

    @patch("netra.simulation.utils.httpx.get")
    def test_concurrent_downloads(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        files = [
            FileData(file_name=f"f{i}.txt", content_type="text/plain", description=None, download_url=f"https://x/{i}")
            for i in range(3)
        ]
        result = process_files(files)
        assert len(result) == 3
        assert mock_get.call_count == 3


class TestExecuteTaskFiles:
    """Tests for file handling in execute_task."""

    @patch("netra.simulation.utils.process_files", return_value=[])
    def test_files_downloaded_when_raw_files_present(self, mock_pf: MagicMock) -> None:
        """Files are always downloaded and passed to the task."""
        fd = FileData(file_name="a.txt", content_type="text/plain", description=None, download_url="https://x/a.txt")
        result = asyncio.run(execute_task(FileAwareTask(), "hi", None, raw_files=[fd]))
        mock_pf.assert_called_once_with([fd])
        assert result[0] == "files=0"

    def test_no_files_passed_as_none(self) -> None:
        """When no raw_files are provided, files=None is passed to the task."""
        msg, sid = asyncio.run(execute_task(SyncTask(), "hi", None, raw_files=None))
        assert msg == "echo: hi"

    @patch("netra.simulation.utils.process_files")
    def test_empty_raw_files_skips_download(self, mock_pf: MagicMock) -> None:
        """An empty raw_files list should not trigger downloads."""
        asyncio.run(execute_task(SyncTask(), "hi", None, raw_files=[]))
        mock_pf.assert_not_called()


class TestExecuteTask:
    """Tests for execute_task."""

    def test_sync_task(self) -> None:
        msg, sid = asyncio.run(execute_task(SyncTask(), "hello", None))
        assert msg == "echo: hello"
        assert sid == "sid-1"

    def test_async_task(self) -> None:
        msg, sid = asyncio.run(execute_task(AsyncTask(), "hello", None))
        assert msg == "async-echo: hello"
        assert sid == "sid-async"

    def test_raises_on_bad_return_type(self) -> None:
        class BadTask(BaseTask):
            def run(
                self,
                message: str,
                session_id: Optional[str] = None,
                files: Optional[list[ProcessedFile]] = None,
            ) -> Any:
                return "not a TaskResult"

        with pytest.raises(ValueError, match="Task must return TaskResult"):
            asyncio.run(execute_task(BadTask(), "x", None))


# ---------------------------------------------------------------------------
# Section 3: Client
# ---------------------------------------------------------------------------


class TestSimulationHttpClient:
    """Tests for SimulationHttpClient."""

    def _make_config(self, endpoint: str = "https://api.getnetra.ai/telemetry", api_key: str = "key-1") -> MagicMock:
        """Create a mock Config."""
        cfg = MagicMock()
        cfg.otlp_endpoint = endpoint
        cfg.api_key = api_key
        cfg.headers = {}
        return cfg

    def test_create_client_with_valid_config(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config())
        assert client._client is not None
        client.close()

    def test_create_client_strips_telemetry_suffix(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config(endpoint="https://api.getnetra.ai/telemetry"))
        assert client._client is not None
        assert "/telemetry" not in str(client._client.base_url)
        client.close()

    def test_create_client_returns_none_on_empty_endpoint(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config(endpoint=""))
        assert client._client is None

    def test_close_sets_client_to_none(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config())
        assert client._client is not None
        client.close()
        assert client._client is None

    def test_close_idempotent(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config())
        client.close()
        client.close()
        assert client._client is None

    def test_ensure_client_returns_none_when_not_initialized(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config(endpoint=""))
        assert client._ensure_client() is None

    @patch("netra.simulation.client.httpx.Client")
    def test_trigger_conversation_stop(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.client import SimulationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "decision": "stop",
                "reason": "all done",
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = SimulationHttpClient(self._make_config())
        resp = client.trigger_conversation(message="hi", turn_id="t1", session_id="s1", trace_id="trace")

        assert resp is not None
        assert resp.decision == ConversationStatus.STOP
        assert resp.reason == "all done"

    @patch("netra.simulation.client.httpx.Client")
    def test_trigger_conversation_continue(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.client import SimulationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "data": {
                "decision": "continue",
                "userMessages": [
                    {
                        "turnId": "turn-2",
                        "userMessage": "follow-up",
                        "testRunItemId": "item-2",
                        "attachments": None,
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = SimulationHttpClient(self._make_config())
        resp = client.trigger_conversation(message="hi", turn_id="t1", session_id="s1", trace_id="trace")

        assert resp is not None
        assert resp.decision == ConversationStatus.CONTINUE
        assert resp.next_turn_id == "turn-2"
        assert resp.next_user_message == "follow-up"

    def test_trigger_conversation_returns_none_without_client(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        client = SimulationHttpClient(self._make_config(endpoint=""))
        resp = client.trigger_conversation(message="hi", turn_id="t1", session_id="s1", trace_id="trace")
        assert resp is None

    @patch("netra.simulation.client.httpx.Client")
    def test_report_failure(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.client import SimulationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.patch.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = SimulationHttpClient(self._make_config())
        client.report_failure(run_id="run-1", run_item_id="item-1", error="boom")
        mock_instance.patch.assert_called_once()

    @patch("netra.simulation.client.httpx.Client")
    def test_post_run_status_success(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.client import SimulationHttpClient

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"data": {"status": "completed"}}
        mock_response.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.post.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        client = SimulationHttpClient(self._make_config())
        result = client.post_run_status(run_id="run-1", status="completed")
        assert result == {"status": "completed"}

    @patch("netra.simulation.client.httpx.Client")
    def test_post_run_status_returns_error_on_failure(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.client import SimulationHttpClient

        mock_instance = MagicMock()
        mock_instance.post.side_effect = httpx.ConnectError("timeout")
        mock_client_cls.return_value = mock_instance

        client = SimulationHttpClient(self._make_config())
        result = client.post_run_status(run_id="run-1", status="completed")
        assert result == {"success": False}

    def test_parse_files_none(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        assert SimulationHttpClient._parse_files(None) == []

    def test_parse_files_valid(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        raw = [{"fileName": "a.txt", "downloadUrl": "https://x/a", "contentType": "text/plain"}]
        result = SimulationHttpClient._parse_files(raw)
        assert len(result) == 1
        assert result[0].file_name == "a.txt"

    def test_parse_files_skips_malformed(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        raw = [{"fileName": "", "downloadUrl": "https://x/a"}]
        result = SimulationHttpClient._parse_files(raw)
        assert result == []

    def test_extract_error_message_from_response(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        cfg = MagicMock()
        cfg.otlp_endpoint = ""
        cfg.api_key = ""
        cfg.headers = {}
        client = SimulationHttpClient(cfg)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"error": {"message": "custom error"}}
        result = client._extract_error_message(mock_response, ValueError("fallback"))
        assert result == "custom error"

    def test_extract_error_message_fallback(self) -> None:
        from netra.simulation.client import SimulationHttpClient

        cfg = MagicMock()
        cfg.otlp_endpoint = ""
        cfg.api_key = ""
        cfg.headers = {}
        client = SimulationHttpClient(cfg)

        result = client._extract_error_message(None, ValueError("fallback"))
        assert result == "fallback"


# ---------------------------------------------------------------------------
# Section 4: API (Simulation class)
# ---------------------------------------------------------------------------


class TestSimulation:
    """Tests for the Simulation public API."""

    def _make_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.otlp_endpoint = "https://api.getnetra.ai/telemetry"
        cfg.api_key = "key-1"
        cfg.headers = {}
        return cfg

    @patch("netra.simulation.api.SimulationHttpClient")
    def test_run_simulation_returns_none_on_invalid_inputs(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.api import Simulation

        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="", task=SyncTask())
        assert result is None

    @patch("netra.simulation.api.SimulationHttpClient")
    def test_run_simulation_returns_none_when_initialize_run_fails(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.api import Simulation

        mock_client_cls.return_value.initialize_run.return_value = None
        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="ds-1", task=SyncTask())
        assert result is None

    @patch("netra.simulation.api.SpanWrapper")
    @patch("netra.simulation.api.SimulationHttpClient")
    def test_run_simulation_success(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.simulation.api import Simulation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        stop_response = ConversationResponse(
            decision=ConversationStatus.STOP,
            reason="done",
        )

        mock_client = MagicMock()
        mock_client.initialize_run.return_value = {
            "run_id": "run-1",
            "items": [
                {"test_run_item_id": "item-1", "dataset_item_id": "ds-item-1"},
            ],
        }
        mock_client.generate_first_turn.return_value = SimulationItem(
            run_item_id="item-1",
            dataset_item_id="ds-item-1",
            message="hello",
            turn_id="turn-1",
        )
        mock_client.trigger_conversation.return_value = stop_response
        mock_client.post_run_status.return_value = {"status": "completed"}
        mock_client_cls.return_value = mock_client

        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="ds-1", task=SyncTask())

        assert result is not None
        assert result["total_items"] == 1
        assert len(result["completed"]) == 1
        assert len(result["failed"]) == 0

    @patch("netra.simulation.api.SpanWrapper")
    @patch("netra.simulation.api.SimulationHttpClient")
    def test_run_simulation_marks_failed_on_exception(
        self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock
    ) -> None:
        from netra.simulation.api import Simulation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        mock_client = MagicMock()
        mock_client.initialize_run.return_value = {
            "run_id": "run-1",
            "items": [
                {"test_run_item_id": "item-1", "dataset_item_id": "ds-item-1"},
            ],
        }
        mock_client.generate_first_turn.return_value = SimulationItem(
            run_item_id="item-1",
            dataset_item_id="ds-item-1",
            message="hello",
            turn_id="turn-1",
        )
        mock_client.trigger_conversation.side_effect = RuntimeError("backend down")
        mock_client.post_run_status.return_value = {}
        mock_client_cls.return_value = mock_client

        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="ds-1", task=SyncTask())

        assert result is not None
        assert len(result["failed"]) == 1
        assert result["failed"][0]["error"] == "backend down"

    @patch("netra.simulation.api.SpanWrapper")
    @patch("netra.simulation.api.SimulationHttpClient")
    def test_max_turns_guard(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.simulation.api import Simulation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        continue_response = ConversationResponse(
            decision=ConversationStatus.CONTINUE,
            next_turn_id="turn-next",
            next_user_message="keep going",
        )

        mock_client = MagicMock()
        mock_client.initialize_run.return_value = {
            "run_id": "run-1",
            "items": [
                {"test_run_item_id": "item-1", "dataset_item_id": "ds-item-1"},
            ],
        }
        mock_client.generate_first_turn.return_value = SimulationItem(
            run_item_id="item-1",
            dataset_item_id="ds-item-1",
            message="hello",
            turn_id="turn-1",
        )
        mock_client.trigger_conversation.return_value = continue_response
        mock_client.post_run_status.return_value = {}
        mock_client_cls.return_value = mock_client

        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="ds-1", task=SyncTask(), max_turns=3)

        assert result is not None
        assert len(result["failed"]) == 1
        assert "Exceeded maximum turns (3)" in result["failed"][0]["error"]

    @patch("netra.simulation.api.SimulationHttpClient")
    def test_close_delegates_to_client(self, mock_client_cls: MagicMock) -> None:
        from netra.simulation.api import Simulation

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        sim = Simulation(self._make_config())
        sim.close()
        mock_client.close.assert_called_once()

    @patch("netra.simulation.api.SpanWrapper")
    @patch("netra.simulation.api.SimulationHttpClient")
    def test_trigger_conversation_none_response(self, mock_client_cls: MagicMock, mock_span_wrapper: MagicMock) -> None:
        from netra.simulation.api import Simulation

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.get_current_span.return_value = None
        mock_span_wrapper.return_value = mock_span

        mock_client = MagicMock()
        mock_client.initialize_run.return_value = {
            "run_id": "run-1",
            "items": [
                {"test_run_item_id": "item-1", "dataset_item_id": "ds-item-1"},
            ],
        }
        mock_client.generate_first_turn.return_value = SimulationItem(
            run_item_id="item-1",
            dataset_item_id="ds-item-1",
            message="hello",
            turn_id="turn-1",
        )
        mock_client.trigger_conversation.return_value = None
        mock_client.post_run_status.return_value = {}
        mock_client_cls.return_value = mock_client

        sim = Simulation(self._make_config())
        result = sim.run_simulation(name="test", dataset_id="ds-1", task=SyncTask())

        assert result is not None
        assert len(result["failed"]) == 1
        assert "Failed to get conversation response" in result["failed"][0]["error"]


# ---------------------------------------------------------------------------
# Section 5: BaseTask
# ---------------------------------------------------------------------------


class TestBaseTask:
    """Tests for BaseTask abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTask()  # type: ignore[abstract]

    def test_sync_subclass(self) -> None:
        task = SyncTask()
        result = task.run(message="hi")
        assert isinstance(result, TaskResult)
        assert result.message == "echo: hi"

    def test_async_subclass(self) -> None:
        task = AsyncTask()
        result = asyncio.run(task.run(message="hi"))  # type: ignore[arg-type]
        assert isinstance(result, TaskResult)
        assert result.message == "async-echo: hi"


# ---------------------------------------------------------------------------
# Section 6: SimulationHooks
# ---------------------------------------------------------------------------


class TestSimulationHooks:
    """Tests for SimulationHooks lifecycle management."""

    def test_before_hook_for_specific_item(self) -> None:
        """Test that before hook runs only for matching items."""
        from netra.simulation.hooks import SimulationHooks, run_before

        call_log = []

        def setup_refund(shared_context: Optional[dict]) -> dict:
            call_log.append("before:refund")
            return {"refund_account": "12345"}

        hooks = SimulationHooks(before={"refund-item": setup_refund})

        # Run for matching item
        result = asyncio.run(run_before(hooks, "refund-item", {"employee_id": "emp-1"}))
        assert call_log == ["before:refund"]
        assert result == {"employee_id": "emp-1", "refund_account": "12345"}

        # Run for non-matching item
        call_log.clear()
        result = asyncio.run(run_before(hooks, "other-item", {"employee_id": "emp-1"}))
        assert call_log == []
        assert result == {"employee_id": "emp-1"}

    def test_before_hook_merges_context(self) -> None:
        """Test that before hook merges context correctly."""
        from netra.simulation.hooks import SimulationHooks, run_before

        call_log = []

        def setup_refund(shared_context: Optional[dict]) -> dict:
            call_log.append("before:refund")
            return {"refund_account": "12345", "token": "abc123"}

        hooks = SimulationHooks(before={"refund-item": setup_refund})

        result = asyncio.run(run_before(hooks, "refund-item", {"employee_id": "emp-1"}))

        assert call_log == ["before:refund"]
        assert result == {"employee_id": "emp-1", "token": "abc123", "refund_account": "12345"}

    def test_after_hook_for_specific_item(self) -> None:
        """Test that after hook runs only for matching items."""
        from netra.simulation.hooks import SimulationHooks, run_after

        call_log = []

        def teardown_refund(result: dict, shared_context: Optional[dict]) -> None:
            call_log.append("after:refund")

        hooks = SimulationHooks(after={"refund-item": teardown_refund})

        # Run for matching item
        asyncio.run(run_after(hooks, "refund-item", {"success": True}, {"employee_id": "emp-1"}))
        assert call_log == ["after:refund"]

        # Run for non-matching item
        call_log.clear()
        asyncio.run(run_after(hooks, "other-item", {"success": True}, {"employee_id": "emp-1"}))
        assert call_log == []

    def test_async_hooks(self) -> None:
        """Test that async hooks are properly awaited."""
        from netra.simulation.hooks import SimulationHooks, run_before

        call_log = []

        async def async_setup(shared_context: Optional[dict]) -> dict:
            await asyncio.sleep(0.001)
            call_log.append("async_before")
            return {"token": "xyz"}

        hooks = SimulationHooks(before={"item-1": async_setup})
        result = asyncio.run(run_before(hooks, "item-1", {}))

        assert call_log == ["async_before"]
        assert result == {"token": "xyz"}

    def test_hooks_describe(self) -> None:
        """Test that hooks.describe() returns run-level + per-item metadata.

        Run-level hooks land on ``beforeAll`` / ``afterAll``. Item-level hooks
        are under ``items``, each keyed by ``datasetItemId``.
        """
        from netra.simulation.hooks import SimulationHooks

        def setup_all() -> dict:
            """Setup shared resources."""
            return {}

        def setup_refund(shared_context: Optional[dict]) -> dict:
            """Setup refund scenario."""
            return {}

        def teardown_refund(result: dict, setup_context: Optional[dict]) -> None:
            """Teardown refund scenario."""

        def teardown_all(results: dict, shared_context: Optional[dict]) -> None:
            """Teardown shared resources."""

        hooks = SimulationHooks(
            before_all=setup_all,
            before={"refund-item": setup_refund},
            after={"refund-item": teardown_refund},
            after_all=teardown_all,
        )

        meta = hooks.describe()

        assert "beforeAll" in meta
        assert meta["beforeAll"]["configured"] is True
        assert meta["beforeAll"]["name"] == "setup_all"
        assert meta["beforeAll"]["description"] == "Setup shared resources."

        assert "afterAll" in meta
        assert meta["afterAll"]["configured"] is True
        assert meta["afterAll"]["name"] == "teardown_all"
        assert meta["afterAll"]["description"] == "Teardown shared resources."

        assert "items" in meta
        assert len(meta["items"]) == 1
        item = meta["items"][0]
        assert item["datasetItemId"] == "refund-item"
        assert item["before"]["configured"] is True
        assert item["before"]["name"] == "setup_refund"
        assert item["before"]["description"] == "Setup refund scenario."
        assert item["after"]["configured"] is True
        assert item["after"]["name"] == "teardown_refund"
        assert item["after"]["description"] == "Teardown refund scenario."

    def test_hooks_describe_empty(self) -> None:
        """Test that hooks.describe() returns empty dict when no hooks configured."""
        from netra.simulation.hooks import SimulationHooks

        hooks = SimulationHooks()
        meta = hooks.describe()

        assert meta == {}

    def test_hooks_describe_multiple_items(self) -> None:
        """Test that item-level hooks are emitted per datasetItemId under items."""
        from netra.simulation.hooks import SimulationHooks

        def setup_scenario(shared_context: Optional[dict]) -> dict:
            """Setup for any scenario."""
            return {}

        def teardown_scenario(result: dict, setup_context: Optional[dict]) -> None:
            """Teardown for any scenario."""

        hooks = SimulationHooks(
            before={
                "item-1": setup_scenario,
                "item-2": setup_scenario,
                "item-3": setup_scenario,
            },
            after={
                "item-1": teardown_scenario,
                "item-2": teardown_scenario,
            },
        )

        meta = hooks.describe()

        assert "items" in meta
        by_id = {entry["datasetItemId"]: entry for entry in meta["items"]}
        assert set(by_id) == {"item-1", "item-2", "item-3"}

        for item_id in ("item-1", "item-2"):
            assert by_id[item_id]["before"]["name"] == "setup_scenario"
            assert by_id[item_id]["after"]["name"] == "teardown_scenario"

        # item-3 has before only
        assert by_id["item-3"]["before"]["name"] == "setup_scenario"
        assert "after" not in by_id["item-3"]
