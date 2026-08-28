"""
Unit tests for the netra/redteam/ module and netra/shutdown_hooks.py.

Covers models, handler normalization, utils, client, api, and the shared
shutdown-hook registry with mocked HTTP interactions.
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from netra.redteam.exceptions import (
    RedTeamAuthError,
    RedTeamConfigError,
    RedTeamError,
    RedTeamGenerationError,
    RedTeamGenerationTimeoutError,
    RedTeamRunError,
)
from netra.redteam.handler import execute_handler
from netra.redteam.models import RedTeamResult, RunPromptItem, RunResultItem, SubmitTurnResult
from netra.redteam.utils import (
    parse_env_float,
    resolve_max_concurrency,
    unwrap_envelope,
    validate_red_team_inputs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_shutdown_hooks(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Stub out real signal handlers and reset shutdown-hook state between tests."""
    import netra.shutdown_hooks as sh

    monkeypatch.setattr(sh.signal, "signal", MagicMock(return_value=None))
    sh._hooks.clear()
    sh._next_token = 0
    sh._installed_signals.clear()
    sh._running = False
    yield
    sh._hooks.clear()
    sh._next_token = 0
    sh._installed_signals.clear()
    sh._running = False


def _make_config(endpoint: str = "https://api.getnetra.ai/telemetry", api_key: str = "key-1") -> MagicMock:
    """Create a mock Config."""
    cfg = MagicMock()
    cfg.otlp_endpoint = endpoint
    cfg.api_key = api_key
    cfg.headers = {}
    return cfg


def _mock_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    else:
        resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_run_prompt_item_defaults(self) -> None:
        item = RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")
        assert item.evaluator_slug is None

    def test_submit_turn_result_defaults(self) -> None:
        result = SubmitTurnResult(done=True)
        assert result.next_prompt is None
        assert result.next_turn_index is None

    def test_run_result_item_defaults(self) -> None:
        item = RunResultItem(evaluator_id="e1", status="pass")
        assert item.score is None
        assert item.session_id is None

    def test_red_team_result_defaults(self) -> None:
        result = RedTeamResult(success=True, status="completed", run_id="r1", config_id="c1")
        assert result.results == []
        assert result.run_number is None
        assert result.progress is None
        assert result.risk_score is None

    def test_red_team_error_run_id(self) -> None:
        assert RedTeamError("boom").run_id is None
        assert RedTeamError("boom", run_id="r1").run_id == "r1"


# ---------------------------------------------------------------------------
# handler.py — execute_handler normalization
# ---------------------------------------------------------------------------


class TestExecuteHandler:
    def test_sync_handler_returning_string(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> str:
            return f"reply-{prompt}"

        message, session_id = asyncio.run(execute_handler(handler, "hi", "s1", 1))
        assert message == "reply-hi"
        assert session_id == "s1"

    def test_async_handler_returning_string(self) -> None:
        async def handler(prompt: str, session_id: str, turn_index: int) -> str:
            return f"async-{prompt}"

        message, session_id = asyncio.run(execute_handler(handler, "hi", "s1", 1))
        assert message == "async-hi"
        assert session_id == "s1"

    def test_handler_returning_dict_with_message(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> dict:
            return {"message": "reply"}

        message, session_id = asyncio.run(execute_handler(handler, "hi", "s1", 1))
        assert message == "reply"
        assert session_id == "s1"

    def test_handler_returning_dict_with_session_override(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> dict:
            return {"message": "reply", "session_id": "custom"}

        message, session_id = asyncio.run(execute_handler(handler, "hi", "s1", 1))
        assert message == "reply"
        assert session_id == "custom"

    def test_handler_returning_dict_without_message_raises(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> dict:
            return {"foo": "bar"}

        with pytest.raises(TypeError):
            asyncio.run(execute_handler(handler, "hi", "s1", 1))

    def test_handler_returning_int_raises(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> int:
            return 42

        with pytest.raises(TypeError):
            asyncio.run(execute_handler(handler, "hi", "s1", 1))

    def test_handler_raising_propagates(self) -> None:
        def handler(prompt: str, session_id: str, turn_index: int) -> str:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            asyncio.run(execute_handler(handler, "hi", "s1", 1))


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------


class TestParseEnvFloat:
    def test_parses_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NETRA_REDTEAM_TEST_VAR", "3.5")
        assert parse_env_float("NETRA_REDTEAM_TEST_VAR", 1.0) == 3.5

    def test_returns_default_on_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NETRA_REDTEAM_TEST_VAR", raising=False)
        assert parse_env_float("NETRA_REDTEAM_TEST_VAR", 1.0) == 1.0

    def test_returns_default_on_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NETRA_REDTEAM_TEST_VAR", "not-a-number")
        assert parse_env_float("NETRA_REDTEAM_TEST_VAR", 1.0) == 1.0


class TestValidateRedTeamInputs:
    def test_valid_inputs(self) -> None:
        assert validate_red_team_inputs("cfg-1", lambda p, s, t: "ok", 5) is True

    def test_missing_config_id(self) -> None:
        assert validate_red_team_inputs("", lambda p, s, t: "ok", None) is False

    def test_non_callable_handler(self) -> None:
        assert validate_red_team_inputs("cfg-1", "not-a-fn", None) is False  # type: ignore[arg-type]

    def test_zero_max_concurrency(self) -> None:
        assert validate_red_team_inputs("cfg-1", lambda p, s, t: "ok", 0) is False

    def test_negative_max_concurrency(self) -> None:
        assert validate_red_team_inputs("cfg-1", lambda p, s, t: "ok", -1) is False

    def test_non_int_max_concurrency(self) -> None:
        assert validate_red_team_inputs("cfg-1", lambda p, s, t: "ok", 2.5) is False  # type: ignore[arg-type]

    def test_none_max_concurrency_is_valid(self) -> None:
        assert validate_red_team_inputs("cfg-1", lambda p, s, t: "ok", None) is True


class TestResolveMaxConcurrency:
    def test_clamps_above_default(self) -> None:
        assert resolve_max_concurrency(20) == 5

    def test_passes_through_below_default(self) -> None:
        assert resolve_max_concurrency(2) == 2

    def test_none_uses_default(self) -> None:
        assert resolve_max_concurrency(None) == 5


class TestUnwrapEnvelope:
    def test_unwraps_single_envelope(self) -> None:
        assert unwrap_envelope({"success": True, "data": {"foo": "bar"}}) == {"foo": "bar"}

    def test_does_not_recursively_unwrap(self) -> None:
        raw = {"success": True, "data": {"data": [{"foo": "bar"}], "total": 1}}
        assert unwrap_envelope(raw) == {"data": [{"foo": "bar"}], "total": 1}

    def test_passthrough_when_not_enveloped(self) -> None:
        assert unwrap_envelope({"foo": "bar"}) == {"foo": "bar"}


# ---------------------------------------------------------------------------
# client.py
# ---------------------------------------------------------------------------


class TestRedTeamHttpClient:
    def test_create_client_with_valid_config(self) -> None:
        from netra.redteam.client import RedTeamHttpClient

        client = RedTeamHttpClient(_make_config())
        assert client._client is not None
        client.close()

    def test_create_client_strips_telemetry_suffix(self) -> None:
        from netra.redteam.client import RedTeamHttpClient

        client = RedTeamHttpClient(_make_config(endpoint="https://api.getnetra.ai/telemetry"))
        assert "/telemetry" not in str(client._client.base_url)
        client.close()

    def test_create_client_raises_on_empty_endpoint(self) -> None:
        from netra.redteam.client import RedTeamHttpClient

        with pytest.raises(RedTeamAuthError):
            RedTeamHttpClient(_make_config(endpoint=""))

    def test_close_is_idempotent(self) -> None:
        from netra.redteam.client import RedTeamHttpClient

        client = RedTeamHttpClient(_make_config())
        client.close()
        client.close()  # should not raise

    @patch("netra.redteam.client.httpx.Client")
    def test_create_run_running(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(
            202, {"success": True, "data": {"runId": "run-1", "configId": "cfg-1", "status": "running"}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        result = client.create_run("cfg-1")
        assert result == {"run_id": "run-1", "config_id": "cfg-1", "status": "running"}

    @patch("netra.redteam.client.httpx.Client")
    def test_create_run_generating_has_no_run_id(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(
            202, {"success": True, "data": {"configId": "cfg-1", "status": "generating"}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        result = client.create_run("cfg-1")
        assert result["status"] == "generating"
        assert "run_id" not in result

    @patch("netra.redteam.client.time.sleep", return_value=None)
    @patch("netra.redteam.client.httpx.Client")
    def test_await_run_ready_polls_until_running(self, mock_client_cls: MagicMock, _mock_sleep: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.side_effect = [
            _mock_response(202, {"success": True, "data": {"configId": "cfg-1", "status": "generating"}}),
            _mock_response(202, {"success": True, "data": {"configId": "cfg-1", "status": "generating"}}),
            _mock_response(
                202, {"success": True, "data": {"runId": "run-1", "configId": "cfg-1", "status": "running"}}
            ),
        ]
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        result = client.await_run_ready("cfg-1")
        assert result == {"run_id": "run-1", "config_id": "cfg-1", "status": "running"}
        assert mock_instance.post.call_count == 3

    @patch("netra.redteam.client.time.sleep", return_value=None)
    @patch("netra.redteam.client.time.monotonic")
    @patch("netra.redteam.client.httpx.Client")
    def test_await_run_ready_times_out(
        self, mock_client_cls: MagicMock, mock_monotonic: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        from netra.redteam.client import RedTeamHttpClient

        # First call establishes `start`; every call after must read past the deadline.
        mock_monotonic.side_effect = [0.0] + [10_000.0] * 10
        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(
            202, {"success": True, "data": {"configId": "cfg-1", "status": "generating"}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        with pytest.raises(RedTeamGenerationTimeoutError):
            client.await_run_ready("cfg-1")

    @patch("netra.redteam.client.httpx.Client")
    def test_get_prompts(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.get.return_value = _mock_response(
            200,
            {
                "success": True,
                "data": {
                    "runId": "run-1",
                    "status": "running",
                    "turnType": "multi",
                    "multiTurnCount": 5,
                    "prompts": [
                        {"id": "p1", "prompt": "hi", "evaluatorId": "e1", "evaluatorSlug": "slug-1"},
                    ],
                },
            },
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        prompts = client.get_prompts("run-1")
        assert len(prompts) == 1
        assert prompts[0] == RunPromptItem(id="p1", prompt="hi", evaluator_id="e1", evaluator_slug="slug-1")

    @patch("netra.redteam.client.httpx.Client")
    def test_get_run_status_maps_generating_to_completed(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.get.return_value = _mock_response(
            200, {"success": True, "data": {"status": "generating", "prompts": []}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        assert client.get_run_status("run-1") == "completed"

    @patch("netra.redteam.client.httpx.Client")
    def test_get_run_status_passes_through(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.get.return_value = _mock_response(
            200, {"success": True, "data": {"status": "cancelled", "prompts": []}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        assert client.get_run_status("run-1") == "cancelled"

    @patch("netra.redteam.client.httpx.Client")
    def test_submit_turn_continue(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(
            200, {"success": True, "data": {"done": False, "nextPrompt": "next", "nextTurnIndex": 2}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        result = client.submit_turn(
            run_id="run-1", prompt_id="p1", session_id="s1", turn_index=1, prompt_text="hi", output="reply"
        )
        assert result == SubmitTurnResult(done=False, next_prompt="next", next_turn_index=2)
        sent_body = mock_instance.post.call_args.kwargs["json"]
        assert sent_body["output"] == "reply"
        assert "error" not in sent_body

    @patch("netra.redteam.client.httpx.Client")
    def test_submit_turn_error_field(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(200, {"success": True, "data": {"done": True}})
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        client.submit_turn(
            run_id="run-1", prompt_id="p1", session_id="s1", turn_index=1, prompt_text="hi", error="boom"
        )
        sent_body = mock_instance.post.call_args.kwargs["json"]
        assert sent_body["error"] == "boom"
        assert "output" not in sent_body

    @patch("netra.redteam.client.httpx.Client")
    def test_submit_turn_409_normalized_to_done(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(409, {"error": {"message": "already submitted"}})
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        result = client.submit_turn(
            run_id="run-1", prompt_id="p1", session_id="s1", turn_index=1, prompt_text="hi", output="reply"
        )
        assert result == SubmitTurnResult(done=True)

    @pytest.mark.parametrize(
        "status_code,expected_exc",
        [
            (400, RedTeamConfigError),
            (401, RedTeamAuthError),
            (403, RedTeamAuthError),
            (404, RedTeamConfigError),
            (422, RedTeamConfigError),
            (502, RedTeamGenerationError),
            (503, RedTeamGenerationTimeoutError),
        ],
    )
    @patch("netra.redteam.client.httpx.Client")
    def test_get_prompts_error_mapping(self, mock_client_cls: MagicMock, status_code: int, expected_exc: type) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.get.return_value = _mock_response(status_code, {"error": {"message": "failed"}})
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        with pytest.raises(expected_exc):
            client.get_prompts("run-1")

    @patch("netra.redteam.client.httpx.Client")
    def test_create_run_409_raises_run_error(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(
            409, {"error": {"message": "A run is already active for this config."}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        with pytest.raises(RedTeamRunError):
            client.create_run("cfg-1")

    @patch("netra.redteam.client.httpx.Client")
    def test_cancel_409_raises_run_error(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(409, {"error": {"message": "Run is not in RUNNING status."}})
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        with pytest.raises(RedTeamRunError):
            client.cancel("run-1")

    @patch("netra.redteam.client.httpx.Client")
    def test_get_all_results_paginates(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient
        from netra.redteam.constants import RESULTS_PAGE_LIMIT

        first_page_items = [
            {"evaluatorId": "e1", "status": "pass", "sessionId": f"s{i}", "turnIndex": 1}
            for i in range(RESULTS_PAGE_LIMIT)
        ]
        second_page_items = [{"evaluatorId": "e1", "status": "fail", "sessionId": "sX", "turnIndex": 1}]

        mock_instance = MagicMock()
        mock_instance.get.side_effect = [
            _mock_response(200, {"success": True, "data": {"data": first_page_items, "page": 1, "total": 201}}),
            _mock_response(200, {"success": True, "data": {"data": second_page_items, "page": 2, "total": 201}}),
        ]
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        results = client.get_all_results("run-1")
        assert len(results) == RESULTS_PAGE_LIMIT + 1
        assert results[-1].status == "fail"
        assert mock_instance.get.call_count == 2

    @patch("netra.redteam.client.httpx.Client")
    def test_get_risk_score(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.get.return_value = _mock_response(
            200, {"success": True, "data": {"configId": "cfg-1", "latestSafetyScore": 90}}
        )
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        assert client.get_risk_score("cfg-1") == {"configId": "cfg-1", "latestSafetyScore": 90}

    @patch("netra.redteam.client.httpx.Client")
    def test_cancel_success(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.client import RedTeamHttpClient

        mock_instance = MagicMock()
        mock_instance.post.return_value = _mock_response(200, {"success": True, "data": {"status": "cancelled"}})
        mock_client_cls.return_value = mock_instance

        client = RedTeamHttpClient(_make_config())
        assert client.cancel("run-1") == {"status": "cancelled"}


# ---------------------------------------------------------------------------
# api.py — the public RedTeam class
# ---------------------------------------------------------------------------


class TestRedTeam:
    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_returns_none_on_invalid_inputs(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="", handler=lambda p, s, t: "ok")
        assert result is None
        mock_client_cls.return_value.create_run.assert_not_called()

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_polls_through_generation_gating(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"config_id": "cfg-1", "status": "generating"}
        mock_client.await_run_ready.return_value = {"run_id": "run-1", "config_id": "cfg-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.success is True
        mock_client.await_run_ready.assert_called_once_with("cfg-1")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_empty_prompts_still_succeeds(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = []
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.success is True
        assert result.results == []
        mock_client.submit_turn.assert_not_called()

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_multi_turn_loop_threads_prompt_and_index(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="turn1", evaluator_id="e1")]

        submit_calls = []

        def fake_submit_turn(**kwargs: Any) -> SubmitTurnResult:
            submit_calls.append(kwargs)
            if kwargs["turn_index"] < 3:
                return SubmitTurnResult(
                    done=False, next_prompt=f"turn{kwargs['turn_index'] + 1}", next_turn_index=kwargs["turn_index"] + 1
                )
            return SubmitTurnResult(done=True)

        mock_client.submit_turn.side_effect = fake_submit_turn
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        def handler(prompt: str, session_id: str, turn_index: int) -> str:
            return f"reply-to-{prompt}"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=handler)

        assert result is not None and result.success is True
        assert len(submit_calls) == 3
        assert [c["turn_index"] for c in submit_calls] == [1, 2, 3]
        assert [c["prompt_text"] for c in submit_calls] == ["turn1", "turn2", "turn3"]
        assert [c["output"] for c in submit_calls] == ["reply-to-turn1", "reply-to-turn2", "reply-to-turn3"]

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_sessions_do_not_cross_talk(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [
            RunPromptItem(id="p1", prompt="prompt-1", evaluator_id="e1"),
            RunPromptItem(id="p2", prompt="prompt-2", evaluator_id="e1"),
            RunPromptItem(id="p3", prompt="prompt-3", evaluator_id="e1"),
        ]

        submit_calls = []

        def fake_submit_turn(**kwargs: Any) -> SubmitTurnResult:
            submit_calls.append(kwargs)
            if kwargs["turn_index"] < 2:
                return SubmitTurnResult(done=False, next_prompt=f"{kwargs['prompt_text']}-t2", next_turn_index=2)
            return SubmitTurnResult(done=True)

        mock_client.submit_turn.side_effect = fake_submit_turn
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        def handler(prompt: str, session_id: str, turn_index: int) -> str:
            return f"reply-{prompt}"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=handler, max_concurrency=3)

        assert result is not None and result.success is True
        # 3 sessions x 2 turns = 6 total submissions, each session sees only its own prompt lineage
        assert len(submit_calls) == 6
        by_session: dict[str, list[dict]] = {}
        for c in submit_calls:
            by_session.setdefault(c["session_id"], []).append(c)
        assert set(by_session.keys()) == {"p1", "p2", "p3"}
        for session_id, calls in by_session.items():
            assert len(calls) == 2
            assert calls[0]["prompt_id"] == session_id
            assert calls[1]["prompt_id"] == session_id
            assert calls[1]["prompt_text"] == f"prompt-{session_id[-1]}-t2"

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_handler_error_submitted_but_run_completes(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        def bad_handler(prompt: str, session_id: str, turn_index: int) -> str:
            raise ValueError("agent exploded")

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=bad_handler)

        assert result is not None
        assert result.success is True  # overall run completion, not per-turn pass rate
        submitted = mock_client.submit_turn.call_args.kwargs
        assert submitted["error"] == "agent exploded"
        assert submitted["output"] is None

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_fatal_submit_failure_propagates(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.side_effect = RedTeamError("network died")

        rt = RedTeam(_make_config())
        with pytest.raises(RedTeamError) as exc_info:
            rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        # Carries run_id so the caller can inspect/manually cancel, and the
        # run is best-effort cancelled server-side before the error propagates.
        assert exc_info.value.run_id == "run-1"
        mock_client.cancel.assert_called_once_with("run-1")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_fatal_failure_cancel_error_does_not_mask_original(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.side_effect = RedTeamError("network died")
        mock_client.cancel.side_effect = RedTeamError("cancel also failed")

        rt = RedTeam(_make_config())
        with pytest.raises(RedTeamError, match="network died"):
            rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_keyboard_interrupt_returns_cancelled_result(self, mock_client_cls: MagicMock) -> None:
        """A KeyboardInterrupt mid-run is swallowed into a cancelled result, not raised."""
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "running"  # would only be consulted if not interrupted

        def fake_drive_all(self: Any, run_id: str, h: Any, prompts: Any, max_c: int, stop_event: Any) -> None:
            stop_event.set()
            raise KeyboardInterrupt

        with patch.object(RedTeam, "_drive_all_sessions", fake_drive_all):
            rt = RedTeam(_make_config())
            result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.status == "cancelled"
        assert result.success is False
        mock_client.get_run_status.assert_not_called()

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_progress_failure_is_best_effort(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.side_effect = RedTeamError("progress endpoint down")
        mock_client.get_risk_score.return_value = {"latestSafetyScore": 80}
        mock_client.get_run_status.return_value = "completed"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.success is True
        assert result.progress is None
        assert result.risk_score == {"latestSafetyScore": 80}

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_risk_score_failure_is_best_effort(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {"runNumber": 3}
        mock_client.get_risk_score.side_effect = RedTeamError("risk score endpoint down")
        mock_client.get_run_status.return_value = "completed"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.success is True
        assert result.risk_score is None
        assert result.run_number == 3

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_final_status_not_completed_is_unsuccessful(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "failed"

        rt = RedTeam(_make_config())
        result = rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert result is not None
        assert result.status == "failed"
        assert result.success is False

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_missing_run_id_raises(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"status": "running"}  # malformed: no run_id

        rt = RedTeam(_make_config())
        with pytest.raises(RedTeamError):
            rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_get_results_delegates_to_client(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        expected = [RunResultItem(evaluator_id="e1", status="pass")]
        mock_client.get_all_results.return_value = expected

        rt = RedTeam(_make_config())
        assert rt.get_results("run-1") is expected
        mock_client.get_all_results.assert_called_once_with("run-1")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_cancel_delegates_to_client(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.cancel.return_value = {"status": "cancelled"}

        rt = RedTeam(_make_config())
        assert rt.cancel("run-1") == {"status": "cancelled"}
        mock_client.cancel.assert_called_once_with("run-1")

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_close_delegates_to_client(self, mock_client_cls: MagicMock) -> None:
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        rt = RedTeam(_make_config())
        rt.close()
        mock_client.close.assert_called_once()

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_unregisters_shutdown_hook_after_completion(self, mock_client_cls: MagicMock) -> None:
        import netra.shutdown_hooks as sh
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.return_value = SubmitTurnResult(done=True)
        mock_client.get_all_results.return_value = []
        mock_client.get_progress.return_value = {}
        mock_client.get_risk_score.return_value = {}
        mock_client.get_run_status.return_value = "completed"

        rt = RedTeam(_make_config())
        rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert len(sh._hooks) == 0

    @patch("netra.redteam.api.RedTeamHttpClient")
    def test_run_red_team_unregisters_shutdown_hook_even_if_drive_raises(self, mock_client_cls: MagicMock) -> None:
        """Even on a fatal error mid-run, the hook is still unregistered (finally block)."""
        import netra.shutdown_hooks as sh
        from netra.redteam.api import RedTeam

        mock_client = mock_client_cls.return_value
        mock_client.create_run.return_value = {"run_id": "run-1", "status": "running"}
        mock_client.get_prompts.return_value = [RunPromptItem(id="p1", prompt="hi", evaluator_id="e1")]
        mock_client.submit_turn.side_effect = RedTeamError("boom")

        rt = RedTeam(_make_config())
        with pytest.raises(RedTeamError):
            rt.run_red_team(config_id="cfg-1", handler=lambda p, s, t: "ok")

        assert len(sh._hooks) == 0


# ---------------------------------------------------------------------------
# shutdown_hooks.py
# ---------------------------------------------------------------------------


class TestShutdownHooks:
    def test_register_returns_unique_tokens(self) -> None:
        from netra.shutdown_hooks import register_shutdown_hook

        token1 = register_shutdown_hook(lambda: None)
        token2 = register_shutdown_hook(lambda: None)
        assert token1 != token2

    def test_unregister_removes_hook(self) -> None:
        import netra.shutdown_hooks as sh

        calls = []
        token = sh.register_shutdown_hook(lambda: calls.append(1))
        sh.unregister_shutdown_hook(token)
        sh.run_shutdown_hooks()
        assert calls == []

    def test_unregister_unknown_token_is_a_noop(self) -> None:
        from netra.shutdown_hooks import unregister_shutdown_hook

        unregister_shutdown_hook(9999)  # should not raise

    def test_run_shutdown_hooks_runs_all_registered_hooks(self) -> None:
        import netra.shutdown_hooks as sh

        calls: list[int] = []
        sh.register_shutdown_hook(lambda: calls.append(1))
        sh.register_shutdown_hook(lambda: calls.append(2))
        sh.run_shutdown_hooks()
        assert sorted(calls) == [1, 2]

    def test_run_shutdown_hooks_isolates_a_raising_hook(self) -> None:
        import netra.shutdown_hooks as sh

        calls: list[str] = []

        def bad_hook() -> None:
            raise RuntimeError("bad hook")

        sh.register_shutdown_hook(bad_hook)
        sh.register_shutdown_hook(lambda: calls.append("good"))
        sh.run_shutdown_hooks()  # must not raise
        assert calls == ["good"]

    def test_run_shutdown_hooks_is_reentrancy_guarded(self) -> None:
        import netra.shutdown_hooks as sh

        calls: list[str] = []

        def recursive_hook() -> None:
            calls.append("outer")
            sh.run_shutdown_hooks()  # should no-op, not recurse

        sh.register_shutdown_hook(recursive_hook)
        sh.run_shutdown_hooks()
        assert calls == ["outer"]

    def test_run_shutdown_hooks_does_not_block_past_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import netra.shutdown_hooks as sh

        monkeypatch.setattr(sh, "SHUTDOWN_HOOK_TIMEOUT_S", 0.2)
        sh.register_shutdown_hook(lambda: time.sleep(2))
        sh.register_shutdown_hook(lambda: None)

        start = time.monotonic()
        sh.run_shutdown_hooks()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_register_installs_signal_handlers_lazily(self) -> None:
        import netra.shutdown_hooks as sh

        assert len(sh._installed_signals) == 0
        sh.register_shutdown_hook(lambda: None)
        assert sh.signal.SIGINT in sh._installed_signals
        assert sh.signal.SIGTERM in sh._installed_signals

    def test_signal_handler_runs_hooks_restores_handler_and_redelivers(self) -> None:
        import signal as real_signal

        import netra.shutdown_hooks as sh

        killed: list[tuple[int, int]] = []
        with patch.object(sh.os, "kill", lambda pid, sig: killed.append((pid, sig))):
            calls: list[str] = []
            sh.register_shutdown_hook(lambda: calls.append("hook"))

            sigint_handler = None
            for call in sh.signal.signal.call_args_list:  # type: ignore[attr-defined]
                if call.args[0] == real_signal.SIGINT:
                    sigint_handler = call.args[1]
            assert sigint_handler is not None

            sigint_handler(real_signal.SIGINT, None)

            assert calls == ["hook"]
            assert killed == [(__import__("os").getpid(), real_signal.SIGINT)]

    def test_ensure_signal_handlers_installed_skips_outside_main_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import netra.shutdown_hooks as sh

        def _raise_value_error(sig: Any, handler: Any) -> None:
            raise ValueError("signal only works in main thread")

        monkeypatch.setattr(sh.signal, "signal", _raise_value_error)
        # Should not raise, just skip installing and log a debug message.
        sh.register_shutdown_hook(lambda: None)
        assert sh._installed_signals == {}
