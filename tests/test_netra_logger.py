"""NetraLogger: level methods follow debug_mode, notice() is always visible."""

import logging
import os
import subprocess
import sys
from typing import Any, Callable, Generator, Optional, cast
from unittest.mock import MagicMock

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult

from netra import Netra
from netra.config import Config
from netra.evaluation.api import Evaluation
from netra.evaluation.client import EvaluationHttpClient
from netra.evaluation.models import GetDatasetItemsResponse
from netra.exporters.trial_aware_otlp_exporter import TrialAwareOTLPExporter
from netra.logging_utils import configure_package_logging
from netra.netra_logger import NOTICE_LOGGER_NAME, NoticeLevel, get_logger

ENDPOINT_ENV_VARS = ("NETRA_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT", "NETRA_DEBUG", "NETRA_API_KEY")
MISSING_ENDPOINT_TEXT = "NETRA_OTLP_ENDPOINT is not set"
# Port 9 (discard) is closed locally; nothing in these scripts exports, so it is never dialled.
UNUSED_LOCAL_ENDPOINT = "http://127.0.0.1:9/telemetry"
MODULE_LOGGER_NAME = "netra.tests.module"
OTEL_TRACE_EXPORTER_LOGGER_NAME = "opentelemetry.exporter.otlp.proto.http.trace_exporter"


@pytest.fixture(autouse=True)
def restore_netra_logging() -> Generator[None, None, None]:
    pkg_logger = logging.getLogger("netra")
    saved = (list(pkg_logger.handlers), pkg_logger.level, pkg_logger.propagate)
    yield
    pkg_logger.handlers[:] = saved[0]
    pkg_logger.setLevel(saved[1])
    pkg_logger.propagate = saved[2]


def _notices(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == NOTICE_LOGGER_NAME]


def _log_one_of_each(logger_name: str) -> None:
    logger = get_logger(logger_name)
    logger.debug("debug line")
    logger.info("info line")
    logger.warning("warning line")
    logger.error("error line")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("exception line")
    logger.log(logging.WARNING, "log line")


def _run_init_script(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run Netra.init() with debug_mode off in a fresh interpreter with no logging configured."""
    env = {key: value for key, value in os.environ.items() if key not in ENDPOINT_ENV_VARS}
    env.update(extra_env)
    script = "from netra import Netra\nNetra.init(app_name='notice-test', debug_mode=False)\nNetra.shutdown()\n"
    return subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120, check=True
    )


class TestLevelMethods:
    def test_every_level_method_is_silent_when_debug_mode_off(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)

        with caplog.at_level(logging.DEBUG):
            _log_one_of_each(MODULE_LOGGER_NAME)

        assert caplog.records == []

    def test_every_level_method_is_visible_when_debug_mode_on(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_package_logging(debug_mode=True)

        _log_one_of_each(MODULE_LOGGER_NAME)

        err = capsys.readouterr().err
        for level, text in [
            ("DEBUG", "debug line"),
            ("INFO", "info line"),
            ("WARNING", "warning line"),
            ("ERROR", "error line"),
            ("ERROR", "exception line"),
            ("WARNING", "log line"),
        ]:
            assert f"| {level} | {MODULE_LOGGER_NAME} | {text}" in err
        assert "RuntimeError: boom" in err  # exception() attaches the traceback

    def test_record_points_at_the_caller_not_the_wrapper(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_package_logging(debug_mode=True)
        pkg_logger = logging.getLogger("netra")
        pkg_logger.handlers[0].setFormatter(logging.Formatter("%(filename)s:%(funcName)s:%(message)s"))

        get_logger(MODULE_LOGGER_NAME).warning("where am I")
        get_logger(MODULE_LOGGER_NAME).notice("where is the notice")

        err = capsys.readouterr().err
        assert "test_netra_logger.py:test_record_points_at_the_caller_not_the_wrapper:where am I" in err
        assert "test_netra_logger.py:test_record_points_at_the_caller_not_the_wrapper:where is the notice" in err

    def test_name_is_the_module_name(self) -> None:
        assert get_logger(MODULE_LOGGER_NAME).name == MODULE_LOGGER_NAME


class TestNotice:
    def test_reaches_root_logger_when_debug_mode_off(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)

        with caplog.at_level(logging.DEBUG):
            get_logger(MODULE_LOGGER_NAME).notice("netra: something to fix")

        assert [(r.levelno, r.getMessage()) for r in _notices(caplog)] == [(logging.WARNING, "netra: something to fix")]

    def test_error_level_is_kept(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)

        with caplog.at_level(logging.DEBUG):
            get_logger(MODULE_LOGGER_NAME).notice("netra: call failed: %s", "401", level=NoticeLevel.ERROR)

        assert [(r.levelno, r.getMessage()) for r in _notices(caplog)] == [(logging.ERROR, "netra: call failed: 401")]

    def test_every_repeat_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        logger = get_logger(MODULE_LOGGER_NAME)

        with caplog.at_level(logging.DEBUG):
            for _ in range(3):
                logger.notice("netra: same problem")

        assert [r.getMessage() for r in _notices(caplog)] == ["netra: same problem"] * 3

    def test_visible_in_debug_mode(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_package_logging(debug_mode=True)

        get_logger(MODULE_LOGGER_NAME).notice("netra: something to fix")

        assert f"| WARNING | {NOTICE_LOGGER_NAME} | netra: something to fix" in capsys.readouterr().err

    def test_application_can_silence_the_notice_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        notice_logger = logging.getLogger(NOTICE_LOGGER_NAME)
        saved_level = notice_logger.level
        notice_logger.setLevel(logging.CRITICAL)
        try:
            with caplog.at_level(logging.DEBUG):
                get_logger(MODULE_LOGGER_NAME).notice("netra: muted", level=NoticeLevel.ERROR)
        finally:
            notice_logger.setLevel(saved_level)

        assert _notices(caplog) == []


class TestMissingEndpointNotice:
    def test_printed_to_stderr_when_debug_mode_off_and_logging_unconfigured(self) -> None:
        result = _run_init_script({})

        assert MISSING_ENDPOINT_TEXT in result.stderr
        assert "Fix: set NETRA_OTLP_ENDPOINT" in result.stderr

    def test_not_printed_when_endpoint_is_set(self) -> None:
        result = _run_init_script({"NETRA_OTLP_ENDPOINT": UNUSED_LOCAL_ENDPOINT})

        assert MISSING_ENDPOINT_TEXT not in result.stderr


class TestMissingApiKeyNotice:
    def test_logged_as_error_for_netra_endpoint_without_key(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NETRA_OTLP_ENDPOINT", "https://api.getnetra.ai/telemetry")
        monkeypatch.delenv("NETRA_API_KEY", raising=False)

        with caplog.at_level(logging.DEBUG):
            Config(app_name="notice-test")

        api_key_notices = [r for r in _notices(caplog) if "NETRA_API_KEY is not set" in r.getMessage()]
        assert [r.levelno for r in api_key_notices] == [logging.ERROR]

    @pytest.mark.parametrize("header_name", ["x-api-key", "X-Api-Key"])
    def test_not_logged_when_key_is_passed_in_headers_argument(
        self, header_name: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NETRA_OTLP_ENDPOINT", "https://api.getnetra.ai/telemetry")
        monkeypatch.delenv("NETRA_API_KEY", raising=False)
        monkeypatch.delenv("NETRA_HEADERS", raising=False)

        with caplog.at_level(logging.DEBUG):
            Config(app_name="notice-test", headers={header_name: "test-key"})

        assert [r for r in _notices(caplog) if "NETRA_API_KEY is not set" in r.getMessage()] == []

    def test_not_logged_when_key_is_passed_in_netra_headers_env(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NETRA_OTLP_ENDPOINT", "https://api.getnetra.ai/telemetry")
        monkeypatch.delenv("NETRA_API_KEY", raising=False)
        monkeypatch.setenv("NETRA_HEADERS", "x-api-key=test-key")

        with caplog.at_level(logging.DEBUG):
            Config(app_name="notice-test")

        assert [r for r in _notices(caplog) if "NETRA_API_KEY is not set" in r.getMessage()] == []

    def test_logged_when_headers_carry_only_an_authorization_header(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The Netra backend authenticates with x-api-key only; a bearer token is not a Netra credential.
        monkeypatch.setenv("NETRA_OTLP_ENDPOINT", "https://api.getnetra.ai/telemetry")
        monkeypatch.delenv("NETRA_API_KEY", raising=False)
        monkeypatch.delenv("NETRA_HEADERS", raising=False)

        with caplog.at_level(logging.DEBUG):
            Config(app_name="notice-test", headers={"Authorization": "Bearer test-key"})

        api_key_notices = [r for r in _notices(caplog) if "NETRA_API_KEY is not set" in r.getMessage()]
        assert [r.levelno for r in api_key_notices] == [logging.ERROR]


def _fake_response(status_code: int, reason: str, message: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.reason = reason
    response.json.return_value = {"error": {"message": message}}
    response.request.body = b"0123456789"
    return response


class TestRejectedSpanExportNotice:
    def _exporter_answering(self, responses: list[Optional[Any]]) -> TrialAwareOTLPExporter:
        wrapped = MagicMock()
        wrapped._session = None  # skip response interception; the test sets last_response itself
        exporter = TrialAwareOTLPExporter(wrapped)
        pending = iter(responses)

        def export(_spans: Any) -> SpanExportResult:
            exporter._export_state.last_response = next(pending)
            return SpanExportResult.FAILURE

        wrapped.export.side_effect = export
        return exporter

    def test_every_rejected_batch_is_visible_when_debug_mode_off(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        bad_key = _fake_response(401, "Unauthorized", "Invalid API key")
        too_large = _fake_response(413, "Payload Too Large", "Request body too large")
        exporter = self._exporter_answering([bad_key, bad_key, too_large])

        with caplog.at_level(logging.DEBUG):
            for _ in range(3):
                exporter.export([MagicMock()])

        bad_key_line = (
            'netra: Span export failed with HTTP 401 Unauthorized: "Invalid API key". '
            "Dropped a batch of 1 spans (10 bytes)."
        )
        assert [r.getMessage() for r in _notices(caplog)] == [
            bad_key_line,
            bad_key_line,
            'netra: Span export failed with HTTP 413 Payload Too Large: "Request body too large". '
            "Dropped a batch of 1 spans (10 bytes). Lower OTEL_BSP_MAX_EXPORT_BATCH_SIZE or reduce large "
            "span attributes such as prompts and completions.",
        ]

    def test_no_rejection_notice_when_export_failed_without_a_response(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        exporter = self._exporter_answering([None])

        with caplog.at_level(logging.DEBUG):
            exporter.export([MagicMock()])

        assert _notices(caplog) == []

    def test_unreachable_endpoint_is_visible_when_debug_mode_off(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        wrapped = MagicMock()
        wrapped._session = None
        wrapped.export.side_effect = ConnectionError("Connection refused")
        exporter = TrialAwareOTLPExporter(wrapped)

        with caplog.at_level(logging.DEBUG):
            result = exporter.export([MagicMock(), MagicMock()])

        assert result == SpanExportResult.FAILURE
        assert [(r.levelno, r.getMessage()) for r in _notices(caplog)] == [
            (logging.WARNING, "netra: Span export failed: Connection refused. Dropped a batch of 2 spans.")
        ]


def _otlp_exporter_over(fake_request: Callable[..., requests.Response]) -> TrialAwareOTLPExporter:
    """A real OTLPSpanExporter whose HTTP session is replaced by ``fake_request`` (no network)."""
    otlp_exporter = OTLPSpanExporter(endpoint=UNUSED_LOCAL_ENDPOINT)
    # Swapped before TrialAwareOTLPExporter wraps session.request, so its interceptor sees the fake.
    otlp_exporter._session.request = fake_request
    return TrialAwareOTLPExporter(otlp_exporter)


def _finished_span() -> ReadableSpan:
    span = TracerProvider().get_tracer("notice-test").start_span("notice-test-span")
    span.end()
    return cast(ReadableSpan, span)


def _visible_records(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [(r.name, r.getMessage()) for r in caplog.records if r.levelno >= logging.WARNING]


class TestSpanExportFailureReportedOnce:
    """Each failed batch yields exactly one visible line: Netra's notice, not also OpenTelemetry's error."""

    def test_rejected_batch_is_reported_once(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)

        def reject_with_401(method: str, url: str, **kwargs: Any) -> requests.Response:
            response = requests.Response()
            response.status_code = 401
            response.reason = "Unauthorized"
            response._content = b'{"error": {"message": "Invalid API key"}}'
            response.request = requests.Request(method, url, data=kwargs.get("data")).prepare()
            return response

        exporter = _otlp_exporter_over(reject_with_401)

        with caplog.at_level(logging.DEBUG):
            result = exporter.export([_finished_span()])

        assert result == SpanExportResult.FAILURE
        assert len(_visible_records(caplog)) == 1
        assert _visible_records(caplog)[0][0] == NOTICE_LOGGER_NAME
        assert _visible_records(caplog)[0][1].startswith(
            'netra: Span export failed with HTTP 401 Unauthorized: "Invalid API key". Dropped a batch of 1 spans'
        )

    def test_batch_without_a_response_is_reported_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """OpenTelemetry raises a request error on some versions and logs it on others; one notice either way."""
        configure_package_logging(debug_mode=False)

        def time_out(method: str, url: str, **kwargs: Any) -> requests.Response:
            raise requests.exceptions.ReadTimeout("read timed out")

        exporter = _otlp_exporter_over(time_out)

        with caplog.at_level(logging.DEBUG):
            result = exporter.export([_finished_span()])

        assert result == SpanExportResult.FAILURE
        assert _visible_records(caplog) == [
            (NOTICE_LOGGER_NAME, "netra: Span export failed: read timed out. Dropped a batch of 1 spans.")
        ]

    def test_opentelemetry_errors_outside_netra_exports_are_kept(self, caplog: pytest.LogCaptureFixture) -> None:
        configure_package_logging(debug_mode=False)
        _otlp_exporter_over(MagicMock())  # installs Netra's filter on the OpenTelemetry exporter logger

        with caplog.at_level(logging.DEBUG):
            logging.getLogger(OTEL_TRACE_EXPORTER_LOGGER_NAME).error("app-owned exporter failed")

        assert _visible_records(caplog) == [(OTEL_TRACE_EXPORTER_LOGGER_NAME, "app-owned exporter failed")]


class TestPublicApiMisuseNotice:
    @pytest.mark.parametrize(
        ("call", "expected_level", "expected_text"),
        [
            (
                lambda: Netra.set_session_id(123),  # type: ignore[arg-type]
                logging.ERROR,
                "set_session_id: session_id must be a string",
            ),
            (
                lambda: Netra.set_user_id(None),  # type: ignore[arg-type]
                logging.ERROR,
                "set_user_id: user_id must be a string",
            ),
            (lambda: Netra.set_tenant_id(""), logging.WARNING, "set_tenant_id: Tenant ID must be provided"),
            (lambda: Netra.set_custom_attributes("", None), logging.WARNING, "Both key and value must be provided"),
        ],
    )
    def test_invalid_argument_is_visible_when_debug_mode_off(
        self, call: Any, expected_level: int, expected_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        configure_package_logging(debug_mode=False)

        with caplog.at_level(logging.DEBUG):
            call()

        assert [(r.levelno, expected_text in r.getMessage()) for r in _notices(caplog)] == [(expected_level, True)]


class TestSkippedDatasetItemNotice:
    """get_dataset() drops items it cannot use; each drop is a notice, naming the item but not its payload."""

    def _get_dataset_with_items(
        self, items: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> GetDatasetItemsResponse:
        monkeypatch.setattr(EvaluationHttpClient, "get_dataset", lambda _self, _dataset_id: items)
        return cast(GetDatasetItemsResponse, Evaluation(Config(app_name="notice-test")).get_dataset("dataset-1"))

    def test_item_missing_required_fields_is_visible_when_debug_mode_off(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure_package_logging(debug_mode=False)
        items = [
            {"id": "item-1", "datasetId": "dataset-1", "input": "kept"},
            {"id": "item-2", "datasetId": "dataset-1"},
            {"input": "secret prompt text"},
        ]

        with caplog.at_level(logging.DEBUG):
            result = self._get_dataset_with_items(items, monkeypatch)

        assert [record.id for record in result.items] == ["item-1"]
        assert [(r.levelno, r.getMessage()) for r in _notices(caplog)] == [
            (logging.WARNING, "netra.evaluation: Skipping dataset item item-2: missing required field(s): input"),
            (
                logging.WARNING,
                "netra.evaluation: Skipping dataset item (no id): missing required field(s): id, datasetId",
            ),
        ]

    def test_item_failing_validation_is_visible_when_debug_mode_off(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure_package_logging(debug_mode=False)
        items = [{"id": 42, "datasetId": "dataset-1", "input": "secret prompt text"}]

        with caplog.at_level(logging.DEBUG):
            result = self._get_dataset_with_items(items, monkeypatch)

        assert result.items == []
        assert [
            (r.levelno, r.getMessage().startswith("netra.evaluation: Skipping dataset item 42:"))
            for r in _notices(caplog)
        ] == [(logging.ERROR, True)]
        assert "secret prompt text" not in _notices(caplog)[0].getMessage()


class TestApiClientFailureNotice:
    def test_call_on_unconfigured_client_is_visible_when_debug_mode_off(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ENDPOINT_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        configure_package_logging(debug_mode=False)
        client = EvaluationHttpClient(Config(app_name="notice-test"))

        with caplog.at_level(logging.DEBUG):
            result = client.create_dataset("my-dataset")

        assert result is None
        assert [(r.levelno, r.getMessage()) for r in _notices(caplog)] == [
            (logging.ERROR, "netra.evaluation: Evaluation client is not initialized; cannot create dataset")
        ]
