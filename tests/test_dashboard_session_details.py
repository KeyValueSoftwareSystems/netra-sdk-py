"""Tests for the optional time window on Dashboard.get_session_details."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from netra.dashboard.api import Dashboard
from netra.dashboard.client import DashboardHttpClient

SESSION_ID = "session-1"
SESSION_URL = f"/public/dashboard/session/{SESSION_ID}"
START_TIME = "2026-09-01T00:00:00.000Z"
END_TIME = "2026-09-02T00:00:00.000Z"


class RecordingResponse:
    """Minimal stand-in for httpx.Response covering the client's usage."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class RecordingHttpxClient:
    """Captures the GET calls the dashboard client makes."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.calls: List[Tuple[str, Optional[Dict[str, str]]]] = []

    def get(self, url: str, params: Optional[Dict[str, str]] = None) -> RecordingResponse:
        self.calls.append((url, params))
        return RecordingResponse(self._payload)


@pytest.fixture
def dashboard_with_recorder() -> Tuple[Dashboard, RecordingHttpxClient]:
    payload = {"data": {"sessionId": SESSION_ID, "traces": []}}
    recorder = RecordingHttpxClient(payload)

    dashboard = Dashboard.__new__(Dashboard)
    client = DashboardHttpClient.__new__(DashboardHttpClient)
    client._client = recorder  # type: ignore[assignment]
    dashboard._client = client

    return dashboard, recorder


@pytest.mark.unit
def test_get_session_details_sends_no_params_when_time_window_omitted(
    dashboard_with_recorder: Tuple[Dashboard, RecordingHttpxClient],
) -> None:
    dashboard, recorder = dashboard_with_recorder

    result = dashboard.get_session_details(session_id=SESSION_ID)

    assert recorder.calls == [(SESSION_URL, {})]
    assert result == {"sessionId": SESSION_ID, "traces": []}


@pytest.mark.unit
@pytest.mark.parametrize(
    "start_time,end_time,expected_params",
    [
        (START_TIME, END_TIME, {"startTime": START_TIME, "endTime": END_TIME}),
        (START_TIME, None, {"startTime": START_TIME}),
        (None, END_TIME, {"endTime": END_TIME}),
    ],
)
def test_get_session_details_forwards_time_window_as_query_params(
    dashboard_with_recorder: Tuple[Dashboard, RecordingHttpxClient],
    start_time: Optional[str],
    end_time: Optional[str],
    expected_params: Dict[str, str],
) -> None:
    dashboard, recorder = dashboard_with_recorder

    dashboard.get_session_details(session_id=SESSION_ID, start_time=start_time, end_time=end_time)

    assert recorder.calls == [(SESSION_URL, expected_params)]


@pytest.mark.unit
def test_get_session_details_rejects_empty_session_id_without_calling_backend(
    dashboard_with_recorder: Tuple[Dashboard, RecordingHttpxClient],
) -> None:
    dashboard, recorder = dashboard_with_recorder

    assert dashboard.get_session_details(session_id="", start_time=START_TIME) is None
    assert recorder.calls == []


@pytest.mark.unit
def test_get_session_details_returns_none_when_request_fails(
    dashboard_with_recorder: Tuple[Dashboard, RecordingHttpxClient],
) -> None:
    dashboard, recorder = dashboard_with_recorder

    def failing_get(url: str, params: Optional[Dict[str, str]] = None) -> RecordingResponse:
        recorder.calls.append((url, params))
        raise RuntimeError("connection reset")

    recorder.get = failing_get  # type: ignore[method-assign]

    assert dashboard.get_session_details(session_id=SESSION_ID, start_time=START_TIME) is None
    assert recorder.calls == [(SESSION_URL, {"startTime": START_TIME})]
