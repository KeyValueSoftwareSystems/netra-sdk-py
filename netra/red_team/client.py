"""Internal HTTP client for the ``/redteam/sdk/*`` backend endpoints."""

import logging
import time
from typing import Any, Optional

import httpx

from netra.config import Config
from netra.red_team.constants import (
    DEFAULT_GENERATION_POLL_INTERVAL_S,
    DEFAULT_GENERATION_TIMEOUT_S,
    DEFAULT_TIMEOUT_S,
    ENV_GENERATION_POLL_INTERVAL,
    ENV_GENERATION_TIMEOUT,
    ENV_TIMEOUT,
    LOG_PREFIX,
    RESULTS_PAGE_LIMIT,
    TELEMETRY_SUFFIX,
    URL_CANCEL_RUN,
    URL_CREATE_RUN,
    URL_GET_PROGRESS,
    URL_GET_PROMPTS,
    URL_GET_RESULTS,
    URL_GET_RISK_SCORE,
    URL_SUBMIT_TURN,
)
from netra.red_team.exceptions import (
    RedTeamAuthError,
    RedTeamConfigError,
    RedTeamError,
    RedTeamGenerationError,
    RedTeamGenerationTimeoutError,
    RedTeamRunError,
)
from netra.red_team.models import RunPromptItem, RunResultItem, SubmitTurnResult
from netra.red_team.utils import parse_env_float, unwrap_envelope
from netra.utils import extract_error_message

logger = logging.getLogger(__name__)

_STATUS_TO_ERROR: dict[int, type[RedTeamError]] = {
    400: RedTeamConfigError,
    401: RedTeamAuthError,
    403: RedTeamAuthError,
    404: RedTeamConfigError,
    409: RedTeamRunError,
    422: RedTeamConfigError,
    502: RedTeamGenerationError,
    503: RedTeamGenerationTimeoutError,
}


class RedTeamHttpClient:
    """Internal HTTP client for redteam API endpoints.

    Raises typed exceptions from :mod:`netra.red_team.exceptions` on failure.
    """

    __slots__ = ("_client",)

    def __init__(self, config: Config) -> None:
        """Initialize the redteam HTTP client.

        Raises:
            RedTeamAuthError: If ``NETRA_OTLP_ENDPOINT`` is not configured.
        """
        self._client = self._create_client(config)

    def close(self) -> None:
        """Close the underlying HTTP client and release connection resources."""
        try:
            self._client.close()
        except Exception:
            logger.debug("%s: Error closing HTTP client", LOG_PREFIX, exc_info=True)

    def _create_client(self, config: Config) -> httpx.Client:
        endpoint = (config.otlp_endpoint or "").strip()
        if not endpoint:
            raise RedTeamAuthError("NETRA_OTLP_ENDPOINT is required to use Netra.red_team")

        base_url = self._resolve_base_url(endpoint)
        headers = self._build_headers(config)
        timeout = parse_env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT_S)
        return httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def _resolve_base_url(self, endpoint: str) -> str:
        base_url = endpoint.rstrip("/")
        if base_url.endswith(TELEMETRY_SUFFIX):
            base_url = base_url[: -len(TELEMETRY_SUFFIX)]
        return base_url

    def _build_headers(self, config: Config) -> dict[str, str]:
        headers: dict[str, str] = dict(config.headers or {})
        if config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    def _to_typed_error(self, response: Optional[httpx.Response], exc: Exception) -> RedTeamError:
        message = extract_error_message(response, exc)
        error_cls = _STATUS_TO_ERROR.get(response.status_code, RedTeamError) if response is not None else RedTeamError
        return error_cls(message)

    def create_run(self, config_id: str) -> dict[str, Any]:
        """Create (or re-poll) a run for ``config_id``.

        Returns:
            ``{"run_id": ..., "config_id": ..., "status": "running"}`` or
            ``{"config_id": ..., "status": "generating"}`` (no ``run_id``)
            while prompt generation is still in progress.
        """
        response: Optional[httpx.Response] = None
        try:
            response = self._client.post(URL_CREATE_RUN, json={"configId": config_id})
            response.raise_for_status()
            data = unwrap_envelope(response.json())
            result = {"config_id": data.get("configId", config_id), "status": data.get("status", "running")}
            if "runId" in data:
                result["run_id"] = data["runId"]
            return result
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def await_run_ready(self, config_id: str) -> dict[str, Any]:
        """Poll ``create_run`` until the run is ``"running"`` or a deadline elapses.

        Raises:
            RedTeamGenerationTimeoutError: If the deadline elapses first.
        """
        interval = parse_env_float(ENV_GENERATION_POLL_INTERVAL, DEFAULT_GENERATION_POLL_INTERVAL_S)
        deadline_s = parse_env_float(ENV_GENERATION_TIMEOUT, DEFAULT_GENERATION_TIMEOUT_S)
        start = time.monotonic()

        while True:
            result = self.create_run(config_id)
            if result.get("status") != "generating":
                return result
            if time.monotonic() - start > deadline_s:
                raise RedTeamGenerationTimeoutError(
                    f"Prompt generation for config '{config_id}' did not finish within {deadline_s}s"
                )
            time.sleep(interval)

    def _fetch_run_prompts_response(self, run_id: str) -> dict[str, Any]:
        """Fetch the raw (unwrapped) ``GET /runs/{id}/prompts`` response body."""
        response: Optional[httpx.Response] = None
        try:
            url = URL_GET_PROMPTS.format(run_id=run_id)
            response = self._client.get(url)
            response.raise_for_status()
            return unwrap_envelope(response.json())  # type:ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def get_prompts(self, run_id: str) -> list[RunPromptItem]:
        """Fetch the full prompt list for a run. Should be called exactly once per run."""
        data = self._fetch_run_prompts_response(run_id)
        prompts = data.get("prompts", [])
        return [
            RunPromptItem(
                id=p.get("id", ""),
                prompt=p.get("prompt", ""),
                evaluator_id=p.get("evaluatorId", ""),
                evaluator_slug=p.get("evaluatorSlug"),
            )
            for p in prompts
        ]

    def get_run_status(self, run_id: str) -> str:
        """Re-read the run's own current status via the prompts endpoint."""
        status = self._fetch_run_prompts_response(run_id).get("status", "completed")
        return "completed" if status == "generating" else str(status)

    def submit_turn(
        self,
        run_id: str,
        prompt_id: str,
        session_id: str,
        turn_index: int,
        prompt_text: str,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> SubmitTurnResult:
        """Submit one turn's result.

        A ``409`` (already submitted, e.g. a network-retried duplicate) is
        normalized to ``done=True`` rather than raised.
        """
        payload: dict[str, Any] = {
            "promptId": prompt_id,
            "sessionId": session_id,
            "turnIndex": turn_index,
            "promptText": prompt_text,
        }
        if error is not None:
            payload["error"] = error
        else:
            payload["output"] = output or ""

        response: Optional[httpx.Response] = None
        try:
            url = URL_SUBMIT_TURN.format(run_id=run_id)
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = unwrap_envelope(response.json())
            return SubmitTurnResult(
                done=bool(data.get("done", False)),
                next_prompt=data.get("nextPrompt"),
                next_turn_index=data.get("nextTurnIndex"),
            )
        except httpx.HTTPStatusError as exc:
            if response is not None and response.status_code == 409:
                logger.debug(
                    "%s: turn (promptId=%s, turnIndex=%s) already submitted; treating as done",
                    LOG_PREFIX,
                    prompt_id,
                    turn_index,
                )
                return SubmitTurnResult(done=True)
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def get_progress(self, run_id: str) -> dict[str, Any]:
        """Fetch the opaque progress blob for a run."""
        response: Optional[httpx.Response] = None
        try:
            url = URL_GET_PROGRESS.format(run_id=run_id)
            response = self._client.get(url)
            response.raise_for_status()
            return unwrap_envelope(response.json())  # type:ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def get_results_page(
        self, run_id: str, page: int, limit: int = RESULTS_PAGE_LIMIT, evaluator_id: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch one page of graded results.

        Returns:
            ``(items, has_next_page)`` — ``has_next_page`` is the backend's own
            ``PaginatedResponseDto.hasNextPage`` field, not inferred from page length (which
            would be wrong whenever ``total`` is an exact multiple of ``limit``).
        """
        response: Optional[httpx.Response] = None
        try:
            url = URL_GET_RESULTS.format(run_id=run_id)
            params: dict[str, Any] = {"page": page, "limit": limit}
            if evaluator_id:
                params["evaluatorId"] = evaluator_id
            response = self._client.get(url, params=params)
            response.raise_for_status()
            data = unwrap_envelope(response.json())
            items = data.get("data", [])
            has_next_page = bool(data.get("hasNextPage", len(items) >= limit))
            return (list(items) if isinstance(items, list) else [], has_next_page)
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def get_all_results(self, run_id: str) -> list[RunResultItem]:
        """Fetch every page of graded results for a run."""
        results: list[RunResultItem] = []
        page = 1
        while True:
            raw_items, has_next_page = self.get_results_page(run_id, page=page, limit=RESULTS_PAGE_LIMIT)
            for item in raw_items:
                results.append(
                    RunResultItem(
                        evaluator_id=item.get("evaluatorId", ""),
                        evaluator_slug=item.get("evaluatorSlug"),
                        status=item.get("status", ""),
                        score=item.get("score"),
                        judge_output=item.get("judgeOutput"),
                        session_id=item.get("sessionId"),
                        turn_index=item.get("turnIndex"),
                        conversation_history=item.get("conversationHistory"),
                    )
                )
            if not has_next_page:
                break
            page += 1
        return results

    def get_risk_score(self, config_id: str) -> dict[str, Any]:
        """Fetch the opaque risk-score blob for a config."""
        response: Optional[httpx.Response] = None
        try:
            url = URL_GET_RISK_SCORE.format(config_id=config_id)
            response = self._client.get(url)
            response.raise_for_status()
            return unwrap_envelope(response.json())  # type:ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel an in-progress run."""
        response: Optional[httpx.Response] = None
        try:
            url = URL_CANCEL_RUN.format(run_id=run_id)
            response = self._client.post(url, json={})
            response.raise_for_status()
            return unwrap_envelope(response.json())  # type:ignore[no-any-return]
        except httpx.HTTPStatusError as exc:
            raise self._to_typed_error(response, exc) from exc
        except Exception as exc:
            raise RedTeamError(extract_error_message(response, exc)) from exc
