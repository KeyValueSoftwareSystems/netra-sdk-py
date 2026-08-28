import concurrent.futures
import logging
import threading
from typing import Any, Optional

from netra.config import Config
from netra.red_team.client import RedTeamHttpClient
from netra.red_team.constants import LOG_PREFIX, MAX_AGENT_RESPONSE_CHARS, MAX_TURN_INDEX, SPAN_NAME
from netra.red_team.exceptions import RedTeamError
from netra.red_team.models import RedTeamResult, RunPromptItem, RunResultItem
from netra.red_team.task import RedTeamAgentHandler, execute_task
from netra.red_team.utils import resolve_max_concurrency, validate_red_team_inputs
from netra.shutdown_hooks import register_shutdown_hook, unregister_shutdown_hook
from netra.span_wrapper import SpanWrapper
from netra.utils import run_async_safely, truncate_string

logger = logging.getLogger(__name__)


class RedTeam:
    """Public API for triggering an existing red-team config and driving its
    multi-turn adversarial conversation loop against a local agent function.
    """

    __slots__ = ("_config", "_client")

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = RedTeamHttpClient(config)

    def close(self) -> None:
        """Release resources held by the redteam client."""
        self._client.close()

    def run_red_team(
        self,
        config_id: str,
        task: RedTeamAgentHandler,
        max_concurrency: Optional[int] = None,
    ) -> Optional[RedTeamResult]:
        """Trigger an existing red-team config and drive its run to completion.

        Fetches the run's prompt list once, then drives every session's
        turns locally against ``task``, submitting each turn's result and
        following the next-prompt/turn-index response until it's done.

        Args:
            config_id: Identifier of a red-team config already created ahead
                of time (e.g. in the dashboard).
            task: A plain callback ``(prompt, session_id, turn_index) ->
                str | {"message": str, "session_id"?: str}``, sync or async —
                matching the ``task`` naming used in the simulation/evaluation
                modules.
            max_concurrency: Maximum number of sessions driven in parallel.
                Capped at 5. Defaults to 5.

        Returns:
            A :class:`RedTeamResult`, or ``None`` if the inputs are invalid
            (logged, no network call made).

        Raises:
            netra.red_team.exceptions.RedTeamError: Or a subclass, for any
                failure other than invalid input. Carries ``.run_id`` when a
                run was already created, and the run is best-effort cancelled
                server-side before this is raised.
        """
        if not validate_red_team_inputs(config_id, task, max_concurrency):
            return None

        effective_concurrency = resolve_max_concurrency(max_concurrency)

        create_result = self._client.create_run(config_id)
        if create_result.get("status") == "generating":
            create_result = self._client.await_run_ready(config_id)

        run_id = create_result.get("run_id")
        if not run_id:
            raise RedTeamError(f"Backend did not return a run_id for config '{config_id}'")

        stop_event = threading.Event()

        def _cancel_on_shutdown() -> None:
            stop_event.set()
            try:
                self._client.cancel(run_id)
            except Exception:
                logger.debug("%s: shutdown-triggered cancel failed for run %s", LOG_PREFIX, run_id, exc_info=True)

        hook_token = register_shutdown_hook(_cancel_on_shutdown)
        try:
            try:
                # get_prompts is inside this block deliberately: it used to sit outside,
                # so a fetch failure skipped straight past the cancel-on-exception handler
                # below and left the run orphaned as "running" server-side.
                prompts = self._client.get_prompts(run_id)
                if not prompts:
                    logger.warning("%s: run %s has no prompts", LOG_PREFIX, run_id)

                self._drive_all_sessions(run_id, task, prompts, effective_concurrency, stop_event)
            except KeyboardInterrupt:
                # Already cancelled server-side by _cancel_on_shutdown; report
                # a clean cancelled result instead of an uncaught traceback.
                logger.info("%s: run %s interrupted; reporting as cancelled", LOG_PREFIX, run_id)
            except Exception as exc:
                # A fatal error would otherwise leave the run orphaned as
                # "running" server-side with no run_id in hand to cancel it.
                if isinstance(exc, RedTeamError):
                    exc.run_id = run_id
                try:
                    self._client.cancel(run_id)
                except Exception:
                    logger.debug("%s: best-effort cancel failed for run %s", LOG_PREFIX, run_id, exc_info=True)
                raise
        finally:
            unregister_shutdown_hook(hook_token)

        interrupted = stop_event.is_set()

        results: list[RunResultItem] = self._client.get_all_results(run_id)

        progress: Optional[dict[str, Any]] = None
        try:
            progress = self._client.get_progress(run_id)
        except Exception as exc:
            logger.warning("%s: failed to fetch progress for run %s: %s", LOG_PREFIX, run_id, exc)

        risk_score: Optional[dict[str, Any]] = None
        try:
            risk_score = self._client.get_risk_score(config_id)
        except Exception as exc:
            logger.warning("%s: failed to fetch risk score for config %s: %s", LOG_PREFIX, config_id, exc)

        status = "cancelled" if interrupted else self._client.get_run_status(run_id)
        run_number = progress.get("runNumber") if progress else None

        return RedTeamResult(
            success=status == "completed",
            status=status,
            run_id=run_id,
            config_id=config_id,
            results=results,
            run_number=run_number if isinstance(run_number, int) else None,
            progress=progress,
            risk_score=risk_score,
        )

    def get_results(self, run_id: str) -> list[RunResultItem]:
        """Fetch every graded turn result for a run."""
        return self._client.get_all_results(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel an in-progress run."""
        return self._client.cancel(run_id)

    def _drive_all_sessions(
        self,
        run_id: str,
        task: RedTeamAgentHandler,
        prompts: list[RunPromptItem],
        max_concurrency: int,
        stop_event: threading.Event,
    ) -> None:
        """Drive every prompt's session to completion, ``max_concurrency`` at a time.

        A fatal error from any session trips ``stop_event`` for the rest
        (checked cooperatively at the top of each session's loop) and is
        re-raised here once every session has settled.
        """
        if not prompts:
            return

        def _drive_in_thread(prompt: RunPromptItem) -> None:
            run_async_safely(self._drive_session(run_id, task, prompt, stop_event))

        first_exception: Optional[BaseException] = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrency, len(prompts))) as executor:
            futures = [executor.submit(_drive_in_thread, prompt) for prompt in prompts]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    stop_event.set()
                    if first_exception is None:
                        first_exception = exc

        if first_exception is not None:
            raise first_exception

    async def _drive_session(
        self,
        run_id: str,
        task: RedTeamAgentHandler,
        prompt: RunPromptItem,
        stop_event: threading.Event,
    ) -> None:
        """Drive one prompt's session from turn 1 through to ``done``."""
        session_id = prompt.id
        turn_index = 1
        prompt_text = prompt.prompt

        while True:
            if stop_event.is_set():
                return

            if turn_index > MAX_TURN_INDEX:
                # Fail fast client-side instead of spending a turn on a submit the backend
                # would just 400 on anyway.
                stop_event.set()
                raise RedTeamError(f"session {session_id} exceeded the {MAX_TURN_INDEX}-turn limit without finishing")

            with SpanWrapper(
                SPAN_NAME,
                attributes={Config.TRACE_ORIGIN_KEY: Config.TRACE_ORIGIN_RED_TEAM},
                module_name=LOG_PREFIX,
            ):
                output: Optional[str] = None
                error: Optional[str] = None
                try:
                    message, session_id = await execute_task(task, prompt_text, session_id, turn_index)
                    output = self._truncate_output(message)
                except Exception as exc:
                    error = str(exc)
                    logger.warning(
                        "%s: task failed for run_id=%s session_id=%s turn=%d: %s",
                        LOG_PREFIX,
                        run_id,
                        session_id,
                        turn_index,
                        error,
                    )

                try:
                    result = self._client.submit_turn(
                        run_id=run_id,
                        prompt_id=prompt.id,
                        session_id=session_id,
                        turn_index=turn_index,
                        prompt_text=prompt_text,
                        output=output,
                        error=error,
                    )
                except Exception:
                    stop_event.set()
                    raise

            if result.done:
                return
            prompt_text = result.next_prompt or ""
            turn_index = result.next_turn_index or (turn_index + 1)

    def _truncate_output(self, output: str) -> str:
        """Truncate an agent response to ``MAX_AGENT_RESPONSE_CHARS``."""
        if len(output) <= MAX_AGENT_RESPONSE_CHARS:
            return output
        logger.warning(
            "%s: agent response truncated from %d to %d chars",
            LOG_PREFIX,
            len(output),
            MAX_AGENT_RESPONSE_CHARS,
        )
        return truncate_string(output, MAX_AGENT_RESPONSE_CHARS)
