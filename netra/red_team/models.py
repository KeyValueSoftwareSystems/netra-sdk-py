"""Data models for the redteam module."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True, frozen=True)
class RunPromptItem:
    """One catalog prompt for a run, as returned by ``GET /runs/{id}/prompts``.

    Attributes:
        id: Prompt identifier. Used as ``promptId`` on every turn submission
            for the session driven from this prompt, and as that session's
            default ``sessionId``.
        prompt: The initial attacker prompt text (turn 1's ``promptText``).
        evaluator_id: Identifier of the evaluator that grades this prompt's turns.
        evaluator_slug: Human-readable slug for the evaluator.
    """

    id: str
    prompt: str
    evaluator_id: str
    evaluator_slug: Optional[str] = None


@dataclass(slots=True, frozen=True)
class SubmitTurnResult:
    """Response from ``POST /runs/{id}/turns``.

    Attributes:
        done: Whether the session this turn belongs to has finished.
        next_prompt: The next attacker prompt to send, when ``done`` is False.
        next_turn_index: The turn index to submit next, when ``done`` is False.
    """

    done: bool
    next_prompt: Optional[str] = None
    next_turn_index: Optional[int] = None


@dataclass(slots=True, frozen=True)
class RunResultItem:
    """A single graded turn result, as returned by ``GET /runs/{id}/results``.

    Attributes:
        evaluator_id: Identifier of the evaluator that graded this turn.
        evaluator_slug: Human-readable slug for the evaluator.
        status: One of ``"pass"``, ``"fail"``, ``"error"``, or ``"cancelled"``.
        score: Optional numeric judge score.
        judge_output: Optional raw judge reasoning/output.
        session_id: The session this result belongs to.
        turn_index: The turn index this result belongs to.
        conversation_history: This turn's own prompt/output exchange.
    """

    evaluator_id: str
    status: str
    evaluator_slug: Optional[str] = None
    score: Optional[float] = None
    judge_output: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    conversation_history: Optional[Any] = None


@dataclass(slots=True)
class RedTeamResult:
    """Aggregated outcome of a ``run_red_team()`` call.

    Attributes:
        success: True iff ``status == "completed"``.
        status: Final run status: ``"running"``, ``"completed"``, ``"failed"``,
            or ``"cancelled"``.
        run_id: Identifier of the run that was driven.
        config_id: Identifier of the config the run was created from.
        run_number: The dashboard's "Run #N" for this config, when available.
        results: All graded turn results for the run.
        progress: Per-evaluator progress from the backend, or ``None`` if the
            trailing fetch failed.
        risk_score: Risk-score summary from the backend, or ``None`` if the
            trailing fetch failed.
    """

    success: bool
    status: str
    run_id: str
    config_id: str
    results: list[RunResultItem] = field(default_factory=list)
    run_number: Optional[int] = None
    progress: Optional[dict[str, Any]] = None
    risk_score: Optional[dict[str, Any]] = None
