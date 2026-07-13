"""Pre/post script hooks for multi-turn simulation runs.

Hooks let you run setup and teardown logic around scenario execution
without merging all scenarios into one or relying on sequential ordering.

Hook levels:
    before_all  -- runs once before any scenario starts (dataset-level setup)
    before      -- runs before each individual scenario (item-level setup)
    after       -- runs after each individual scenario (item-level teardown)
    after_all   -- runs once after all scenarios complete (dataset-level teardown)

Context flow:
    before_all()        -> returns shared_context (dict | None)
    before(item, shared_context) -> returns item_context (dict | None)
    BaseTask.run(..., setup_context)  <- receives merged context
    after(item, result, shared_context)
    after_all(results, shared_context)

Failure semantics:
    - before_all failure  -> entire run is marked failed (prescript_failed), no scenarios run
    - before failure      -> that scenario is marked failed (prescript_failed), others continue
    - after / after_all failures are logged as warnings and do not affect run/scenario status
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Hook function type aliases
BeforeAllFn = Callable[[], Any | Awaitable[Any]]
BeforeFn = Callable[..., Any | Awaitable[Any]]
AfterFn = Callable[..., Any | Awaitable[Any]]
AfterAllFn = Callable[..., Any | Awaitable[Any]]


@dataclass
class SimulationHooks:
    """Container for lifecycle hook functions attached to a simulation run.

    All hooks are optional. When a hook is not provided the corresponding
    lifecycle phase is silently skipped.

    Attributes:
        before_all: Called once before any scenario starts. May return a
            ``dict`` that is forwarded to every ``before`` call and every
            ``BaseTask.run()`` call as ``setup_context``.
        before: Called before each individual scenario. Receives the
            ``run_item_id`` (str) and the ``shared_context`` returned by
            ``before_all``. May return a ``dict`` that is merged into
            ``shared_context`` and passed as ``setup_context`` to
            ``BaseTask.run()`` for that specific scenario.
        after: Called after each scenario completes (success or failure).
            Receives the ``run_item_id``, the result ``dict`` from the
            conversation loop, and ``shared_context``. Return value is ignored.
        after_all: Called once after all scenarios finish. Receives the
            aggregated results ``dict`` and ``shared_context``. Return value
            is ignored.

    Example::

        def setup():
            employee = create_employee()
            return {"employee_id": employee.id}

        def setup_item(run_item_id, shared_context):
            token = login(shared_context["employee_id"])
            return {"token": token}

        def teardown_item(run_item_id, result, shared_context):
            logout(shared_context.get("token"))

        def teardown(results, shared_context):
            delete_employee(shared_context["employee_id"])

        hooks = SimulationHooks(
            before_all=setup,
            before=setup_item,
            after=teardown_item,
            after_all=teardown,
        )
    """

    before_all: Optional[BeforeAllFn] = field(default=None)
    before: Optional[BeforeFn] = field(default=None)
    after: Optional[AfterFn] = field(default=None)
    after_all: Optional[AfterAllFn] = field(default=None)

    def describe(self) -> dict[str, Any]:
        """Return a metadata dict suitable for sending to the backend.

        The backend stores this so the UI can display which hooks are
        configured on a test run without storing the script itself.
        """
        def _desc(fn: Optional[Callable]) -> Optional[dict[str, Any]]:
            if fn is None:
                return None
            doc = inspect.getdoc(fn)
            return {
                "configured": True,
                "name": getattr(fn, "__name__", None),
                "description": doc[:200] if doc else None,
            }

        return {
            k: v
            for k, v in {
                "beforeAll": _desc(self.before_all),
                "before": _desc(self.before),
                "after": _desc(self.after),
                "afterAll": _desc(self.after_all),
            }.items()
            if v is not None
        }


async def _call_hook(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Invoke a hook function that may be sync or async.

    Args:
        fn: The hook callable.
        *args: Positional arguments to forward.
        **kwargs: Keyword arguments to forward.

    Returns:
        The hook's return value, or ``None`` if it returns nothing.
    """
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    return result


async def run_before_all(hooks: Optional[SimulationHooks]) -> Optional[dict]:
    """Execute the ``before_all`` hook and return the shared context.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.

    Returns:
        A ``dict`` shared context, or ``None`` if no hook is configured or
        the hook returned nothing.

    Raises:
        Exception: Re-raises any exception from the hook so the caller can
            mark the run as failed.
    """
    if hooks is None or hooks.before_all is None:
        return None

    logger.info("netra.simulation: running before_all hook")
    result = await _call_hook(hooks.before_all)
    if result is not None and not isinstance(result, dict):
        logger.warning(
            "netra.simulation: before_all returned %s (expected dict or None); ignoring value",
            type(result).__name__,
        )
        return None
    return result


async def run_before(
    hooks: Optional[SimulationHooks],
    run_item_id: str,
    shared_context: Optional[dict],
) -> Optional[dict]:
    """Execute the ``before`` hook for a single scenario.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        run_item_id: The identifier of the scenario being set up.
        shared_context: The dict returned by ``before_all``, or ``None``.

    Returns:
        A merged context dict (``shared_context`` + item-specific overrides),
        or ``shared_context`` unchanged when no ``before`` hook is configured.

    Raises:
        Exception: Re-raises any exception so the caller can mark the
            scenario as ``prescript_failed``.
    """
    if hooks is None or hooks.before is None:
        return shared_context

    logger.info("netra.simulation: running before hook for run_item_id=%s", run_item_id)
    result = await _call_hook(hooks.before, run_item_id, shared_context)

    base = dict(shared_context or {})
    if result is not None and isinstance(result, dict):
        base.update(result)
    elif result is not None:
        logger.warning(
            "netra.simulation: before hook returned %s (expected dict or None); ignoring value",
            type(result).__name__,
        )

    return base or None


async def run_after(
    hooks: Optional[SimulationHooks],
    run_item_id: str,
    item_result: dict,
    shared_context: Optional[dict],
) -> None:
    """Execute the ``after`` hook for a single scenario (best-effort).

    Exceptions are caught and logged; they do not affect the scenario status.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        run_item_id: The identifier of the scenario that just finished.
        item_result: The result dict from the conversation loop.
        shared_context: The dict returned by ``before_all``, or ``None``.
    """
    if hooks is None or hooks.after is None:
        return

    logger.info("netra.simulation: running after hook for run_item_id=%s", run_item_id)
    try:
        await _call_hook(hooks.after, run_item_id, item_result, shared_context)
    except Exception:
        logger.warning(
            "netra.simulation: after hook raised an exception for run_item_id=%s (ignored)",
            run_item_id,
            exc_info=True,
        )


async def run_after_all(
    hooks: Optional[SimulationHooks],
    results: dict,
    shared_context: Optional[dict],
) -> None:
    """Execute the ``after_all`` hook (best-effort).

    Exceptions are caught and logged; they do not affect the run status.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        results: The aggregated results dict from the simulation.
        shared_context: The dict returned by ``before_all``, or ``None``.
    """
    if hooks is None or hooks.after_all is None:
        return

    logger.info("netra.simulation: running after_all hook")
    try:
        await _call_hook(hooks.after_all, results, shared_context)
    except Exception:
        logger.warning(
            "netra.simulation: after_all hook raised an exception (ignored)",
            exc_info=True,
        )
