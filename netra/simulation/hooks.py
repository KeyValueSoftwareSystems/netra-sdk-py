"""Pre/post script hooks for multi-turn simulation runs.

Hooks let you run setup and teardown logic around scenario execution
without merging all scenarios into one or relying on sequential ordering.

Hook levels:
    before_all   -- runs once before any scenario starts (dataset-level setup)
    before_each  -- runs before every scenario (common per-item setup)
    before       -- runs before specific scenarios only (item-specific setup, keyed by dataset_item_id)
    after        -- runs after specific scenarios only (item-specific teardown, keyed by dataset_item_id)
    after_each   -- runs after every scenario (common per-item teardown)
    after_all    -- runs once after all scenarios complete (dataset-level teardown)

Execution order per item:
    before_all()                           -> returns shared_context (dict | None)
    before_each(shared_context)            -> returns each_context (dict | None)
    before[dataset_item_id](merged_context) -> returns item_context (dict | None), if registered
    BaseTask.run(..., setup_context)       <- receives merged context
    after[dataset_item_id](result, setup_context), if registered
    after_each(result, setup_context)
    after_all(results, shared_context)

Failure semantics:
    - before_all failure  -> entire run is marked failed (prescript_failed), no scenarios run
    - before_each failure -> that scenario is marked failed (prescript_failed), others continue
    - before failure      -> that scenario is marked failed (prescript_failed), others continue
    - after / after_each / after_all failures are logged as warnings and do not affect run/scenario status
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Hook function type aliases
BeforeAllFn = Callable[[], Any | Awaitable[Any]]
BeforeEachFn = Callable[..., Any | Awaitable[Any]]
BeforeFn = Callable[..., Any | Awaitable[Any]]
AfterFn = Callable[..., Any | Awaitable[Any]]
AfterEachFn = Callable[..., Any | Awaitable[Any]]
AfterAllFn = Callable[..., Any | Awaitable[Any]]


@dataclass
class SimulationHooks:
    """Container for lifecycle hook functions attached to a simulation run.

    All hooks are optional. When a hook is not provided the corresponding
    lifecycle phase is silently skipped.

    Attributes:
        before_all: Called once before any scenario starts. May return a
            ``dict`` that is forwarded to ``before_each``/``before`` hooks and
            every ``BaseTask.run()`` call as ``setup_context``.
        before_each: Called before every scenario. Receives ``shared_context``
            and may return a ``dict`` that is merged with ``shared_context``
            before being passed to the item-specific ``before`` hook (if any)
            and ultimately to ``BaseTask.run()`` as ``setup_context``.
        before: Dict mapping ``dataset_item_id`` to hook functions. Each function
            receives the merged context (from ``before_all`` + ``before_each``)
            and may return a ``dict`` that is merged into ``setup_context``
            for that specific scenario. Only called for specific items that
            have registered hooks.
        after: Dict mapping ``dataset_item_id`` to hook functions. Each function
            receives the result ``dict`` from the conversation loop and
            ``setup_context``. Only called for specific items that have
            registered hooks. Return value is ignored.
        after_each: Called after every scenario. Receives the item result and
            ``setup_context``. Return value is ignored.
        after_all: Called once after all scenarios finish. Receives the
            aggregated results ``dict`` and ``shared_context``. Return value
            is ignored.

    Example::

        def setup():
            employee = create_employee()
            return {"employee_id": employee.id}

        def setup_each(shared_context):
            # Runs before every scenario
            token = get_fresh_token()
            return {"auth_token": token}

        def setup_refund_item(shared_context):
            # Only for refund scenario
            token = login(shared_context["employee_id"])
            return {"refund_account": "12345", "token": token}

        def teardown_refund_item(result, setup_context):
            # Cleanup only for refund scenario
            logout(setup_context.get("token"))
            cleanup_refund(setup_context.get("refund_account"))

        def teardown_each(result, setup_context):
            # Runs after every scenario
            invalidate_token(setup_context.get("auth_token"))

        def teardown(results, shared_context):
            delete_employee(shared_context["employee_id"])

        hooks = SimulationHooks(
            before_all=setup,
            before_each=setup_each,
            before={"refund-scenario-id": setup_refund_item},
            after={"refund-scenario-id": teardown_refund_item},
            after_each=teardown_each,
            after_all=teardown,
        )
    """

    before_all: Optional[BeforeAllFn] = field(default=None)
    before_each: Optional[BeforeEachFn] = field(default=None)
    before: Optional[dict[str, BeforeFn]] = field(default=None)
    after: Optional[dict[str, AfterFn]] = field(default=None)
    after_each: Optional[AfterEachFn] = field(default=None)
    after_all: Optional[AfterAllFn] = field(default=None)

    def describe(self) -> dict[str, Any]:
        """Return a metadata dict suitable for sending to the backend.

        Run-level hooks (``beforeAll`` / ``beforeEach`` / ``afterEach`` / ``afterAll``)
        are stored on the test run. Item-level hooks (``before`` / ``after``) are
        sent under ``items`` keyed by ``datasetItemId`` and stored on each matching
        test run item.
        """

        def _desc(fn: Optional[Callable[..., Any]]) -> Optional[dict[str, Any]]:
            if fn is None:
                return None
            doc = inspect.getdoc(fn)
            return {
                "configured": True,
                "name": getattr(fn, "__name__", None),
                "description": doc[:200] if doc else None,
            }

        payload: dict[str, Any] = {}
        before_all = _desc(self.before_all)
        before_each = _desc(self.before_each)
        after_each = _desc(self.after_each)
        after_all = _desc(self.after_all)
        if before_all:
            payload["beforeAll"] = before_all
        if before_each:
            payload["beforeEach"] = before_each
        if after_each:
            payload["afterEach"] = after_each
        if after_all:
            payload["afterAll"] = after_all

        item_ids = set(self.before or {}) | set(self.after or {})
        if item_ids:
            items: list[dict[str, Any]] = []
            for item_id in item_ids:
                entry: dict[str, Any] = {"datasetItemId": item_id}
                before_fn = (self.before or {}).get(item_id)
                after_fn = (self.after or {}).get(item_id)
                before_desc = _desc(before_fn)
                after_desc = _desc(after_fn)
                if before_desc:
                    entry["before"] = before_desc
                if after_desc:
                    entry["after"] = after_desc
                if "before" in entry or "after" in entry:
                    items.append(entry)
            if items:
                payload["items"] = items

        return payload


async def _call_hook(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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


async def run_before_all(hooks: Optional[SimulationHooks]) -> Optional[dict[str, Any]]:
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
    dataset_item_id: str,
    shared_context: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Execute the item-specific ``before`` hook for a single scenario.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        shared_context: The dict returned by ``before_all`` (possibly merged
            with ``before_each`` output), or ``None``.

    Returns:
        A merged context dict (``shared_context`` + item-specific ``before`` result),
        or ``shared_context`` unchanged when no hook is registered for this item.

    Raises:
        Exception: Re-raises any exception so the caller can mark the
            scenario as ``prescript_failed``.
    """
    # Execute item-specific before hook (only if registered for this item)
    if hooks and hooks.before and dataset_item_id in hooks.before:
        logger.info("netra.simulation: running before hook for dataset_item_id=%s", dataset_item_id)
        item_hook = hooks.before[dataset_item_id]
        result = await _call_hook(item_hook, shared_context)

        base = dict(shared_context or {})
        if result is not None and isinstance(result, dict):
            base.update(result)
        elif result is not None:
            logger.warning(
                "netra.simulation: before hook returned %s (expected dict or None); ignoring value",
                type(result).__name__,
            )
        return base or None

    return shared_context


async def run_before_each(
    hooks: Optional[SimulationHooks],
    dataset_item_id: str,
    shared_context: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Execute the ``before_each`` hook for a single scenario.

    Unlike ``before`` (which is item-specific), ``before_each`` runs for
    every dataset item.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        shared_context: The dict returned by ``before_all``, or ``None``.

    Returns:
        A merged context dict (``shared_context`` + ``before_each`` result),
        or ``shared_context`` unchanged when no hook is configured.

    Raises:
        Exception: Re-raises any exception so the caller can mark the
            scenario as ``prescript_failed``.
    """
    if hooks is None or hooks.before_each is None:
        return shared_context

    logger.info("netra.simulation: running before_each hook for dataset_item_id=%s", dataset_item_id)
    result = await _call_hook(hooks.before_each, shared_context)

    base = dict(shared_context or {})
    if result is not None and isinstance(result, dict):
        base.update(result)
    elif result is not None:
        logger.warning(
            "netra.simulation: before_each hook returned %s (expected dict or None); ignoring value",
            type(result).__name__,
        )
    return base or None


async def run_after(
    hooks: Optional[SimulationHooks],
    dataset_item_id: str,
    item_result: dict[str, Any],
    setup_context: Optional[dict[str, Any]],
) -> None:
    """Execute the item-specific ``after`` hook for a single scenario (best-effort).

    The ``after`` hook is called regardless of whether the scenario succeeded,
    failed, or had its ``before`` hook fail. This ensures cleanup logic runs
    even when setup fails. Exceptions are caught and logged; they do not affect
    the scenario status.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        item_result: The result dict from the conversation loop (or error result
            if ``before`` failed).
        setup_context: Merged context from ``before_all`` + ``before_each`` + item ``before``
            (same dict passed to ``BaseTask.run``). When a before hook fails,
            this is the furthest successfully built context (e.g. ``before_all``
            only, or ``before_all`` + ``before_each`` if ``before`` failed).
    """
    # Execute item-specific after hook (only if registered for this item)
    if hooks and hooks.after and dataset_item_id in hooks.after:
        logger.info("netra.simulation: running after hook for dataset_item_id=%s", dataset_item_id)
        try:
            item_hook = hooks.after[dataset_item_id]
            await _call_hook(item_hook, item_result, setup_context)
        except Exception:
            logger.warning(
                "netra.simulation: after hook raised an exception for dataset_item_id=%s (ignored)",
                dataset_item_id,
                exc_info=True,
            )


async def run_after_each(
    hooks: Optional[SimulationHooks],
    dataset_item_id: str,
    item_result: dict[str, Any],
    setup_context: Optional[dict[str, Any]],
) -> None:
    """Execute the ``after_each`` hook for a single scenario (best-effort).

    Unlike ``after`` (which is item-specific), ``after_each`` runs for every
    dataset item. Exceptions are caught and logged; they do not affect the
    scenario status.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        item_result: The result dict from the conversation loop.
        setup_context: Merged context passed to ``BaseTask.run``.
    """
    if hooks is None or hooks.after_each is None:
        return

    logger.info("netra.simulation: running after_each hook for dataset_item_id=%s", dataset_item_id)
    try:
        await _call_hook(hooks.after_each, item_result, setup_context)
    except Exception:
        logger.warning(
            "netra.simulation: after_each hook raised an exception for dataset_item_id=%s (ignored)",
            dataset_item_id,
            exc_info=True,
        )


async def run_after_all(
    hooks: Optional[SimulationHooks],
    results: dict[str, Any],
    shared_context: Optional[dict[str, Any]],
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
