"""Pre/post script hooks for multi-turn simulation runs.

Hooks let you run setup and teardown logic around scenario execution
without merging all scenarios into one or relying on sequential ordering.

Hook levels:
    before_all   -- runs once before any scenario starts (dataset-level setup)
    before       -- runs before specific scenarios only (item-specific setup, keyed by dataset_item_id)
    after        -- runs after specific scenarios only (item-specific teardown, keyed by dataset_item_id)
    after_all    -- runs once after all scenarios complete (dataset-level teardown)

Execution order per item:
    before_all()                           -> returns shared_context (dict | None)
    before[dataset_item_id](shared_context) -> returns item_context (dict | None), if registered
    BaseTask.run(..., setup_context)       <- receives merged context
    after[dataset_item_id](result, shared_context), if registered
    after_all(results, shared_context)

Failure semantics:
    - before_all failure -> entire run is marked failed (prescript_failed), no scenarios run
    - before failure     -> that scenario is marked failed (prescript_failed), others continue
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
            ``dict`` that is forwarded to ``before`` hooks and every
            ``BaseTask.run()`` call as ``setup_context``.
        before: Dict mapping ``dataset_item_id`` to hook functions. Each function
            receives ``shared_context`` and may return a ``dict`` that is merged
            with ``shared_context`` and passed as ``setup_context`` to
            ``BaseTask.run()`` for that specific scenario. Only called for
            specific items that have registered hooks.
        after: Dict mapping ``dataset_item_id`` to hook functions. Each function
            receives the result ``dict`` from the conversation loop and
            ``shared_context``. Only called for specific items that have
            registered hooks. Return value is ignored.
        after_all: Called once after all scenarios finish. Receives the
            aggregated results ``dict`` and ``shared_context``. Return value
            is ignored.

    Example::

        def setup():
            employee = create_employee()
            return {"employee_id": employee.id}
            
        def setup_refund_item(shared_context):
            # Only for refund scenario
            token = login(shared_context["employee_id"])
            return {"refund_account": "12345", "token": token}
            
        def teardown_refund_item(result, shared_context):
            # Cleanup only for refund scenario
            logout(shared_context.get("token"))
            cleanup_refund(shared_context.get("refund_account"))

        def teardown(results, shared_context):
            delete_employee(shared_context["employee_id"])

        hooks = SimulationHooks(
            before_all=setup,
            before={"refund-scenario-id": setup_refund_item},
            after={"refund-scenario-id": teardown_refund_item},
            after_all=teardown,
        )
    """

    before_all: Optional[BeforeAllFn] = field(default=None)
    before: Optional[dict[str, BeforeFn]] = field(default=None)
    after: Optional[dict[str, AfterFn]] = field(default=None)
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
        
        def _desc_dict(hook_dict: Optional[dict[str, Callable]]) -> Optional[dict[str, Any]]:
            """Summarize item-keyed hooks for backend UI metadata.

            Local execution still uses the per-item dict. The create-run API
            currently validates ``before`` / ``after`` as a single descriptor
            object (not a map of item IDs), so we flatten for wire format.
            """
            if not hook_dict:
                return None
            first_fn = next(iter(hook_dict.values()))
            doc = inspect.getdoc(first_fn)
            item_ids = list(hook_dict.keys())
            base_desc = (doc[:160] if doc else "") or ""
            suffix = f" ({len(item_ids)} item(s))"
            description = (base_desc + suffix)[:200] if base_desc or item_ids else None
            return {
                "configured": True,
                "name": getattr(first_fn, "__name__", None),
                "description": description,
            }

        return {
            k: v
            for k, v in {
                "beforeAll": _desc(self.before_all),
                "before": _desc_dict(self.before),
                "after": _desc_dict(self.after),
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
    dataset_item_id: str,
    shared_context: Optional[dict],
) -> Optional[dict]:
    """Execute the item-specific ``before`` hook for a single scenario.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        shared_context: The dict returned by ``before_all``, or ``None``.

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


async def run_after(
    hooks: Optional[SimulationHooks],
    dataset_item_id: str,
    item_result: dict,
    shared_context: Optional[dict],
) -> None:
    """Execute the item-specific ``after`` hook for a single scenario (best-effort).

    Exceptions are caught and logged; they do not affect the scenario status.

    Args:
        hooks: The :class:`SimulationHooks` instance, or ``None``.
        dataset_item_id: The stable identifier from the dataset item.
        item_result: The result dict from the conversation loop.
        shared_context: The dict returned by ``before_all``, or ``None``.
    """
    # Execute item-specific after hook (only if registered for this item)
    if hooks and hooks.after and dataset_item_id in hooks.after:
        logger.info("netra.simulation: running after hook for dataset_item_id=%s", dataset_item_id)
        try:
            item_hook = hooks.after[dataset_item_id]
            await _call_hook(item_hook, item_result, shared_context)
        except Exception:
            logger.warning(
                "netra.simulation: after hook raised an exception for dataset_item_id=%s (ignored)",
                dataset_item_id,
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
