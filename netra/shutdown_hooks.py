"""Shared shutdown-hook registry for SIGINT/SIGTERM cleanup.

Lets independent parts of the SDK register cleanup callbacks that run before
the process terminates, sharing one signal handler per signal instead of each
caller installing its own.
"""

import concurrent.futures
import logging
import os
import signal
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "netra.shutdown_hooks"

# Max time to wait for hooks to finish before re-delivering the signal anyway.
SHUTDOWN_HOOK_TIMEOUT_S = 5.0

ShutdownHook = Callable[[], None]

_lock = threading.Lock()
_hooks: Dict[int, ShutdownHook] = {}
_next_token = 0
_installed_signals: Dict[int, Any] = {}
_running = False


def register_shutdown_hook(hook: ShutdownHook) -> int:
    """Register a callback to run on SIGINT/SIGTERM. Returns a token for
    :func:`unregister_shutdown_hook`."""
    global _next_token
    with _lock:
        token = _next_token
        _next_token += 1
        _hooks[token] = hook
        _ensure_signal_handlers_installed()
    return token


def unregister_shutdown_hook(token: int) -> None:
    """Remove a previously registered shutdown hook, if still present."""
    with _lock:
        _hooks.pop(token, None)


def run_shutdown_hooks() -> None:
    """Run every registered hook, bounded by ``SHUTDOWN_HOOK_TIMEOUT_S``. Re-entrancy guarded."""
    global _running
    with _lock:
        if _running:
            return
        _running = True
        hooks = list(_hooks.values())

    # Not `with ThreadPoolExecutor()`: that blocks on exit until every hook
    # finishes, defeating the timeout. `shutdown(wait=False)` lets slow hooks
    # keep running in the background instead.
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    try:
        if not hooks:
            return
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(hooks))
        futures = [executor.submit(_run_one_hook, hook) for hook in hooks]
        _done, not_done = concurrent.futures.wait(futures, timeout=SHUTDOWN_HOOK_TIMEOUT_S)
        if not_done:
            logger.warning(
                "%s: %d shutdown hook(s) did not finish within %.1fs",
                LOG_PREFIX,
                len(not_done),
                SHUTDOWN_HOOK_TIMEOUT_S,
            )
    finally:
        with _lock:
            _running = False
        if executor is not None:
            executor.shutdown(wait=False)


def _run_one_hook(hook: ShutdownHook) -> None:
    try:
        hook()
    except Exception:
        logger.error("%s: shutdown hook raised", LOG_PREFIX, exc_info=True)


def _ensure_signal_handlers_installed() -> None:
    """Install the shared SIGINT/SIGTERM handlers once. Must be called with
    ``_lock`` held. Skips silently if not on the main thread."""
    if _installed_signals:
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.signal(sig, _make_signal_handler(sig))
            _installed_signals[sig] = previous
        except ValueError:
            logger.debug(
                "%s: cannot install handler for %s outside the main thread; "
                "shutdown hooks will not run on this signal",
                LOG_PREFIX,
                sig.name,
            )


def _make_signal_handler(sig: signal.Signals) -> Callable[[int, object], None]:
    def _handler(signum: int, frame: object) -> None:
        try:
            run_shutdown_hooks()
        finally:
            previous = _installed_signals.get(sig)
            try:
                signal.signal(sig, previous if previous is not None else signal.SIG_DFL)
            except ValueError:
                pass
            os.kill(os.getpid(), signum)

    return _handler
