"""Defers each instrumentation until the library it patches is first imported.

Instrumenting a library means importing it, and importing an LLM library is
expensive: eagerly instrumenting the default set costs ~3 s in an environment
with common LLM libraries installed, whether or not the process uses any of
them.  Registering a post-import hook per instrumentation moves that cost to
the client's own ``import`` statement, and skips it entirely for libraries the
process never imports.

The invariant this module preserves:

    An enabled instrumentation is applied **exactly once**, **no earlier than**
    ``Netra.init()``, and **no later than** the completion of the first import
    of the library it patches.

Nothing can call into a library before its import returns, and wrapt runs a
post-import hook inside that import — so deferring is behaviour-preserving.
wrapt also fires a hook **immediately and synchronously** when the target
module is already in ``sys.modules``, which is what makes this correct whether
the client imports their library before or after ``Netra.init()``.

Accepted limitations:

1. **Objects created during the library's own module execution are not
   patched.**  A post-import hook runs after the target module finishes
   executing, so a module-level singleton built at import time keeps an
   unpatched bound method.  This limitation already exists today for any client
   who imports their library before calling ``Netra.init()``; deferring does
   not widen it, but it becomes the common path rather than the rare one.
2. **A library loaded by a non-standard loader may not fire wrapt's hook.**
   Frozen imports, zipimport and some vendored loaders are untested; coverage
   would be lost silently for such a library.
3. **Trigger table drift.**  An instrumentation with no entry in
   ``INSTRUMENT_TRIGGERS`` is applied immediately, so it is slow rather than
   broken.
4. **Activation cost moves into the client's import statement.**  ``import
   openai`` becomes ~350 ms slower in a process that will use OpenAI.  Total
   work is strictly lower; only its position changes.
"""

import logging
import threading
from collections import defaultdict
from typing import Callable, Sequence

from netra.instrumentation.activation import Activation, run_activation
from netra.instrumentation.triggers import INSTRUMENT_TRIGGERS

logger = logging.getLogger(__name__)

# Trigger lookup is by member *name* rather than by ``InstrumentSet`` member,
# because a traceloop instrumentation is carried through activation as the name
# of a ``traceloop.sdk.Instruments`` member — naming the member itself would
# mean importing traceloop.  The names always agree.
_TRIGGERS_BY_NAME: dict[str, tuple[str, ...]] = {
    instrument.name: triggers for instrument, triggers in INSTRUMENT_TRIGGERS.items()
}


class _ActivationLedger:
    """Records which instrumentations have been claimed for activation.

    One instrumentation can be reachable from several trigger modules, and a
    trigger module can be imported concurrently by several threads, so the
    claim has to be atomic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claimed: set[str] = set()

    def claim(self, name: str) -> bool:
        """Claim *name* for activation.

        Args:
            name: The instrument's enum member name.

        Returns:
            True the first time *name* is claimed, False every time after.
        """
        with self._lock:
            if name in self._claimed:
                return False
            self._claimed.add(name)
            return True


def register_lazy_instrumentations(activations: Sequence[Activation]) -> None:
    """Register *activations* to run when the libraries they patch are imported.

    An activation whose instrument has no trigger mapping runs immediately, so
    an incomplete trigger table degrades to eager activation — slow, not
    silently untraced.

    Args:
        activations: The instrumentations to apply, in activation order.
    """
    import wrapt

    ledger = _ActivationLedger()
    activations_by_trigger: dict[str, list[Activation]] = defaultdict(list)

    for activation in activations:
        triggers = _TRIGGERS_BY_NAME.get(activation.name)
        if not triggers:
            logger.debug("No trigger module for %s; activating eagerly", activation.name)
            _activate_once(activation, ledger)
            continue
        for trigger in triggers:
            activations_by_trigger[trigger].append(activation)

    for trigger, triggered in activations_by_trigger.items():
        wrapt.register_post_import_hook(_post_import_hook(triggered, ledger), trigger)


def _post_import_hook(activations: Sequence[Activation], ledger: _ActivationLedger) -> Callable[[object], None]:
    """Build the post-import hook that applies *activations*.

    Args:
        activations: Instrumentations triggered by the hooked module.
        ledger: Shared ledger, so an instrumentation reachable from several
            triggers is applied only once.

    Returns:
        A callable matching wrapt's post-import hook signature.
    """

    def hook(module: object) -> None:
        for activation in activations:
            _activate_once(activation, ledger)

    return hook


def _activate_once(activation: Activation, ledger: _ActivationLedger) -> None:
    """Apply *activation* unless it has already been claimed.

    Args:
        activation: The instrumentation to apply.
        ledger: Ledger of instrumentations already claimed.
    """
    # Claimed under the ledger's lock, applied outside it.  A hook already runs
    # while an import lock is held and activation itself imports modules, so
    # holding a second lock across that import is the deadlock-shaped risk.
    if ledger.claim(activation.name):
        run_activation(activation)
