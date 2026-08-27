"""Enabling the instrumentations the SDK traces with.

``Netra.init()`` calls :func:`init_instrumentations` once, which decides *what*
to instrument and hands each instrumentation to ``wiring.deferral``, which
decides *when*.

The package is laid out in four parts:

* ``instruments``  — every instrumentation the SDK knows about, and the defaults.
  Kept at this level because it is public: callers name ``InstrumentSet`` members
  when they pass ``instruments=`` to ``Netra.init()``.
* ``wiring/``      — resolving the requested set and applying it: ``selection``,
  ``registry``, ``triggers``, ``activation``, ``deferral``.
* ``capture/``     — bounding and shaping recorded values: ``bounded_capture``,
  ``stream_formats``, ``stream_utils``.
* ``http/``        — what the HTTP instrumentations share: ``headers``, ``body``.

Everything else here is one directory per instrumented library, each following
the layout in CLAUDE.md: ``__init__.py`` holds the ``BaseInstrumentor``
subclass, ``wrappers.py`` the wrapper factories, ``utils.py`` the attribute
extraction, ``version.py`` the pinned library version.

``traceloop.sdk`` is never imported at module scope.  Importing it costs
~620 ms (it transitively pulls in pandas, aiohttp and numpy) and that cost
would land on every ``import netra``, including in processes that never call
``Netra.init()``.  Every traceloop symbol is imported inside the function that
needs it, so the cost is paid on first *activation* instead.
"""

import logging
import os
from typing import AbstractSet, Optional

# Re-exported for import-path compatibility: these four were reachable as
# ``from netra.instrumentation import ...`` before activation was split out of
# this module, and nothing about that split needed to break it.  The supported
# public path remains ``from netra import NetraInstruments``.
from netra.instrumentation.instruments import (
    DEFAULT_INSTRUMENTS,
    CustomInstruments,
    InstrumentSet,
    NetraInstruments,
)
from netra.instrumentation.wiring.activation import (
    SUBPROCESS_ACTIVATION,
    build_activations,
    run_activation,
)
from netra.instrumentation.wiring.deferral import register_lazy_instrumentations
from netra.instrumentation.wiring.selection import select_instrumentations

__all__ = [
    "CustomInstruments",
    "DEFAULT_INSTRUMENTS",
    "InstrumentSet",
    "NetraInstruments",
    "init_instrumentations",
]

logger = logging.getLogger(__name__)


def init_instrumentations(
    should_enrich_metrics: bool,
    instruments: Optional[AbstractSet[NetraInstruments]] = None,
    block_instruments: Optional[AbstractSet[NetraInstruments]] = None,
) -> None:
    """Enable the requested instrumentations.

    Each enabled instrumentation is applied when the library it patches is
    first imported, which may be immediately if that library is already loaded.

    Args:
        should_enrich_metrics: Whether to enrich metrics.
        instruments: Instruments to enable.  ``None`` falls back to the curated
            default set; a set containing ``InstrumentSet.ALL`` enables every
            instrumentation available in the environment.
        block_instruments: Instruments to block.  A set containing
            ``InstrumentSet.ALL`` blocks every instrumentation.
    """
    selection = select_instrumentations(instruments, block_instruments)
    activations = build_activations(selection, should_enrich_metrics)

    os.environ["TRACELOOP_TELEMETRY"] = "false"

    register_lazy_instrumentations(activations)

    # Subprocess instrumentation is always enabled: it propagates trace context
    # into subprocesses and is not tied to any third-party library.
    run_activation(SUBPROCESS_ACTIVATION)
