"""Enabling the instrumentations the SDK traces with.

``Netra.init()`` calls :func:`init_instrumentations` once, which decides *what*
to instrument and hands each instrumentation to
``netra.instrumentation.lazy``, which decides *when*.  The work is split across
four modules:

* ``instruments`` — every instrumentation the SDK knows about, and the defaults
* ``selection``   — requested/blocked sets to the instrumentations to enable
* ``registry``    — how to build each instrumentor Netra provides itself
* ``activation``  — applying one instrumentation, whoever implements it

``traceloop.sdk`` is never imported at module scope.  Importing it costs
~620 ms (it transitively pulls in pandas, aiohttp and numpy) and that cost
would land on every ``import netra``, including in processes that never call
``Netra.init()``.  Every traceloop symbol is imported inside the function that
needs it, so the cost is paid on first *activation* instead.
"""

import logging
import os
from typing import AbstractSet, Callable, Optional

from netra.instrumentation.activation import (
    SUBPROCESS_ACTIVATION,
    build_activations,
    run_activation,
)
from netra.instrumentation.instruments import NetraInstruments
from netra.instrumentation.lazy import register_lazy_instrumentations
from netra.instrumentation.selection import select_instrumentations

__all__ = ["init_instrumentations"]

logger = logging.getLogger(__name__)


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
    instruments: Optional[AbstractSet[NetraInstruments]] = None,
    block_instruments: Optional[AbstractSet[NetraInstruments]] = None,
) -> None:
    """Enable the requested instrumentations.

    Each enabled instrumentation is applied when the library it patches is
    first imported, which may be immediately if that library is already loaded.

    Args:
        should_enrich_metrics: Whether to enrich metrics.
        base64_image_uploader: Optional callback for image uploads.
        instruments: Instruments to enable.  ``None`` falls back to the curated
            default set; a set containing ``InstrumentSet.ALL`` enables every
            instrumentation available in the environment.
        block_instruments: Instruments to block.  A set containing
            ``InstrumentSet.ALL`` blocks every instrumentation.
    """
    selection = select_instrumentations(instruments, block_instruments)
    activations = build_activations(selection, should_enrich_metrics, base64_image_uploader)

    os.environ["TRACELOOP_TELEMETRY"] = "false"

    register_lazy_instrumentations(activations)

    # Subprocess instrumentation is always enabled: it propagates trace context
    # into subprocesses and is not tied to any third-party library.
    run_activation(SUBPROCESS_ACTIVATION)
