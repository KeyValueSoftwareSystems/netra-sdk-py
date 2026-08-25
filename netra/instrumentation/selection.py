"""Turns the caller's instrument sets into the instrumentations to enable.

Pure resolution: nothing is imported or applied here, and in particular
``traceloop.sdk`` is never reached — an explicit request is answered entirely
from :class:`InstrumentSet`, so selection costs nothing on any path.

The rules are inherited from traceloop's own ``init_instrumentations`` so that
deferring activation does not change *which* instrumentations end up enabled,
with one deliberate exception documented on :func:`_select_traceloop_names`.
"""

import logging
from dataclasses import dataclass
from typing import AbstractSet, Optional

from netra.instrumentation.instruments import (
    ALL_INSTRUMENTS,
    DEFAULT_INSTRUMENTS,
    InstrumentSet,
    NetraInstruments,
    _Origin,
)

logger = logging.getLogger(__name__)

# Traceloop instrumentors Netra replaces with its own implementation.  Letting
# traceloop install these too would double-instrument the same call sites.
#
# Held as names, not enum members: naming a member means importing
# ``traceloop.sdk``, which costs ~620 ms and is what deferred activation exists
# to avoid.  A name the installed traceloop-sdk does not define simply never
# matches.
TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA: frozenset[str] = frozenset(
    {
        "AGNO",
        "COHERE",
        "GOOGLE_GENERATIVEAI",
        "GROQ",
        "MISTRAL",
        "OPENAI",
        "PYMYSQL",
        "QDRANT",
        "REDIS",
        "REQUESTS",
        "URLLIB3",
        "WEAVIATE",
    }
)


@dataclass(frozen=True)
class InstrumentationSelection:
    """The instrumentations to enable, split by who implements them.

    Attributes:
        traceloop_instrument_names: Names of ``traceloop.sdk.Instruments``
            members to enable.  Names rather than members so that resolving
            them — and importing traceloop — can wait until activation.
        custom_instruments: Instrumentations Netra applies itself.
    """

    traceloop_instrument_names: frozenset[str]
    custom_instruments: frozenset[InstrumentSet]


NOTHING_SELECTED = InstrumentationSelection(frozenset(), frozenset())


def select_instrumentations(
    requested: Optional[AbstractSet[NetraInstruments]],
    blocked: Optional[AbstractSet[NetraInstruments]],
) -> InstrumentationSelection:
    """Resolve requested and blocked instrument sets into what to enable.

    Args:
        requested: Instruments to enable.  ``None`` or empty falls back to
            :data:`DEFAULT_INSTRUMENTS`; a set containing ``InstrumentSet.ALL``
            enables every instrumentation the SDK knows about.
        blocked: Instruments to block.  A set containing ``InstrumentSet.ALL``
            blocks everything, whatever *requested* asks for.

    Returns:
        The instrumentations to enable, split by implementing family.
    """
    if contains_all_sentinel(blocked):
        return NOTHING_SELECTED

    enable_everything = contains_all_sentinel(requested)
    selected = ALL_INSTRUMENTS if enable_everything else (requested or DEFAULT_INSTRUMENTS)

    requested_traceloop, requested_custom = partition_by_origin(selected)
    blocked_traceloop, blocked_custom = partition_by_origin(blocked or frozenset())

    return InstrumentationSelection(
        traceloop_instrument_names=_select_traceloop_names(requested_traceloop, blocked_traceloop),
        # Neither family has an "empty means everything" fallback: a request
        # naming no instrumentation of a family enables none of that family.
        custom_instruments=frozenset(requested_custom - blocked_custom),
    )


def contains_all_sentinel(instruments: Optional[AbstractSet[NetraInstruments]]) -> bool:
    """Report whether a set contains the ``InstrumentSet.ALL`` sentinel.

    Args:
        instruments: The set to check, or ``None``.

    Returns:
        True if the set contains ``InstrumentSet.ALL``.
    """
    return instruments is not None and InstrumentSet.ALL in instruments


def partition_by_origin(
    instruments: AbstractSet[NetraInstruments],
) -> tuple[set[str], set[InstrumentSet]]:
    """Split instruments into the traceloop-backed and Netra-backed families.

    The ``ALL`` sentinel belongs to neither family and is skipped.

    Traceloop instruments come back as enum member *names*: turning a name into
    a ``traceloop.sdk.Instruments`` member means importing traceloop, and this
    runs during ``Netra.init()``, where that import is exactly what deferred
    activation exists to avoid.

    Args:
        instruments: The instruments to split.

    Returns:
        A ``(traceloop_names, custom_instruments)`` pair.
    """
    traceloop_names: set[str] = set()
    custom_instruments: set[InstrumentSet] = set()
    for instrument in instruments:
        if instrument.origin is _Origin.TRACELOOP:
            traceloop_names.add(instrument.name)
        elif instrument.origin is _Origin.CUSTOM:
            custom_instruments.add(instrument)
    return traceloop_names, custom_instruments


def _select_traceloop_names(requested: set[str], blocked: set[str]) -> frozenset[str]:
    """Reduce a requested/blocked pair to the traceloop instruments to enable.

    What the caller asked for, minus what they blocked, minus the instruments
    Netra implements itself.  ``requested`` already accounts for the ``None``
    and ``ALL`` cases: :func:`select_instrumentations` expands those to
    :data:`DEFAULT_INSTRUMENTS` and :data:`ALL_INSTRUMENTS` before partitioning,
    so an empty ``requested`` here means the caller named instruments and none
    of them were traceloop-backed.

    **Deliberate behaviour change.**  Previously an empty ``requested`` fell
    through to "every traceloop instrument the environment has", mirroring
    traceloop's own ``instruments=None``.  Because ``None``/``ALL`` are already
    expanded above, that fallback was only ever reachable one way: a caller who
    named Netra-backed instruments *and* blocked at least one traceloop one.
    ``Netra.init(instruments={InstrumentSet.OPENAI},
    block_instruments={InstrumentSet.ANTHROPIC})`` therefore enabled langchain,
    bedrock, vertexai and the rest — the opposite of what it reads as, and a
    direct contradiction of the first rule above.  Blocking one instrument now
    never enables another.

    Args:
        requested: Names of the traceloop instruments the caller asked for.
        blocked: Names of the traceloop instruments the caller blocked.

    Returns:
        Names of the traceloop instruments to enable.
    """
    return frozenset(requested - blocked - TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA)
