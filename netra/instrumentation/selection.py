"""Turns the caller's instrument sets into the instrumentations to enable.

Pure resolution: no instrumentor is imported or applied here.  The rules are
inherited from traceloop's own ``init_instrumentations`` so that deferring
activation cannot change *which* instrumentations end up enabled.
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
        traceloop_instrument_names=_select_traceloop_names(
            requested_traceloop, blocked_traceloop, enable_everything=enable_everything
        ),
        # Unlike traceloop's, this family has no "empty means everything"
        # fallback: an empty request enables no Netra instrumentation.
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


def _select_traceloop_names(
    requested: set[str],
    blocked: set[str],
    *,
    enable_everything: bool,
) -> frozenset[str]:
    """Reduce a requested/blocked pair to the traceloop instruments to enable.

    Two inherited rules are preserved here because changing either would
    change which instrumentations a released version enables:

    * An explicit selection naming no traceloop instrument enables none of
      them, rather than falling through to "all of them".
    * An empty request that *does* come with a block list means "every
      traceloop instrument except the blocked ones" — this mirrors traceloop's
      own ``instruments=None``.

    Args:
        requested: Names of the traceloop instruments the caller asked for.
        blocked: Names of the traceloop instruments the caller blocked.
        enable_everything: Whether the caller passed ``InstrumentSet.ALL``.

    Returns:
        Names of the traceloop instruments to enable.
    """
    if not requested and not blocked and not enable_everything:
        return frozenset()

    enabled = (requested or _installed_traceloop_instrument_names()) - blocked
    return frozenset(enabled - TRACELOOP_INSTRUMENTS_REPLACED_BY_NETRA)


def _installed_traceloop_instrument_names() -> set[str]:
    """Return every instrument name the installed traceloop-sdk offers.

    The only branch of selection that needs traceloop's member list, and so
    the only one that pays for importing it.
    """
    from traceloop.sdk.instruments import Instruments

    return {member.name for member in Instruments}
