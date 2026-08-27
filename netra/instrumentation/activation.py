"""Applies one instrumentation, whoever implements it.

Every instrumentation — Netra's own or one delegated to traceloop — is wrapped
in an :class:`Activation`: a name plus a callable that applies it.  That single
shape is what lets ``netra.instrumentation.deferred_activation`` defer activation to the first
import of the target library without knowing anything about instrumentors.

No instrumentor module is imported until its instrumentation is actually
applied — importing an instrumentor imports the library it patches, and
deferring that import is the point of the whole arrangement.
"""

import logging
import re
import sys
import threading
from contextlib import contextmanager
from functools import lru_cache, partial
from importlib import import_module
from importlib.metadata import distributions
from io import StringIO
from typing import TYPE_CHECKING, AbstractSet, Callable, Iterator, NamedTuple, Optional, TextIO

from netra.instrumentation.instruments import InstrumentSet
from netra.instrumentation.registry import CUSTOM_INSTRUMENTORS, SUBPROCESS_INSTRUMENTOR, InstrumentorSpec
from netra.instrumentation.selection import InstrumentationSelection

if TYPE_CHECKING:
    # Type-only: importing traceloop.sdk at runtime costs ~620 ms.
    from traceloop.sdk.instruments import Instruments

logger = logging.getLogger(__name__)

# Guards the depth counter below — never held across an activation, only across
# the counter update.  Holding a lock while an instrumentor imports its library
# would deadlock against the import lock a post-import hook already runs under.
_output_suppression_lock = threading.Lock()
_output_suppression_depth = 0
_suppressed_streams: Optional[tuple[TextIO, TextIO]] = None


class Activation(NamedTuple):
    """One instrumentation and the callable that applies it.

    Attributes:
        name: The instrument's enum member name.  Used to look up its trigger
            modules and to identify it in logs.
        run: Applies the instrumentation.  Called at most once, and allowed to
            raise — :func:`run_activation` is what contains the failure.
    """

    name: str
    run: Callable[[], None]


def build_activations(selection: InstrumentationSelection, should_enrich_metrics: bool) -> list[Activation]:
    """Build one activation per selected instrumentation, in registration order.

    Traceloop instrumentations come first, in name order, then Netra's own in
    registry order, so the list does not depend on set iteration.  That fixes
    the order activations are *registered* in, and the order they are applied
    in on the eager fallback path and within a single post-import hook.  It
    does not fix the order across hooks: once activation is deferred, two
    instrumentations with different trigger modules are applied in whatever
    order the client imports those modules.  No instrumentation depends on
    another having been applied first.

    Args:
        selection: The instrumentations to enable.
        should_enrich_metrics: Whether to enrich metrics.

    Returns:
        The activations, in registration order.
    """
    activations = [
        Activation(name, partial(apply_traceloop_instrumentation, name, should_enrich_metrics))
        for name in sorted(selection.traceloop_instrument_names)
    ]
    activations.extend(
        Activation(instrument.name, partial(apply_custom_instrumentation, instrument))
        for instrument in CUSTOM_INSTRUMENTORS
        if instrument in selection.custom_instruments
    )

    unregistered = selection.custom_instruments - CUSTOM_INSTRUMENTORS.keys()
    if unregistered:
        # Selectable but not implemented: enabling one is a no-op, not an error.
        # A caller who named one deserves to hear about it; an InstrumentSet.ALL
        # expansion sweeps in six of them every time, so that stays at debug.
        log = logger.warning if selection.instruments_were_named_by_caller else logger.debug
        log("No instrumentor registered for: %s", ", ".join(sorted(i.name for i in unregistered)))

    return activations


def run_activation(activation: Activation) -> None:
    """Apply one instrumentation, containing any failure it raises.

    Args:
        activation: The instrumentation to apply.
    """
    try:
        activation.run()
    except Exception:
        # Deliberately broad, and the only place instrumentation failures are
        # swallowed.  Deferred activation runs inside the client's own
        # ``import`` statement, where a third-party instrumentor failing to
        # patch must not break that import; and one instrumentor failing must
        # not cost the client every other instrumentation.
        logger.exception("Failed to activate instrumentation: %s", activation.name)


def apply_custom_instrumentation(instrument: InstrumentSet) -> None:
    """Apply the instrumentor Netra provides for *instrument*.

    Candidates are tried in registry order and the first whose distributions
    are all installed is applied, which is how a library published under more
    than one distribution name is handled.  An instrumentation whose library is
    not installed is not an error: nothing is applied.

    Args:
        instrument: The instrumentation to apply.
    """
    for spec in CUSTOM_INSTRUMENTORS.get(instrument, ()):
        if all(is_distribution_installed(name) for name in spec.required_distributions):
            _apply_instrumentor(spec)
            return

    logger.debug("No installed distribution to instrument for: %s", instrument.name)


def apply_traceloop_instrumentation(name: str, should_enrich_metrics: bool) -> None:
    """Apply a single traceloop instrumentation by enum member name.

    Args:
        name: Traceloop ``Instruments`` member name.
        should_enrich_metrics: Whether to enrich metrics.
    """
    from traceloop.sdk.tracing.tracing import init_instrumentations as apply_traceloop_instruments

    instruments = _resolve_traceloop_instruments({name})
    if not instruments:
        return

    # traceloop prints a colour-coded warning to stdout when a call instruments
    # nothing.  Applying one instrument at a time makes that path routine
    # rather than rare, and this runs inside the client's own import statement,
    # so the warnings must not reach their stream.
    with _suppressed_output():
        apply_traceloop_instruments(
            should_enrich_metrics=should_enrich_metrics,
            # Netra hosts no image store, so there is nothing to upload to.
            # None is what the SDK has always passed here.  The instrumentors
            # that receive it declare it Optional and guard on it (see
            # opentelemetry.instrumentation.openai's ``Config`` and
            # ``chat_wrappers``); only traceloop's intermediate signature types
            # it as required, hence the ignore.
            base64_image_uploader=None,
            instruments=instruments,
            block_instruments=set(),
        )


@contextmanager
def _suppressed_output() -> Iterator[None]:
    """Swallow anything written to ``sys.stdout``/``sys.stderr`` in this block.

    ``contextlib.redirect_stdout`` cannot be used here.  It saves the stream it
    displaced on the instance rather than on a shared stack, so two threads
    entering and leaving out of order restore each other's replacements:

        A enters (saves real, installs SA) -> B enters (saves SA, installs SB)
        -> A exits (installs real) -> B exits (installs SA)

    ``sys.stdout`` is then left pointing at a discarded buffer for the rest of
    the process, silently swallowing every later ``print`` and traceback.
    Before deferred activation that could not happen — suppression ran once,
    during ``Netra.init()``, on one thread.  Now it runs inside the client's
    own ``import`` statement, so two libraries first imported on two threads
    reach it concurrently.

    A depth counter fixes the ordering: the first thread in installs the
    buffers, the last one out restores the real streams, whatever order they
    arrive in.  Output written by *other* threads during the window is still
    lost; that is the pre-existing cost of a process-global stream swap, and it
    is bounded by the activation rather than by the process lifetime.
    """
    global _output_suppression_depth, _suppressed_streams

    with _output_suppression_lock:
        if _output_suppression_depth == 0:
            _suppressed_streams = (sys.stdout, sys.stderr)
            sys.stdout = StringIO()
            sys.stderr = StringIO()
        _output_suppression_depth += 1

    try:
        yield
    finally:
        with _output_suppression_lock:
            _output_suppression_depth -= 1
            if _output_suppression_depth == 0 and _suppressed_streams is not None:
                sys.stdout, sys.stderr = _suppressed_streams
                _suppressed_streams = None


def is_distribution_installed(name: str) -> bool:
    """Report whether a distribution is installed.

    Stands in for traceloop's ``is_package_installed``, reimplemented over
    ``importlib.metadata`` so that asking whether a library is installed does
    not drag in the ~620 ms ``traceloop.sdk`` import — which is precisely the
    cost deferred activation exists to avoid.

    Unlike traceloop's, both sides are normalised per PEP 503 before matching.
    traceloop compares lower-cased names only, so a gate spelled with an
    underscore never matches a distribution published with a hyphen: on the
    table this replaced, ``aio_pika`` and ``cerebras_cloud_sdk`` silently
    gated their instrumentors off in every environment.  Normalising makes the
    gate mean what it reads as; it can only ever match more, never less.

    Args:
        name: Distribution name, in any PEP 503-equivalent spelling.

    Returns:
        True if the distribution is installed.
    """
    return _normalize_distribution_name(name) in _installed_distribution_names()


def _normalize_distribution_name(name: str) -> str:
    """Reduce a distribution name to its PEP 503 normalised form.

    Args:
        name: Distribution name as written in a gate or in installed metadata.

    Returns:
        Lower-cased name with every run of ``-``, ``_`` or ``.`` collapsed to a
        single ``-``, so all spellings of one distribution compare equal.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


@lru_cache(maxsize=1)
def _installed_distribution_names() -> frozenset[str]:
    """Return the normalised name of every installed distribution.

    Built on first use rather than at import, so the scan lands in
    ``Netra.init()`` and not in ``import netra``.  A distribution installed
    after the first call is not picked up, matching traceloop's own helper.

    Returns:
        PEP 503-normalised distribution names.
    """
    names: set[str] = set()
    for distribution in distributions():
        try:
            name = distribution.name
        except (KeyError, AttributeError):
            # A partially-written or malformed dist-info has no usable Name;
            # it cannot be matched by name either way, so skip it.
            continue
        names.add(_normalize_distribution_name(name))
    return frozenset(names)


def _apply_instrumentor(spec: InstrumentorSpec) -> None:
    """Instantiate the instrumentor described by *spec* and apply it.

    Args:
        spec: The instrumentor to build and apply.
    """
    instrumentor_class = getattr(import_module(spec.module), spec.class_name)
    instrumentor = instrumentor_class(**spec.constructor_kwargs)
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()


def _resolve_traceloop_instruments(names: AbstractSet[str]) -> set["Instruments"]:
    """Resolve traceloop instrument names to enum members.

    Args:
        names: Traceloop instrument enum member names.

    Returns:
        The members the installed traceloop-sdk defines.  A name it does not
        define is logged and dropped, so the SDK stays usable across traceloop
        versions.
    """
    from traceloop.sdk.instruments import Instruments

    members: set[Instruments] = set()
    for name in names:
        member = getattr(Instruments, name, None)
        if member is None:
            logger.warning("Unknown traceloop instrument: %s", name)
        else:
            members.add(member)
    return members


# Subprocess context propagation is always applied, so it needs no selection
# and no trigger module — but it is activated through the same machinery so a
# failure is contained the same way.
SUBPROCESS_ACTIVATION = Activation("SUBPROCESS", partial(_apply_instrumentor, SUBPROCESS_INSTRUMENTOR))
