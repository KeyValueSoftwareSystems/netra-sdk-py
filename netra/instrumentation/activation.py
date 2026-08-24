"""Applies one instrumentation, whoever implements it.

Every instrumentation — Netra's own or one delegated to traceloop — is wrapped
in an :class:`Activation`: a name plus a callable that applies it.  That single
shape is what lets ``netra.instrumentation.lazy`` defer activation to the first
import of the target library without knowing anything about instrumentors.

No instrumentor module is imported until its instrumentation is actually
applied — importing an instrumentor imports the library it patches, and
deferring that import is the point of the whole arrangement.
"""

import logging
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache, partial
from importlib import import_module
from importlib.metadata import distributions
from io import StringIO
from typing import TYPE_CHECKING, AbstractSet, Callable, NamedTuple, Optional, Sequence

from netra.instrumentation.instruments import InstrumentSet
from netra.instrumentation.registry import CUSTOM_INSTRUMENTORS, SUBPROCESS_INSTRUMENTOR, InstrumentorSpec
from netra.instrumentation.selection import InstrumentationSelection

if TYPE_CHECKING:
    # Type-only: importing traceloop.sdk at runtime costs ~620 ms.
    from traceloop.sdk.instruments import Instruments

logger = logging.getLogger(__name__)


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


def build_activations(
    selection: InstrumentationSelection,
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
) -> list[Activation]:
    """Build one activation per selected instrumentation, in activation order.

    Traceloop instrumentations come first, in name order, then Netra's own in
    registry order, so the order does not depend on set iteration.

    Args:
        selection: The instrumentations to enable.
        should_enrich_metrics: Whether to enrich metrics.
        base64_image_uploader: Optional callback for image uploads.

    Returns:
        The activations, in the order they should be applied.
    """
    activations = [
        Activation(name, partial(apply_traceloop_instrumentation, name, should_enrich_metrics, base64_image_uploader))
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
        logger.debug("No instrumentor registered for: %s", ", ".join(sorted(i.name for i in unregistered)))

    return activations


def activate_now(activations: Sequence[Activation]) -> None:
    """Apply every activation immediately, importing each target library.

    Args:
        activations: The instrumentations to apply, in activation order.
    """
    for activation in activations:
        run_activation(activation)


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


def apply_traceloop_instrumentation(
    name: str,
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
) -> None:
    """Apply a single traceloop instrumentation by enum member name.

    Args:
        name: Traceloop ``Instruments`` member name.
        should_enrich_metrics: Whether to enrich metrics.
        base64_image_uploader: Optional callback for image uploads.
    """
    from traceloop.sdk.tracing.tracing import init_instrumentations as apply_traceloop_instruments

    instruments = _resolve_traceloop_instruments({name})
    if not instruments:
        return

    # traceloop prints a colour-coded warning to stdout when a call instruments
    # nothing.  Applying one instrument at a time makes that path routine
    # rather than rare, and this runs inside the client's own import statement,
    # so the warnings must not reach their stream.  Scoped as tightly as
    # possible: redirect_stdout/stderr are process-global.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        apply_traceloop_instruments(
            should_enrich_metrics=should_enrich_metrics,
            base64_image_uploader=base64_image_uploader,
            instruments=instruments,
            block_instruments=set(),
        )


def is_distribution_installed(name: str) -> bool:
    """Report whether a distribution is installed.

    Equivalent to traceloop's ``is_package_installed``, reimplemented over
    ``importlib.metadata`` so that asking whether a library is installed does
    not drag in the ~620 ms ``traceloop.sdk`` import — which is precisely the
    cost deferred activation exists to avoid.

    Args:
        name: Distribution name as it appears in installed metadata.

    Returns:
        True if the distribution is installed.
    """
    return name.lower() in _installed_distribution_names()


@lru_cache(maxsize=1)
def _installed_distribution_names() -> frozenset[str]:
    """Return the lower-cased name of every installed distribution.

    Built on first use rather than at import, so the scan lands in
    ``Netra.init()`` and not in ``import netra``.  A distribution installed
    after the first call is not picked up, matching traceloop's own helper.

    Returns:
        Lower-cased distribution names.
    """
    names: set[str] = set()
    for distribution in distributions():
        try:
            name = distribution.name
        except (KeyError, AttributeError):
            # A partially-written or malformed dist-info has no usable Name;
            # it cannot be matched by name either way, so skip it.
            continue
        names.add(name.lower())
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
