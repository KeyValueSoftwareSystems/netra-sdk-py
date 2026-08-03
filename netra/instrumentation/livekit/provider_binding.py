"""Binds livekit-agents' OTel tracer to Netra's provider, behind a shield.

``livekit-agents`` does two things to whatever ``TracerProvider`` it is handed,
both of which are wrong for us:

* it calls ``shutdown()`` on every job cleanup, which would permanently disable
  Netra's ``BatchSpanProcessor`` for every later job in the process;
* it calls ``add_span_processor()`` to install its LiveKit Cloud exporter and a
  metadata processor, which are process-wide and would therefore export *every*
  Netra span to a third party.

``_ShieldedTracerProvider`` delegates the reads LiveKit needs and absorbs both.
"""

import logging
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)

# Set on the *delegate* once bound, mirroring ``_netra_processors_installed``
# in netra/tracer.py, so repeat instrument() calls are idempotent.
_BOUND_FLAG = "_netra_livekit_tracer_bound"


class _ShieldedTracerProvider(trace_sdk.TracerProvider):  # type: ignore[misc]
    """Delegates to Netra's TracerProvider but absorbs everything LiveKit does to it.

    Holds no mutable state, so it needs no lock.

    MUST subclass ``trace_sdk.TracerProvider``: LiveKit's ``_setup_cloud_tracer``
    and ``_shutdown_telemetry`` both gate on
    ``isinstance(..., trace_sdk.TracerProvider)``, and a duck-typed object would
    take a different branch — in the cloud-tracer case, one that never reads our
    resource.
    """

    def __init__(self, delegate: trace_api.TracerProvider) -> None:
        """Wrap *delegate* without initialising a second provider.

        Deliberately does not call ``super().__init__()``: every method LiveKit
        touches is overridden and delegated, and constructing real SDK provider
        state here would create a second, useless span pipeline. The contact
        surface was verified against livekit-agents 1.6.7
        (``telemetry/traces.py`` ``set_tracer_provider`` /
        ``_setup_cloud_tracer`` / ``_shutdown_telemetry``).

        Args:
            delegate: Netra's real SDK ``TracerProvider``.
        """
        self._delegate = delegate

    def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
        """Return a tracer from Netra's provider — the whole point of the shield.

        Args:
            *args: Positional arguments forwarded verbatim to the delegate
                (``instrumenting_module_name`` and friends).
            **kwargs: Keyword arguments forwarded verbatim to the delegate.

        Returns:
            A tracer created by Netra's provider, so LiveKit's spans enter Netra's
            pipeline.
        """
        return self._delegate.get_tracer(*args, **kwargs)

    @property
    def resource(self) -> Any:
        """Expose Netra's resource; LiveKit reads it in ``_setup_cloud_tracer``.

        Falls back to an empty ``Resource`` when the delegate has none: LiveKit
        reaches this behind an ``isinstance(..., trace_sdk.TracerProvider)`` check
        that we satisfy by subclassing, so an API-only delegate would otherwise
        raise ``AttributeError`` inside LiveKit's code.

        Returns:
            The delegate's ``Resource``, or an empty one when it has none.
        """
        return getattr(self._delegate, "resource", Resource.get_empty())

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Propagate flushes: we do want the tail of a session exported.

        Args:
            timeout_millis: Maximum time to wait for the delegate's processors to
                flush.

        Returns:
            Whatever the delegate reports, or True when it exposes no
            ``force_flush`` — there is nothing pending in that case.
        """
        flush = getattr(self._delegate, "force_flush", None)
        if flush is None:
            return True
        result: bool = flush(timeout_millis)
        return result

    def shutdown(self) -> None:
        """Absorb LiveKit's per-job teardown of Netra's tracing pipeline."""
        logger.debug(
            "netra.livekit: absorbed a TracerProvider shutdown; Netra owns this provider's lifecycle",
        )

    def add_span_processor(self, span_processor: SpanProcessor) -> None:
        """Refuse every processor LiveKit tries to install on Netra's provider.

        LiveKit registers a ``_MetadataSpanProcessor`` and a Cloud
        ``BatchSpanProcessor`` whenever recording is enabled. Both are
        process-wide, so accepting them would (a) export every Netra span —
        ``openai.chat``, ``httpx``, ``@task`` — to LiveKit Cloud, and (b) stamp
        ``room_id``/``job_id`` on spans from unrelated work, because
        ``_MetadataSpanProcessor.on_start`` is unconditional.

        Netra spans are never exported to a third party. There is no flag to
        change this.

        Args:
            span_processor: The processor LiveKit asked us to install. Discarded.
        """
        logger.info(
            "netra.livekit: refused LiveKit-added span processor %s; Netra spans are never "
            "exported to LiveKit Cloud. LiveKit Cloud trace recording is inactive in this "
            "process (its logs and session reports are unaffected)",
            type(span_processor).__name__,
        )


def bind_livekit_tracer(provider: trace_api.TracerProvider) -> None:
    """Hand LiveKit a shielded view of Netra's TracerProvider. Idempotent.

    Takes no ``Config``: there is nothing left to configure about the binding.

    Accepts the API type rather than the SDK one because
    ``trace.get_tracer_provider()`` may hand back a proxy — binding is still
    correct in that case, since the shield only delegates.

    Args:
        provider: The tracer provider LiveKit's spans should be created from.

    Raises:
        ImportError: If ``livekit.agents.telemetry.set_tracer_provider`` cannot be
            imported. The caller logs this and continues — losing trace binding
            must not disable the session hooks.
    """
    if getattr(provider, _BOUND_FLAG, False):
        return

    from livekit.agents.telemetry import set_tracer_provider

    shield = _ShieldedTracerProvider(provider)
    # No metadata= argument, ever: that path calls add_span_processor() on the
    # object we hand over, so keeping the call single-argument means the
    # guarantee does not depend on our gate holding in a future LiveKit version.
    set_tracer_provider(shield)
    setattr(provider, _BOUND_FLAG, True)
    logger.info("netra.livekit: bound livekit-agents tracer to Netra's TracerProvider")
