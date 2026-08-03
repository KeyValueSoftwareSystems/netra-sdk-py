"""LiveKit voice-agent instrumentation for Netra."""

import logging
import threading
from typing import Any, Collection, Optional

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.sdk import trace as trace_sdk
from wrapt import wrap_function_wrapper

from netra.config import Config, get_active_config
from netra.instrumentation.livekit.processors import LiveKitSpanProcessor
from netra.instrumentation.livekit.provider_binding import bind_livekit_tracer
from netra.instrumentation.livekit.wrappers import wrap_start

logger = logging.getLogger(__name__)

_instruments = ("livekit-agents >= 1.6.0, < 2.0.0",)

_AGENT_SESSION_MODULE = "livekit.agents.voice.agent_session"
_AGENT_SESSION_CLASS = "AgentSession"
_START_METHOD = "start"
# wrapt resolves a dotted attribute path against the module; ``unwrap`` does not
# — see ``_uninstrument``.
_SESSION_START_METHOD = f"{_AGENT_SESSION_CLASS}.{_START_METHOD}"

# Set on the provider once our processor is attached. OTel has no
# remove_span_processor, so a double registration would silently double every
# mapped attribute write; this flag is the only thing preventing that. Mirrors
# ``_netra_processors_installed`` in netra/tracer.py.
_PROCESSORS_FLAG = "_netra_livekit_processors_installed"

# Guards against double-wrapping. BaseInstrumentor.is_instrumented_by_opentelemetry
# already prevents a repeat instrument(); this covers a direct _instrument() call.
_session_hook_lock = threading.Lock()
_session_hook_installed = False


class NetraLiveKitInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Binds livekit-agents' OTel tracer to Netra's provider and installs session hooks.

    Unlike most Netra instrumentors this one creates no spans of its own on the
    trace path — livekit-agents already emits a full span tree
    (``agent_session`` → ``agent_turn`` → ``llm_node`` / ``tts_node`` /
    ``function_tool``). Our job is to make that tree land in Netra's pipeline,
    shield the providers from LiveKit's per-job telemetry teardown, and stamp the
    Netra session id on the session root.

    Note on session-id scope: the id is attached for the duration of
    ``AgentSession.start`` and inherited by every task LiveKit creates during it,
    then detached. Code running in the entrypoint task *after*
    ``await session.start(...)`` therefore carries no session id — call
    ``Netra.set_session_id()`` for that, which is process-wide by design.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the package requirement this instrumentor applies to.

        Returns:
            The ``livekit-agents`` version range this instrumentation was written
            against.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Install the LiveKit integration.

        Each step is isolated so that a LiveKit signature change disables one
        feature rather than the whole integration — and never ``Netra.init()``.

        Args:
            **kwargs: Optional ``config`` and ``tracer_provider`` overrides.
        """
        config: Optional[Config] = kwargs.get("config") or get_active_config()
        if config is None:
            logger.warning(
                "netra.livekit: no active Netra config; LiveKit instrumentation is disabled. "
                "Call Netra.init() before instrumenting"
            )
            return

        provider = kwargs.get("tracer_provider") or trace.get_tracer_provider()

        try:
            bind_livekit_tracer(provider)
        except Exception:
            logger.exception(
                "netra.livekit: could not bind the LiveKit tracer to Netra's provider; "
                "LiveKit spans will NOT reach Netra. Session hooks are unaffected"
            )

        try:
            self._register_processors(provider)
        except Exception:
            logger.exception("netra.livekit: could not register span processors; lk.* mapping is disabled")

        try:
            _install_session_hook()
        except Exception:
            logger.exception("netra.livekit: could not install the session hook; netra.session_id will be missing")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove the session hook.

        Does not un-bind the tracer provider or unregister the processor: OTel
        offers no ``remove_span_processor`` and LiveKit offers no way to restore a
        previous provider. Both are documented limitations; the processor is
        inert without LiveKit spans to act on, so leaving it registered is
        harmless.

        Args:
            **kwargs: Unused.
        """
        global _session_hook_installed

        try:
            # MUST pass the class, not ``(module, "AgentSession.start")``: unwrap()
            # resolves its second argument with a single ``getattr``, which cannot
            # walk a dotted path, and it defaults to None rather than raising — so
            # the dotted form is a silent no-op that leaves the wrapper installed
            # and reports success.
            from livekit.agents.voice.agent_session import AgentSession

            unwrap(AgentSession, _START_METHOD)
        except (AttributeError, ImportError):
            logger.error("netra.livekit: failed to uninstrument %s", _SESSION_START_METHOD)

        with _session_hook_lock:
            _session_hook_installed = False

    @staticmethod
    def _register_processors(provider: Any) -> None:
        """Append this integration's span processor to *provider*.

        Called from ``_instrument()``, so it only runs when livekit-agents is
        installed and ``InstrumentSet.LIVEKIT`` is enabled — exactly the gate we
        want, without ``netra/tracer.py`` having to reimplement it.

        This is appended *after* ``BatchSpanProcessor``; see the module
        docstring in ``processors.py`` for the invariant that makes it safe before
        adding a second.

        Args:
            provider: The tracer provider to register on.
        """
        if not isinstance(provider, trace_sdk.TracerProvider):
            logger.warning("netra.livekit: provider is not an SDK TracerProvider; span mapping disabled")
            return
        if getattr(provider, _PROCESSORS_FLAG, False):
            return

        provider.add_span_processor(LiveKitSpanProcessor())
        setattr(provider, _PROCESSORS_FLAG, True)
        logger.debug("netra.livekit: registered LiveKitSpanProcessor")


def _install_session_hook() -> None:
    """Wrap ``AgentSession.start`` so the session root span carries the session id.

    Guarded by a module flag because ``wrapt`` would otherwise double-wrap on a
    repeat ``_instrument()`` call.
    """
    global _session_hook_installed

    with _session_hook_lock:
        if _session_hook_installed:
            return

        try:
            wrap_function_wrapper(_AGENT_SESSION_MODULE, _SESSION_START_METHOD, wrap_start)
        except Exception:
            logger.exception(
                "netra.livekit: could not wrap AgentSession.start; netra.session_id will be missing "
                "from LiveKit spans"
            )
            return

        _session_hook_installed = True


__all__ = ["NetraLiveKitInstrumentor"]
