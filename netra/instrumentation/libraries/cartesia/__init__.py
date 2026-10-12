import logging
from importlib import import_module
from typing import Any, Callable, Collection, List, Tuple

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Tracer, get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.libraries.cartesia.version import __version__
from netra.instrumentation.libraries.cartesia.wrappers import (
    async_stt_wrapper,
    async_tts_wrapper,
    async_voice_changer_wrapper,
    stt_wrapper,
    stt_ws_wrapper,
    tts_bytes_wrapper,
    tts_ws_wrapper,
    voice_changer_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("cartesia >= 2.0.17",)

# (module, class, method, wrapper factory)
_WrapTarget = Tuple[str, str, str, Callable[[Tracer], Callable[..., Any]]]

# SDK 2.x/3.x: one client class per product.
WRAP_TARGETS_V2: Tuple[_WrapTarget, ...] = (
    ("cartesia.tts.client", "TtsClient", "bytes", tts_bytes_wrapper),
    ("cartesia.tts.client", "TtsClient", "sse", tts_bytes_wrapper),
    ("cartesia.tts.socket_client", "TtsClientWithWebsocket", "websocket", tts_ws_wrapper),
    ("cartesia.stt.client", "SttClient", "transcribe", stt_wrapper),
    ("cartesia.stt.socket_client", "SttClientWithWebsocket", "websocket", stt_ws_wrapper),
    ("cartesia.voice_changer.client", "VoiceChangerClient", "bytes", voice_changer_wrapper),
)

# SDK 4.x moved everything to ``cartesia.resources``. ``TTSResource.bytes``
# delegates to ``generate``, so wrapping both would double-count; ``sse`` is a
# separate class attribute aliasing ``generate_sse``, so it is wrapped on its
# own. The 4.x STT websockets have a different, connection-manager API that the
# v2 proxies do not fit, so they are not traced.
WRAP_TARGETS_V4: Tuple[_WrapTarget, ...] = (
    ("cartesia.resources.tts", "TTSResource", "generate", tts_bytes_wrapper),
    ("cartesia.resources.tts", "TTSResource", "generate_sse", tts_bytes_wrapper),
    ("cartesia.resources.tts", "TTSResource", "sse", tts_bytes_wrapper),
    ("cartesia.resources.tts", "TTSResource", "websocket", tts_ws_wrapper),
    ("cartesia.resources.tts", "AsyncTTSResource", "generate", async_tts_wrapper),
    ("cartesia.resources.tts", "AsyncTTSResource", "generate_sse", async_tts_wrapper),
    ("cartesia.resources.tts", "AsyncTTSResource", "sse", async_tts_wrapper),
    ("cartesia.resources.stt.stt", "STTResource", "transcribe", stt_wrapper),
    ("cartesia.resources.stt.stt", "AsyncSTTResource", "transcribe", async_stt_wrapper),
    ("cartesia.resources.voice_changer", "VoiceChangerResource", "generate", voice_changer_wrapper),
    ("cartesia.resources.voice_changer", "AsyncVoiceChangerResource", "generate", async_voice_changer_wrapper),
)


# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.cartesia"


def _select_wrap_targets() -> Tuple[_WrapTarget, ...]:
    """Return the wrap table matching the installed SDK's layout.

    Returns:
        The 4.x table when ``cartesia.resources`` exists, the 2.x/3.x one otherwise.
    """
    try:
        import_module("cartesia.resources")
    except ImportError:
        return WRAP_TARGETS_V2
    return WRAP_TARGETS_V4


def _installed_version() -> str:
    """Return the installed cartesia version, for the inactive warning."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cartesia")
    except PackageNotFoundError:
        return "(unknown version)"


class NetraCartesiaInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Custom Cartesia instrumentor for Netra SDK:
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Return the instrument dependencies for this instrumentor.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument the Cartesia client methods.
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(_TRACER_NAME, __version__, tracer_provider)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to initialize Cartesia tracer: {e}")
            return

        wrapped: List[_WrapTarget] = []
        for target in _select_wrap_targets():
            module_name, class_name, method_name, wrapper_factory = target
            try:
                wrap_function_wrapper(module_name, f"{class_name}.{method_name}", wrapper_factory(tracer))
            except (AttributeError, ImportError) as e:
                logger.debug("Failed to wrap %s.%s.%s: %s", module_name, class_name, method_name, e)
                continue
            wrapped.append(target)
        self._wrapped_targets = wrapped

        if not wrapped:
            from netra.instrumentation.wiring.activation import warn_instrumentation_inactive

            warn_instrumentation_inactive("CARTESIA", f"no known API found in cartesia {_installed_version()}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument the Cartesia client methods.
        """
        # BaseInstrumentor is a singleton whose __init__ reruns on every
        # construction, so fall back to the table for the installed layout.
        for module_name, class_name, method_name, _ in getattr(self, "_wrapped_targets", None) or (
            _select_wrap_targets()
        ):
            try:
                unwrap(getattr(import_module(module_name), class_name), method_name)
            except (AttributeError, ImportError):
                logger.debug("Failed to unwrap %s.%s.%s", module_name, class_name, method_name)
        self._wrapped_targets = []
