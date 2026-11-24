from typing import Any, Callable, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.libraries.typesafe.version import __version__
from netra.instrumentation.libraries.typesafe.wrappers import async_system_one_wrapper, system_one_wrapper
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# 0.7.0 is the first release built on pydantic. 0.6.x returns msgspec structs, which
# have no ``model_dump``: the span would silently lose its model, token usage and
# output, and the questions would be recorded as struct reprs. Below 0.7.0 the
# dependency check skips instrumentation instead.
_instruments = ("typesafe-sdk >= 0.7.0",)

# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.typesafe"

# The clients are patched through the public package rather than their
# ``typesafe_sdk._core.client.*`` home: the class object is the same either
# way, and the public path survives internal reshuffles of the SDK.
_MODULE = "typesafe_sdk"
_METHODS = ("TypeSafeClient.system_one", "AsyncTypeSafeClient.system_one")


class NetraTypeSafeInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Instrumentor for the TypeSafe AI Python SDK (``typesafe-sdk``), which serves the Jev models.

    One ``typesafe.system_one`` CLIENT span is created per ``system_one`` call on the sync
    and async clients. Libraries built on the SDK, such as ``jev`` and ``jevlang``, are
    covered through the same calls. The SDK sends requests with ``httpx2``, which the
    HTTPX instrumentation does not see, so without this there is no span at all.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the pip requirement strings for the instrumented library."""
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Wrap ``system_one`` on the sync and async clients."""
        try:
            tracer = get_tracer(_TRACER_NAME, __version__, kwargs.get("tracer_provider"))
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        wrappers: dict[str, Callable[..., Any]] = {
            "TypeSafeClient.system_one": system_one_wrapper(tracer),
            "AsyncTypeSafeClient.system_one": async_system_one_wrapper(tracer),
        }
        for name, wrapper in wrappers.items():
            try:
                wrap_function_wrapper(_MODULE, name, wrapper)
            except Exception as e:
                logger.error(f"Failed to instrument TypeSafe {name}: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove the wrappers from both clients."""
        for name in _METHODS:
            class_name, method = name.split(".", 1)
            try:
                unwrap(f"{_MODULE}.{class_name}", method)
            except (AttributeError, ImportError):
                logger.error(f"Failed to uninstrument TypeSafe {name}")
