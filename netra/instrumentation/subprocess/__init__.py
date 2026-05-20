import logging
from typing import Any, Callable, Collection, Dict, Tuple

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from netra.instrumentation.subprocess.utils import inject_subprocess_context

logger = logging.getLogger(__name__)

_instruments: tuple[()] = ()


def _popen_init_wrapper(
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    kwargs["env"] = inject_subprocess_context(kwargs.get("env"))
    return wrapped(*args, **kwargs)


class NetraSubprocessInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Instruments subprocess.Popen to propagate OTel trace context to child processes."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            wrap_function_wrapper("subprocess", "Popen.__init__", _popen_init_wrapper)
            logger.debug("subprocess.Popen patched for OTel context propagation.")
        except Exception as e:
            logger.error("Failed to instrument subprocess: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("subprocess.Popen", "__init__")
            logger.debug("subprocess.Popen patch removed.")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("Failed to uninstrument subprocess: %s", e)
