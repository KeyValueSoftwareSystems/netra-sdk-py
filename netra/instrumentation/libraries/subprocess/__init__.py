import inspect
import logging
import subprocess
from typing import Any, Callable, Collection, Dict, Tuple

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from netra.instrumentation.libraries.subprocess.utils import inject_subprocess_context

logger = logging.getLogger(__name__)

_instruments: tuple[()] = ()

_POPEN_INIT_SIG = inspect.signature(subprocess.Popen.__init__)


def _popen_init_wrapper(
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Inject OTel trace context into the ``env`` argument of ``subprocess.Popen``.

    Uses :func:`inspect.signature` to bind positional and keyword arguments so
    that ``env`` is handled correctly regardless of how the caller supplied it.
    """
    bound = _POPEN_INIT_SIG.bind_partial(instance, *args, **kwargs)
    bound.arguments["env"] = inject_subprocess_context(bound.arguments.get("env"))
    normalized_kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}
    return wrapped(**normalized_kwargs)


class NetraSubprocessInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Instruments subprocess.Popen to propagate OTel trace context to child processes."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            wrap_function_wrapper("subprocess", "Popen.__init__", _popen_init_wrapper)
        except Exception as e:
            logger.error("Failed to instrument subprocess: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("subprocess.Popen", "__init__")
        except (AttributeError, ModuleNotFoundError) as e:
            logger.error("Failed to uninstrument subprocess: %s", e)
