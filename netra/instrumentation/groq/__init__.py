"""OpenTelemetry Groq instrumentation"""

import logging
import os
from typing import Any, Callable, Collection, Dict, Optional, Tuple

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.groq.version import __version__
from netra.instrumentation.groq.wrappers import (
    achat_wrapper,
    chat_wrapper,
)

logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


_instruments: Tuple[str, ...] = ("groq >= 0.9.0",)


class NetraGroqInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom Groq instrumentor for Netra SDK:"""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        try:
            wrap_function_wrapper(
                "groq.resources.chat.completions",
                "Completions.create",
                chat_wrapper(tracer),
            )
            wrap_function_wrapper(
                "groq.resources.chat.completions",
                "AsyncCompletions.create",
                achat_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument chat completions: {e}")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            unwrap("groq.resources.chat.completions", "Completions.create")
            unwrap("groq.resources.chat.completions", "AsyncCompletions.create")
        except (AttributeError, ModuleNotFoundError):
            logger.error("Failed to uninstrument chat completions")
