import logging
from typing import Any

from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

logger = logging.getLogger(__name__)


def init_fastapi_instrumentor() -> None:
    """Monkey-patches FastAPI.__init__ to auto-instrument all new instances."""
    original_init = FastAPI.__init__

    def _patched_init(self: FastAPI, *args: Any, **kwargs: Any) -> None:
        # Call original FastAPI constructor
        original_init(self, *args, **kwargs)

        # Auto-instrument the app
        try:
            FastAPIInstrumentor().instrument_app(self)
            logger.debug("Auto-instrumented FastAPI app")
        except Exception as e:
            logger.warning(f"Failed to auto-instrument FastAPI: {e}")

    # Apply the patch
    FastAPI.__init__ = _patched_init
    logger.info("FastAPI auto-instrumentation patch applied")
