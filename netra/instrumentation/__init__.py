import logging
from typing import Any, Callable, Optional, Set

from fastapi import FastAPI
from traceloop.sdk import Instruments, Telemetry
from traceloop.sdk.utils.package_check import is_package_installed


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
) -> None:
    from traceloop.sdk.tracing.tracing import init_instrumentations

    init_instrumentations(
        should_enrich_metrics=should_enrich_metrics,
        base64_image_uploader=base64_image_uploader,
        instruments=instruments,
        block_instruments=block_instruments,
    )

    # Initialize Google GenAI instrumentation.
    if instruments is None or Instruments.GOOGLE_GENERATIVEAI in instruments:
        init_google_genai_instrumentation()

    # Initialize FastAPI instrumentation.
    if instruments is None or Instruments.FASTAPI in instruments:
        init_fastapi_instrumentor()


def init_google_genai_instrumentation() -> bool:
    """Initialize Google GenAI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("google-genai"):
            Telemetry().capture("instrumentation:genai:init")
            from netra.instrumentation.google_genai import GoogleGenAiInstrumentor

            instrumentor = GoogleGenAiInstrumentor(
                exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Google GenAI instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_fastapi_instrumentor() -> bool:
    """Initialize FastAPI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if not is_package_installed("fastapi"):
            return True
        original_init = FastAPI.__init__

        def _patched_init(self: FastAPI, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

                FastAPIInstrumentor().instrument_app(self)
            except Exception as e:
                logging.warning(f"Failed to auto-instrument FastAPI: {e}")

        FastAPI.__init__ = _patched_init
        return True
    except Exception as e:
        logging.error(f"Error initializing FastAPI instrumentor: {e}")
        Telemetry().log_exception(e)
        return False
