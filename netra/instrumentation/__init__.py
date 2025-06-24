import logging
from typing import Optional, Callable, Set

from traceloop.sdk import Telemetry, Instruments
from traceloop.sdk.utils.package_check import is_package_installed


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Callable[[str, str, str], str],
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
):
    from traceloop.sdk.tracing.tracing import init_instrumentations
    init_instrumentations(
        should_enrich_metrics=should_enrich_metrics,
        base64_image_uploader=base64_image_uploader,
        instruments=instruments,
        block_instruments=block_instruments,
    )

    # Initialize Google GenAI instrumentation.
    if (instruments is None or Instruments.GOOGLE_GENERATIVEAI in instruments):
        init_google_genai_instrumentation()


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