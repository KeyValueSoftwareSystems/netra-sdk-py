import logging
from typing import Any, Callable, Optional, Set

from traceloop.sdk import Instruments, Telemetry
from traceloop.sdk.utils.package_check import is_package_installed


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
    instruments: Optional[Set[Instruments]] = None,
    block_instruments: Optional[Set[Instruments]] = None,
) -> None:
    from traceloop.sdk.tracing.tracing import init_instrumentations

    block_instruments = block_instruments or set()
    block_instruments.update(
        {
            Instruments.WEAVIATE,
            Instruments.QDRANT,
            Instruments.GOOGLE_GENERATIVEAI,
            Instruments.MISTRAL,
        }
    )

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
    if instruments is None:
        init_fastapi_instrumentation()

    # Initialize Qdrant instrumentation.
    if instruments is None or Instruments.QDRANT in instruments:
        init_qdrant_instrumentation()

    # Initialize Weaviate instrumentation.
    if instruments is None or Instruments.WEAVIATE in instruments:
        init_weviate_instrumentation()

    # Initialize HTTPX instrumentation.
    if instruments is None:
        init_httpx_instrumentation()

    # Initialize AIOHTTP instrumentation.
    if instruments is None:
        init_aiohttp_instrumentation()

    # Initialize Cohere instrumentation.
    if instruments is None or Instruments.COHERE in instruments:
        init_cohere_instrumentation()

    if instruments is None or Instruments.MISTRAL in instruments:
        init_mistral_instrumentor()


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


def init_fastapi_instrumentation() -> bool:
    """Initialize FastAPI instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if not is_package_installed("fastapi"):
            return True
        from fastapi import FastAPI

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


def init_qdrant_instrumentation() -> bool:
    """Initialize Qdrant instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("qdrant-client"):
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Qdrant instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_weviate_instrumentation() -> bool:
    """Initialize Weaviate instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("weaviate-client"):
            from netra.instrumentation.weaviate import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Weaviate instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_httpx_instrumentation() -> bool:
    """Initialize HTTPX instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("httpx"):
            from netra.instrumentation.httpx import HTTPXInstrumentor

            instrumentor = HTTPXInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing HTTPX instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_aiohttp_instrumentation() -> bool:
    """Initialize AIOHTTP instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("aiohttp"):
            from netra.instrumentation.aiohttp import AioHttpClientInstrumentor

            instrumentor = AioHttpClientInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing AIOHTTP instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_cohere_instrumentation() -> bool:
    """Initialize Cohere instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("cohere"):
            from netra.instrumentation.cohere import CohereInstrumentor

            instrumentor = CohereInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Cohere instrumentor: {e}")
        Telemetry().log_exception(e)
        return False


def init_mistral_instrumentor() -> bool:
    """Initialize Mistral instrumentation.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    try:
        if is_package_installed("mistralai"):
            from netra.instrumentation.mistralai import MistralAiInstrumentor

            instrumentor = MistralAiInstrumentor(
                exception_logger=lambda e: Telemetry().log_exception(e),
            )
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"-----Error initializing Mistral instrumentor: {e}")
        Telemetry().log_exception(e)
        return False
