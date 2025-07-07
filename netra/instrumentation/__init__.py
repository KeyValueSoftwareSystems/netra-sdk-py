import logging
from typing import Any, Callable, Optional, Set

from traceloop.sdk import Instruments, Telemetry
from traceloop.sdk.utils.package_check import is_package_installed

from netra.instrumentation.instruments import CustomInstruments, NetraInstruments


def init_instrumentations(
    should_enrich_metrics: bool,
    base64_image_uploader: Optional[Callable[[str, str, str], str]],
    instruments: Optional[Set[NetraInstruments]] = None,
    block_instruments: Optional[Set[NetraInstruments]] = None,
) -> None:
    from traceloop.sdk.tracing.tracing import init_instrumentations

    traceloop_instruments = set()
    traceloop_block_instruments = set()
    netra_custom_instruments = set()
    netra_custom_block_instruments = set()
    if instruments is not None:
        for instrument in instruments:
            if instrument.origin == CustomInstruments:  # type: ignore[attr-defined]
                netra_custom_instruments.add(getattr(CustomInstruments, instrument.name))
            else:
                traceloop_instruments.add(getattr(Instruments, instrument.name))
    if block_instruments is not None:
        for instrument in block_instruments:
            if instrument.origin == CustomInstruments:  # type: ignore[attr-defined]
                netra_custom_block_instruments.add(getattr(CustomInstruments, instrument.name))
            else:
                traceloop_block_instruments.add(getattr(Instruments, instrument.name))
    traceloop_block_instruments.update(
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
        instruments=traceloop_instruments,
        block_instruments=traceloop_block_instruments,
    )

    netra_custom_instruments = netra_custom_instruments or set(CustomInstruments)
    netra_custom_instruments = netra_custom_instruments - netra_custom_block_instruments
    # Initialize Google GenAI instrumentation.
    if CustomInstruments.GOOGLE_GENERATIVEAI in netra_custom_instruments:
        init_google_genai_instrumentation()

    # Initialize FastAPI instrumentation.
    if CustomInstruments.FASTAPI in netra_custom_instruments:
        init_fastapi_instrumentation()

    # Initialize Qdrant instrumentation.
    if CustomInstruments.QDRANTDB in netra_custom_instruments:
        init_qdrant_instrumentation()

    # Initialize Weaviate instrumentation.
    if CustomInstruments.WEAVIATEDB in netra_custom_instruments:
        init_weviate_instrumentation()

    # Initialize HTTPX instrumentation.
    if CustomInstruments.HTTPX in netra_custom_instruments:
        init_httpx_instrumentation()

    # Initialize AIOHTTP instrumentation.
    if CustomInstruments.AIOHTTP in netra_custom_instruments:
        init_aiohttp_instrumentation()

    # Initialize Cohere instrumentation.
    if CustomInstruments.COHEREAI in netra_custom_instruments:
        init_cohere_instrumentation()

    if CustomInstruments.MISTRALAI in netra_custom_instruments:
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
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            instrumentor = HTTPXClientInstrumentor()
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
            from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

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
