import logging

from traceloop.sdk.utils.package_check import is_package_installed

logger = logging.getLogger(__name__)


def init_qdrant_instrumentor() -> bool:
    try:
        if is_package_installed("qdrant-client"):
            from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

            instrumentor = QdrantInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Qdrant instrumentor: {e}")
        return False


def init_weviate_instrumentor() -> bool:
    try:
        if is_package_installed("weaviate-client"):
            from .weaviate_instrumentor import WeaviateInstrumentor

            instrumentor = WeaviateInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
        return True
    except Exception as e:
        logging.error(f"Error initializing Weaviate instrumentor: {e}")
        return False
