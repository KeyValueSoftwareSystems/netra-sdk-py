import logging
import threading
from typing import Optional, Dict, Any

from .config import Config
from .tracer import Tracer

# Instrumentor functions
from .instrumentation import init_instrumentations

logger = logging.getLogger(__name__)


class Combat:
    """
    Main SDK class. Call SDK.init(...) at the start of your application
    to configure OpenTelemetry and enable all built-in LLM + VectorDB instrumentations.
    """

    _initialized = False
    _init_lock = threading.Lock()  # Lock for thread safety during initialization

    @classmethod
    def is_initialized(cls) -> bool:
        """Thread-safe check if Combat has been initialized.

        Returns:
            bool: True if Combat has been initialized, False otherwise
        """
        with cls._init_lock:
            return cls._initialized

    @classmethod
    def init(
        cls,
        app_name: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
    ):
        # Acquire lock before checking _initialized to prevent race conditions
        if cls.is_initialized():
            logger.warning(
                "Combat.init() called more than once; ignoring subsequent calls."
            )
            return

        # Build Config
        cfg = Config(
            app_name=app_name,
            otlp_endpoint=otlp_endpoint,
            api_key=api_key,
            headers=headers,
            disable_batch=disable_batch,
            trace_content=trace_content,
            resource_attributes=resource_attributes,
        )

        # Initialize tracer (OTLP exporter, span processor, resource)
        Tracer(cfg)

        # Instrument all supported modules
        #    Pass trace_content flag to instrumentors that can capture prompts/completions
        init_instrumentations(
            should_enrich_metrics=True,
            base64_image_uploader=None,
            instruments=None,
            block_instruments=None,
        )
        cls._initialized = True
        logger.info("Combat successfully initialized.")


__all__ = ["Combat"]
