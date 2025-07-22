import logging
import threading
from typing import Any, Dict, Optional, Set

from netra.instrumentation.instruments import InstrumentSet, NetraInstruments

from .config import Config

# Instrumentor functions
from .instrumentation import init_instrumentations
from .session_manager import SessionManager
from .span_wrapper import ActionModel, SpanWrapper, UsageModel
from .tracer import Tracer

logger = logging.getLogger(__name__)


class Netra:
    """
    Main SDK class. Call SDK.init(...) at the start of your application
    to configure OpenTelemetry and enable all built-in LLM + VectorDB instrumentations.
    """

    _initialized = False
    # Use RLock so the thread that already owns the lock can re-acquire it safely
    _init_lock = threading.RLock()

    @classmethod
    def is_initialized(cls) -> bool:
        """Thread-safe check if Netra has been initialized.

        Returns:
            bool: True if Netra has been initialized, False otherwise
        """
        with cls._init_lock:
            return cls._initialized

    @classmethod
    def init(
        cls,
        app_name: Optional[str] = None,
        headers: Optional[str] = None,
        disable_batch: Optional[bool] = None,
        trace_content: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        instruments: Optional[Set[NetraInstruments]] = None,
        block_instruments: Optional[Set[NetraInstruments]] = None,
    ) -> None:
        # Acquire lock at the start of the method and hold it throughout
        # to prevent race conditions during initialization
        with cls._init_lock:
            # Check if already initialized while holding the lock
            if cls._initialized:
                logger.warning("Netra.init() called more than once; ignoring subsequent calls.")
                return

            # Build Config
            cfg = Config(
                app_name=app_name,
                headers=headers,
                disable_batch=disable_batch,
                trace_content=trace_content,
                resource_attributes=resource_attributes,
                environment=environment,
            )

            # Initialize tracer (OTLP exporter, span processor, resource)
            Tracer(cfg)

            # Instrument all supported modules
            #    Pass trace_content flag to instrumentors that can capture prompts/completions

            init_instrumentations(
                should_enrich_metrics=True,
                base64_image_uploader=None,
                instruments=instruments,
                block_instruments=block_instruments,
            )

            cls._initialized = True
            logger.info("Netra successfully initialized.")

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """
        Set session_id context attributes in the current OpenTelemetry context.

        Args:
            session_id: Session identifier
        """
        if session_id:
            SessionManager.set_session_context("session_id", session_id)
        else:
            logger.warning("Session ID must be provided for setting session_id.")

    @classmethod
    def set_user_id(cls, user_id: str) -> None:
        """
        Set user_id context attributes in the current OpenTelemetry context.

        Args:
            user_id: User identifier
        """
        if user_id:
            SessionManager.set_session_context("user_id", user_id)
        else:
            logger.warning("User ID must be provided for setting user_id.")

    @classmethod
    def set_tenant_id(cls, tenant_id: str) -> None:
        """
        Set user_account_id context attributes in the current OpenTelemetry context.

        Args:
            user_account_id: User account identifier
        """
        if tenant_id:
            SessionManager.set_session_context("tenant_id", tenant_id)
        else:
            logger.warning("Tenant ID must be provided for setting tenant_id.")

    @classmethod
    def set_custom_attributes(cls, key: str, value: Any) -> None:
        """
        Set custom attributes context in the current OpenTelemetry context.

        Args:
            key: Custom attribute key
            value: Custom attribute value
        """
        if key and value:
            SessionManager.set_session_context("custom_attributes", {key: value})
        else:
            logger.warning("Both key and value must be provided for custom attributes.")

    @classmethod
    def set_custom_event(cls, event_name: str, attributes: Any) -> None:
        """
        Set custom event in the current OpenTelemetry context.

        Args:
            event_name: Name of the custom event
            attributes: Attributes of the custom event
        """
        if event_name and attributes:
            SessionManager.set_custom_event(event_name, attributes)
        else:
            logger.warning("Both event_name and attributes must be provided for custom events.")

    @classmethod
    def start_span(
        cls,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        module_name: str = "combat_sdk",
    ) -> SpanWrapper:
        """
        Start a new session.
        """
        return SpanWrapper(name, attributes, module_name)


__all__ = ["Netra", "UsageModel", "ActionModel"]
