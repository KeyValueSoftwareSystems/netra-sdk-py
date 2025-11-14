import atexit
import logging
import threading
from typing import Any, Dict, List, Optional, Set

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from netra.config import Config
from netra.evaluation import Evaluation, EvaluationScore

# Instrumentor functions
from netra.instrumentation import init_instrumentations
from netra.instrumentation.instruments import NetraInstruments
from netra.logging_utils import configure_package_logging
from netra.session_manager import ConversationType, SessionManager
from netra.span_wrapper import ActionModel, SpanType, SpanWrapper, UsageModel
from netra.tracer import Tracer

logger = logging.getLogger(__name__)


class Netra:
    """
    Main SDK class. Call SDK.init(...) at the start of your application
    to configure OpenTelemetry and enable all built-in LLM + VectorDB instrumentations.
    """

    _initialized = False
    # Use RLock so the thread that already owns the lock can re-acquire it safely
    _init_lock = threading.RLock()
    _root_span = None
    _root_ctx_token = None

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
        debug_mode: Optional[bool] = None,
        enable_root_span: Optional[bool] = None,
        resource_attributes: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        enable_scrubbing: Optional[bool] = None,
        blocked_spans: Optional[List[str]] = None,
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
                debug_mode=debug_mode,
                enable_root_span=enable_root_span,
                resource_attributes=resource_attributes,
                environment=environment,
                enable_scrubbing=enable_scrubbing,
                blocked_spans=blocked_spans,
            )

            # Configure logging based on debug mode
            configure_package_logging(debug_mode=cfg.debug_mode)

            # Initialize tracer (OTLP exporter, span processor, resource)
            Tracer(cfg)

            # Initialize evaluation client and expose as class attribute
            try:
                cls.evaluation = Evaluation(cfg)  # type:ignore[attr-defined]
            except Exception as e:
                logger.warning("Failed to initialize evaluation client: %s", e, exc_info=True)
                cls.evaluation = None  # type:ignore[attr-defined]

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

            # Create and attach a long-lived root span if enabled
            if cfg.enable_root_span:
                tracer = trace.get_tracer("netra.root.span")
                root_name = f"{Config.LIBRARY_NAME}.root.span"
                root_span = tracer.start_span(root_name, kind=SpanKind.INTERNAL)
                # Add useful attributes
                if cfg.app_name:
                    root_span.set_attribute("service.name", cfg.app_name)
                root_span.set_attribute("netra.environment", cfg.environment)
                root_span.set_attribute("netra.library.version", Config.LIBRARY_VERSION)

                # Attach span to current context so subsequent spans become its children
                ctx = trace.set_span_in_context(root_span)
                token = context_api.attach(ctx)

                # Save for potential shutdown/cleanup and session tracking
                cls._root_span = root_span
                cls._root_ctx_token = token
                try:
                    SessionManager.set_current_span(root_span)
                except Exception:
                    pass
                logger.info("Netra root span created and attached to context.")

                # Ensure cleanup at process exit
                atexit.register(cls.shutdown)

    @classmethod
    def shutdown(cls) -> None:
        """Optional cleanup to end the root span and detach context."""
        with cls._init_lock:
            if cls._root_ctx_token is not None:
                try:
                    context_api.detach(cls._root_ctx_token)
                except Exception:
                    pass
                finally:
                    cls._root_ctx_token = None
            if cls._root_span is not None:
                try:
                    cls._root_span.end()
                except Exception:
                    pass
                finally:
                    cls._root_span = None
            # Try to flush and shutdown the tracer provider to ensure export
            try:
                provider = trace.get_tracer_provider()
                if hasattr(provider, "force_flush"):
                    provider.force_flush()
                if hasattr(provider, "shutdown"):
                    provider.shutdown()
            except Exception:
                pass

    @classmethod
    def set_session_id(cls, session_id: str) -> None:
        """
        Set session_id context attributes in the current OpenTelemetry context.

        Args:
            session_id: Session identifier
        """
        if not isinstance(session_id, str):
            logger.error(f"set_session_id: session_id must be a string, got {type(session_id)}")
            return
        if session_id:
            SessionManager.set_session_context("session_id", session_id)
        else:
            logger.warning("set_session_id: Session ID must be provided for setting session_id.")

    @classmethod
    def set_user_id(cls, user_id: str) -> None:
        """
        Set user_id context attributes in the current OpenTelemetry context.

        Args:
            user_id: User identifier
        """
        if not isinstance(user_id, str):
            logger.error(f"set_user_id: user_id must be a string, got {type(user_id)}")
            return
        if user_id:
            SessionManager.set_session_context("user_id", user_id)
        else:
            logger.warning("set_user_id: User ID must be provided for setting user_id.")

    @classmethod
    def set_tenant_id(cls, tenant_id: str) -> None:
        """
        Set user_account_id context attributes in the current OpenTelemetry context.

        Args:
            user_account_id: User account identifier
        """
        if not isinstance(tenant_id, str):
            logger.error(f"set_tenant_id: tenant_id must be a string, got {type(tenant_id)}")
            return
        if tenant_id:
            SessionManager.set_session_context("tenant_id", tenant_id)
        else:
            logger.warning("set_tenant_id: Tenant ID must be provided for setting tenant_id.")

    @classmethod
    def set_custom_attributes(cls, key: str, value: Any) -> None:
        """
        Set a custom attribute on the currently active OpenTelemetry span only.

        Args:
            key: Custom attribute key
            value: Custom attribute value
        """
        if key and value:
            SessionManager.set_attribute_on_active_span(f"{Config.LIBRARY_NAME}.custom.{key}", value)
        else:
            logger.warning("Both key and value must be provided for custom attributes.")
            return

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
    def add_conversation(cls, conversation_type: ConversationType, role: str, content: Any) -> None:
        """
        Append a conversation entry and set span attribute 'conversation' as an array.
        If a conversation array already exists for the current active span, this appends
        to it; otherwise, it initializes a new array.
        """
        SessionManager.add_conversation(conversation_type=conversation_type, role=role, content=content)

    @classmethod
    def start_span(
        cls,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        module_name: str = "combat_sdk",
        as_type: Optional[SpanType] = SpanType.SPAN,
    ) -> SpanWrapper:
        """
        Start a new session.
        """
        return SpanWrapper(name, attributes, module_name, as_type=as_type)


__all__ = ["Netra", "UsageModel", "ActionModel", "SpanType", "EvaluationScore"]
