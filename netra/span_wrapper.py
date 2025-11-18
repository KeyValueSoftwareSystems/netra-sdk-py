import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel

from netra.config import Config
from netra.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Baggage key for local-only blocked spans patterns
_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY = "netra.local_blocked_spans"


class ActionModel(BaseModel):  # type: ignore[misc]
    start_time: str = str((datetime.now().timestamp() * 1_000_000_000))
    action: str
    action_type: str
    success: bool
    affected_records: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, str]] = None


class UsageModel(BaseModel):  # type: ignore[misc]
    model: str
    usage_type: str
    units_used: Optional[int] = None
    cost_in_usd: Optional[float] = None


class ATTRIBUTE:
    LLM_SYSTEM = "llm_system"
    MODEL = "model"
    PROMPT = "prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    USAGE = "usage"
    STATUS = "status"
    DURATION_MS = "duration_ms"
    ERROR_MESSAGE = "error_message"
    ACTION = "action"


class SpanType(str, Enum):
    SPAN = "SPAN"
    GENERATION = "GENERATION"
    TOOL = "TOOL"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"


class SpanWrapper:
    """
    Context manager for tracking observability data for external API calls.

    Usage:
        with combat.start_span("video_gen_task") as span:
            span.set_prompt("A cat playing piano").set_image_height("1024")

            # External API call
            result = external_api.generate_video(...)

            span.set_usage(usage_data)
    """

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        module_name: str = "combat_sdk",
        as_type: Optional[SpanType] = SpanType.SPAN,
    ):
        self.name = name
        self.attributes = attributes or {}

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = "pending"
        self.error_message: Optional[str] = None
        self.module_name = module_name

        # OpenTelemetry span management
        self.tracer = trace.get_tracer(module_name)
        self.span: Optional[trace.Span] = None
        # Internal context manager to manage current-span scope safely
        self._span_cm: Optional[Any] = None
        # Token for locally attached baggage (if any)
        self._local_block_token: Optional[object] = None

        if isinstance(as_type, SpanType):
            self.attributes["netra.span.type"] = as_type.value
        else:
            logger.error("Invalid span type: %s", as_type)
            return

    def __enter__(self) -> "SpanWrapper":
        """Start the span wrapper, begin time tracking, and create OpenTelemetry span."""
        self.start_time = time.time()

        # If user provided local blocked patterns in attributes, attach them as baggage
        try:
            patterns = None
            # Accept either explicit key or short key for convenience
            if isinstance(self.attributes.get("netra.local_blocked_spans"), list):
                patterns = [p for p in self.attributes.get("netra.local_blocked_spans", []) if isinstance(p, str) and p]
            elif isinstance(self.attributes.get("blocked_spans"), list):
                patterns = [p for p in self.attributes.get("blocked_spans", []) if isinstance(p, str) and p]
            if patterns:
                payload = json.dumps(patterns)
                self._local_block_token = otel_context.attach(
                    baggage.set_baggage(_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY, payload, context=otel_context.get_current())
                )
        except Exception:
            logger.debug("Failed to attach local blocked spans baggage on span start", exc_info=True)

        # Create OpenTelemetry span and make it current using OTel's context manager
        # Store the context manager so we can close it in __exit__
        self._span_cm = self.tracer.start_as_current_span(
            name=self.name, kind=SpanKind.CLIENT, attributes=self.attributes
        )
        self.span = self._span_cm.__enter__()

        # Register with SessionManager for name-based lookup
        try:
            SessionManager.register_span(self.name, self.span)
            # Optionally set as current span for SDK consumers that rely on it
            SessionManager.set_current_span(self.span)
        except Exception:
            logger.exception("Failed to register span '%s' with SessionManager", self.name)

        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Any) -> Literal[False]:
        """End the span wrapper, calculate duration, handle errors, and close OpenTelemetry span."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000 if self.start_time is not None else None

        # Set duration
        if duration_ms is not None:
            self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.DURATION_MS}", str(round(duration_ms, 2)))

        # Handle status and errors
        if exc_type is None and self.status == "pending":
            self.status = "success"
            if self.span:
                self.span.set_status(Status(StatusCode.OK))
        elif exc_type is not None:
            self.status = "error"
            self.error_message = str(exc_val)
            self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}", self.error_message)
            if self.span:
                self.span.set_status(Status(StatusCode.ERROR, self.error_message))
                if exc_val is not None:
                    self.span.record_exception(exc_val)
            logger.error(f"Span wrapper {self.name} failed: {self.error_message}")

        self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.STATUS}", self.status)

        # Update span attributes with final values
        if self.span:
            for key, value in self.attributes.items():
                self.span.set_attribute(key, value)

        # End OpenTelemetry span via the context manager (also clears current context)
        if self.span:
            # Unregister from SessionManager before ending span
            try:
                SessionManager.unregister_span(self.name, self.span)
            except Exception:
                logger.exception("Failed to unregister span '%s' from SessionManager", self.name)
        if self._span_cm is not None:
            try:
                # Delegate to OTel CM to properly end span and restore context
                self._span_cm.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self._span_cm = None

        # Detach local blocking baggage if we attached it
        if self._local_block_token is not None:
            try:
                otel_context.detach(self._local_block_token)
            except Exception:
                logger.debug("Failed to detach local blocked spans baggage token", exc_info=True)
            finally:
                self._local_block_token = None

        # Don't suppress exceptions
        return False

    def set_attribute(self, key: str, value: str) -> "SpanWrapper":
        """Set a single attribute and return self for method chaining."""
        self.attributes[key] = value
        # Also set on the span if it exists
        if self.span:
            self.span.set_attribute(key, value)
        return self

    def set_prompt(self, prompt: str) -> "SpanWrapper":
        """Set the input prompt."""
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.PROMPT}", prompt)

    def set_negative_prompt(self, negative_prompt: str) -> "SpanWrapper":
        """Set the negative prompt."""
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.NEGATIVE_PROMPT}", negative_prompt)

    def set_usage(self, usage: List[UsageModel]) -> "SpanWrapper":
        """Set the usage data as a JSON string."""
        usage_dict = [u.model_dump() for u in usage]
        usage_json = json.dumps(usage_dict)
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.USAGE}", usage_json)

    def set_action(self, action: List[ActionModel]) -> "SpanWrapper":
        """Set the action data as a JSON string."""
        action_dict = [a.model_dump() for a in action]
        action_json = json.dumps(action_dict)
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ACTION}", action_json)

    def set_model(self, model: str) -> "SpanWrapper":
        """Set the model used."""
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.MODEL}", model)

    def set_llm_system(self, system: str) -> "SpanWrapper":
        """Set the LLM system used."""
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.LLM_SYSTEM}", system)

    def set_error(self, error_message: str) -> "SpanWrapper":
        """Manually set an error message."""
        self.status = "error"
        self.error_message = error_message
        if self.span:
            self.span.set_status(Status(StatusCode.ERROR, error_message))
        return self.set_attribute(f"{Config.LIBRARY_NAME}.{ATTRIBUTE.ERROR_MESSAGE}", error_message)

    def set_success(self) -> "SpanWrapper":
        """Manually mark the span wrapper as successful."""
        self.status = "success"
        if self.span:
            self.span.set_status(Status(StatusCode.OK))
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, str]] = None) -> "SpanWrapper":
        """Add an event to the span."""
        if self.span:
            self.span.add_event(name, attributes or {})
        return self

    def get_current_span(self) -> Optional[trace.Span]:
        """Get the current OpenTelemetry span."""
        return self.span
