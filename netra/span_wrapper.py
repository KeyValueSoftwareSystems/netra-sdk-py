import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.trace.propagation import set_span_in_context
from pydantic import BaseModel, field_validator

from netra.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionModel(BaseModel):  # type: ignore[misc]
    start_time: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    action: str
    action_type: str
    success: bool
    affected_records: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, str]] = None

    @field_validator("start_time")  # type: ignore[misc]
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Validate that start_time is in ISO 8601 format with microseconds and Z suffix."""
        # Pattern for ISO 8601 with microseconds: YYYY-MM-DDTHH:MM:SS.ffffffZ
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"

        if not re.match(pattern, value):
            raise ValueError(
                f"start_time must be in ISO 8601 format with microseconds: "
                f"YYYY-MM-DDTHH:MM:SS.ffffffZ (e.g., 2025-07-18T14:30:45.123456Z). "
                f"Got: {value}"
            )
        return value


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

    def __init__(self, name: str, attributes: Optional[Dict[str, str]] = None, module_name: str = "combat_sdk"):
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
        self.context_token: Optional[Any] = None

    def __enter__(self) -> "SpanWrapper":
        """Start the span wrapper, begin time tracking, and create OpenTelemetry span."""
        self.start_time = time.time()

        # Create OpenTelemetry span
        self.span = self.tracer.start_span(name=self.name, kind=SpanKind.CLIENT, attributes=self.attributes)

        # Set span in context
        ctx = set_span_in_context(self.span)
        self.context_token = context_api.attach(ctx)

        logger.info(f"Started span wrapper: {self.name}")
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

        # End OpenTelemetry span and detach context
        if self.span:
            self.span.end()
        if self.context_token:
            context_api.detach(self.context_token)

        logger.info(
            f"Ended span wrapper: {self.name} (Status: {self.status}, Duration: {duration_ms:.2f}ms)"
            if duration_ms is not None
            else f"Ended span wrapper: {self.name} (Status: {self.status})"
        )

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
