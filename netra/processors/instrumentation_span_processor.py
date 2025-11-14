import logging
from typing import Any, Callable, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

from netra.config import Config
from netra.instrumentation.instruments import InstrumentSet

logger = logging.getLogger(__name__)


ALLOWED_INSTRUMENTATION_NAMES = {member.value for member in InstrumentSet}  # type: ignore[attr-defined]


class InstrumentationSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Span processor to record this span's instrumentation name and wrap set_attribute.

    - Records raw instrumentation scope name for the span
    - Wraps span.set_attribute to truncate string values to max 1000 chars (also inside simple lists/dicts of strings)
    """

    def __init__(self) -> None:
        super().__init__()

    def _detect_raw_instrumentation_name(self, span: trace.Span) -> Optional[str]:
        """Detect the raw instrumentation name for the span."""

        scope = getattr(span, "instrumentation_scope", None)

        if scope is not None:
            name = getattr(scope, "name", None)
            if isinstance(name, str) and name:
                if name.startswith("opentelemetry.instrumentation.") or name.startswith("netra.instrumentation."):
                    try:
                        base = name.rsplit(".", 1)[-1].strip()
                        if base:
                            return base
                    except Exception:
                        pass
                return name
        return None

    def _truncate_value(self, value: Any) -> Any:
        """Truncate string values to max 1000 chars (also inside simple lists/dicts of strings)."""
        try:
            if isinstance(value, str):
                return value if len(value) <= Config.ATTRIBUTE_MAX_LEN else value[: Config.ATTRIBUTE_MAX_LEN]
            if isinstance(value, (bytes, bytearray)):
                return value[: Config.ATTRIBUTE_MAX_LEN]
            if isinstance(value, list):
                return [self._truncate_value(v) if isinstance(v, (str, bytes, bytearray)) else v for v in value]
            if isinstance(value, dict):
                return {
                    k: self._truncate_value(v) if isinstance(v, (str, bytes, bytearray)) else v
                    for k, v in value.items()
                }
        except Exception:
            return value
        return value

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Start span and wrap set_attribute."""
        try:
            # Wrap set_attribute first so subsequent sets are also processed
            original_set_attribute: Callable[[str, Any], None] = span.set_attribute

            def wrapped_set_attribute(key: str, value: Any) -> None:
                try:
                    # Truncate value(s)
                    truncated = self._truncate_value(value)
                    # Forward to original
                    original_set_attribute(key, truncated)
                except Exception:
                    # Best-effort; never break span
                    try:
                        original_set_attribute(key, value)
                    except Exception:
                        pass

            # Monkey patch for this span's lifetime
            setattr(span, "set_attribute", wrapped_set_attribute)

            # Set this span's instrumentation name
            name = self._detect_raw_instrumentation_name(span)
            if name in ALLOWED_INSTRUMENTATION_NAMES:
                span.set_attribute(f"{Config.LIBRARY_NAME}.instrumentation.name", name)
        except Exception:
            pass

    def on_end(self, span: trace.Span) -> None:
        """End span."""

    def force_flush(self, timeout_millis: int = 30000) -> None:
        """Force flush span."""

    def shutdown(self) -> None:
        pass
