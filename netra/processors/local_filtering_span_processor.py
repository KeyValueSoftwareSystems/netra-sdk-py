import json
import logging
from contextlib import contextmanager
from typing import List, Optional, Sequence

from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)

# Baggage key to carry local blocked span patterns in the active context
_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY = "netra.local_blocked_spans"
# Attribute key to copy resolved local blocked patterns onto each span
_LOCAL_BLOCKED_SPANS_ATTR_KEY = "netra.local_blocked_spans"

# Registry of locally blocked spans: span_id -> parent_context
# This lets exporters reparent children reliably even when children export before parents
BLOCKED_LOCAL_PARENT_MAP: dict[object, object] = {}


class LocalFilteringSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Propagates local blocked span patterns from baggage to span attributes.

    - On span start, reads patterns from baggage key `_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY` on the provided
      `parent_context` and, if present, sets them on the span attribute `_LOCAL_BLOCKED_SPANS_ATTR_KEY`.
    - This enables exporters (e.g., `FilteringSpanExporter`) to decide per-span blocking based on the
      active context in which the span was created.
    """

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """
        Start span and wrap set_attribute.

        Args:
            span: The span to start.
            parent_context: The parent context of the span.
        """
        try:
            # Use provided parent_context if available, otherwise fall back to current context
            ctx_to_read = parent_context if parent_context is not None else otel_context.get_current()
            raw = baggage.get_baggage(_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY, context=ctx_to_read)
            if not raw:
                return
            patterns: Optional[List[str]] = _decode_patterns(raw)
            if patterns:
                try:
                    span.set_attribute(_LOCAL_BLOCKED_SPANS_ATTR_KEY, patterns)
                except Exception:
                    # Best-effort: never break span start
                    logger.debug("Failed setting local blocked patterns on span", exc_info=True)
                # If this span matches the local patterns, record it as locally blocked for reparenting
                try:
                    name = getattr(span, "name", None)
                    if isinstance(name, str) and name and _matches_any_pattern(name, patterns):
                        ctx = getattr(span, "context", None)
                        span_id = getattr(ctx, "span_id", None) if ctx else None
                        # Determine the parent SpanContext of this blocked span from the context
                        parent_span = trace.get_current_span(ctx_to_read)
                        parent_span_context = (
                            parent_span.get_span_context() if hasattr(parent_span, "get_span_context") else None
                        )
                        if span_id is not None and parent_span_context is not None:
                            BLOCKED_LOCAL_PARENT_MAP[span_id] = parent_span_context
                            # Mark on the span for visibility/debugging
                            try:
                                span.set_attribute("netra.local_blocked", True)
                            except Exception:
                                pass
                except Exception:
                    logger.debug("Failed to precompute locally blocked mapping on start", exc_info=True)
        except Exception:
            # Never break tracing pipeline
            logger.debug("LocalFilteringSpanProcessor.on_start failed", exc_info=True)

    # No-ops required by interface
    def on_end(self, span: trace.Span) -> None:  # noqa: D401
        """
        End span.

        Args:
            span: The span to end.
        """
        # Cleanup registry entry to avoid leaks
        try:
            ctx = getattr(span, "context", None)
            span_id = getattr(ctx, "span_id", None) if ctx else None
            if span_id is not None:
                BLOCKED_LOCAL_PARENT_MAP.pop(span_id, None)
        except Exception:
            pass
        return

    def shutdown(self) -> None:  # noqa: D401
        """
        Shutdown the processor.
        """
        return

    def force_flush(self, timeout_millis: int = 30000) -> None:  # noqa: D401
        """
        Force flush the processor.

        Args:
            timeout_millis: The timeout in milliseconds.
        """
        return


def _decode_patterns(raw: str) -> Optional[List[str]]:
    """
    Decode patterns stored in baggage.

    Args:
        raw: The raw baggage string.

    Returns:
        The decoded patterns.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
            return [p for p in parsed if p]
    except Exception:
        logger.debug("Failed to decode local blocked patterns from baggage", exc_info=True)
    return None


def _matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    """
    Return True if name matches any pattern (exact, prefix*, *suffix).

    Args:
        name: The name to match.
        patterns: The patterns to match.

    Returns:
        True if name matches any pattern, False otherwise.
    """
    try:
        for p in patterns:
            if not p:
                continue
            if p.endswith("*") and not p.startswith("*"):
                if name.startswith(p[:-1]):
                    return True
            elif p.startswith("*") and not p.endswith("*"):
                if name.endswith(p[1:]):
                    return True
            else:
                if name == p:
                    return True
    except Exception:
        # Be conservative: on error, treat as no match
        return False
    return False


@contextmanager
def block_spans_local(patterns: Sequence[str]):  # type: ignore[no-untyped-def]
    """
    Context manager to locally block spans by name patterns.

    Args:
        patterns: The patterns to block.
    """
    # Normalize incoming sequence to a compact list of non-empty strings
    normalized: List[str] = [p for p in patterns if isinstance(p, str) and p]
    # Encode as JSON string for baggage
    payload = json.dumps(normalized)

    # Attach to current context
    token = otel_context.attach(
        baggage.set_baggage(_LOCAL_BLOCKED_SPANS_BAGGAGE_KEY, payload, context=otel_context.get_current())
    )
    try:
        yield
    finally:
        try:
            otel_context.detach(token)
        except Exception:
            # If context changed unexpectedly, avoid crashing user code
            logger.debug("Failed to detach local blocking context token", exc_info=True)
