"""Span processor for instrumentation name recording and attribute truncation."""

import logging
import os
from typing import Any, Callable, Optional, Set, Union

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from netra.config import Config
from netra.instrumentation.instruments import InstrumentSet

logger = logging.getLogger(__name__)

# Type aliases
TruncatableValue = Union[str, bytes, bytearray]
SetAttributeFunc = Callable[[str, Any], None]

# Constants
_OTEL_INSTRUMENTATION_PREFIX = "opentelemetry.instrumentation."
_NETRA_INSTRUMENTATION_PREFIX = "netra.instrumentation."
_HTTPX_INSTRUMENTATION = "httpx"
_URL_ATTRIBUTE_KEYS = frozenset({"http.url", "url.full"})
_DEFAULT_BLOCKED_URL_PATTERNS = frozenset({"getnetra", "githubusercontent"})


def _load_blocked_url_patterns() -> frozenset[str]:
    """Loads blocked URL patterns from defaults plus optional env var additions.

    Environment variable format:
        BLOCKED_URL_PATTERNS="pattern1,pattern2,pattern3"
    """
    try:
        patterns = set(_DEFAULT_BLOCKED_URL_PATTERNS)
        env_patterns = os.getenv("BLOCKED_URL_PATTERNS", "")

        for pattern in env_patterns.split(","):
            normalized_pattern = pattern.strip().lower()
            if normalized_pattern:
                patterns.add(normalized_pattern)

        return frozenset(patterns)
    except Exception as e:
        logger.warning(f"Failed to load blocked URL patterns: {e}")
        return _DEFAULT_BLOCKED_URL_PATTERNS


_BLOCKED_URL_PATTERNS = _load_blocked_url_patterns()

# Pre-computed allowed instrumentation names
_ALLOWED_INSTRUMENTATION_NAMES: Set[str] = {member.value for member in InstrumentSet}  # type: ignore[attr-defined]


class InstrumentationSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Span processor that records instrumentation names and truncates attribute values.
    The processor also marks spans as locally blocked when HTTP URLs match certain
    patterns (e.g., internal service URLs).
    """

    def on_start(
        self,
        span: Span,
        parent_context: Optional[otel_context.Context] = None,
    ) -> None:
        """Called when a span is started.

        Wraps the span's `set_attribute` method to enable value truncation and
        sets the instrumentation name attribute if applicable.

        Args:
            span: The span that was started.
            parent_context: The parent context of the span, if any.
        """
        try:
            self._wrap_set_attribute(span)
            self._set_instrumentation_name_attribute(span)
        except Exception:
            logger.exception("Error in on_start processing")

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended. No-op for this processor."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Forces export of all spans. No-op for this processor.

        Args:
            timeout_millis: Maximum time to wait for flush completion.

        Returns:
            True, indicating success (no-op).
        """
        return True

    def shutdown(self) -> None:
        """Shuts down the processor. No-op for this processor."""

    def _wrap_set_attribute(self, span: Span) -> None:
        """Wraps span.set_attribute to add truncation and URL blocking logic.

        Args:
            span: The span whose set_attribute method will be wrapped.
        """
        original_set_attribute: SetAttributeFunc = span.set_attribute
        instrumentation_name = self._extract_instrumentation_name(span)
        is_httpx = self._is_httpx_instrumentation(instrumentation_name)

        if is_httpx:
            self._check_and_mark_blocked_url(span, original_set_attribute)

        def wrapped_set_attribute(key: str, value: Any) -> None:
            self._handle_set_attribute(
                key=key,
                value=value,
                original_set_attribute=original_set_attribute,
                is_httpx=is_httpx,
            )

        setattr(span, "set_attribute", wrapped_set_attribute)

    def _handle_set_attribute(
        self,
        key: str,
        value: Any,
        original_set_attribute: SetAttributeFunc,
        is_httpx: bool,
    ) -> None:
        """Handles a set_attribute call with truncation and URL blocking.

        Args:
            key: The attribute key.
            value: The attribute value.
            original_set_attribute: The original set_attribute function.
            is_httpx: Whether this is an HTTPX instrumentation span.
        """
        try:
            if is_httpx and key in _URL_ATTRIBUTE_KEYS:
                self._mark_blocked_if_internal_url(original_set_attribute, value)

            truncated_value = self._truncate_value(value)
            original_set_attribute(key, truncated_value)
        except Exception:
            self._fallback_set_attribute(original_set_attribute, key, value)

    @staticmethod
    def _fallback_set_attribute(
        original_set_attribute: SetAttributeFunc,
        key: str,
        value: Any,
    ) -> None:
        """Attempts to set attribute without truncation as a fallback.

        Args:
            original_set_attribute: The original set_attribute function.
            key: The attribute key.
            value: The attribute value.
        """
        try:
            original_set_attribute(key, value)
        except Exception:
            pass

    def _set_instrumentation_name_attribute(self, span: Span) -> None:
        """Sets the instrumentation name attribute on the span if allowed.

        Args:
            span: The span to set the attribute on.
        """
        instrumentation_name = self._extract_instrumentation_name(span)
        if instrumentation_name in _ALLOWED_INSTRUMENTATION_NAMES:
            attribute_key = f"{Config.LIBRARY_NAME}.instrumentation.name"
            span.set_attribute(attribute_key, instrumentation_name)

    @staticmethod
    def _extract_instrumentation_name(span: Span) -> Optional[str]:
        """Extracts the instrumentation name from the span's scope.

        For scopes with known prefixes (opentelemetry.instrumentation.* or
        netra.instrumentation.*), returns just the final component.
        Otherwise, returns the full scope name.

        Args:
            span: The span to extract the instrumentation name from.

        Returns:
            The instrumentation name, or None if not available.
        """
        scope = getattr(span, "instrumentation_scope", None)
        if scope is None:
            return None

        name = getattr(scope, "name", None)
        if not isinstance(name, str) or not name:
            return None

        if name.startswith(_OTEL_INSTRUMENTATION_PREFIX) or name.startswith(_NETRA_INSTRUMENTATION_PREFIX):
            base_name = name.rsplit(".", 1)[-1].strip()
            return base_name if base_name else name

        return name

    @staticmethod
    def _is_httpx_instrumentation(instrumentation_name: Optional[str]) -> bool:
        """Checks if the instrumentation name indicates HTTPX.

        Args:
            instrumentation_name: The instrumentation name to check.

        Returns:
            True if this is HTTPX instrumentation, False otherwise.
        """
        if not instrumentation_name:
            return False
        return instrumentation_name.lower() == _HTTPX_INSTRUMENTATION

    def _check_and_mark_blocked_url(
        self,
        span: Span,
        original_set_attribute: SetAttributeFunc,
    ) -> None:
        """Checks existing URL attributes and marks span if URL is internal.

        Args:
            span: The span to check.
            original_set_attribute: The original set_attribute function.
        """
        for url_key in _URL_ATTRIBUTE_KEYS:
            url = self._get_span_attribute(span, url_key)
            if url is not None:
                self._mark_blocked_if_internal_url(original_set_attribute, url)
                return

    @staticmethod
    def _mark_blocked_if_internal_url(
        original_set_attribute: SetAttributeFunc,
        url: Any,
    ) -> None:
        """Marks the span as locally blocked if URL matches internal patterns.

        Args:
            original_set_attribute: The original set_attribute function.
            url: The URL to check.
        """
        if not isinstance(url, str):
            return

        url_lower = url.lower()
        for pattern in _BLOCKED_URL_PATTERNS:
            if pattern in url_lower:
                original_set_attribute("netra.local_blocked", True)
                return

    @staticmethod
    def _get_span_attribute(span: Span, key: str) -> Any:
        """Retrieves an attribute value from a span.

        Attempts to access attributes through both public and private interfaces
        for compatibility with different span implementations.

        Args:
            span: The span to get the attribute from.
            key: The attribute key.

        Returns:
            The attribute value, or None if not found.
        """
        for attr_name in ("attributes", "_attributes"):
            try:
                attrs = getattr(span, attr_name, None)
                if attrs is not None and hasattr(attrs, "get"):
                    value = attrs.get(key)
                    if value is not None:
                        return value
            except Exception:
                continue
        return None

    def _truncate_value(self, value: Any) -> Any:
        """Truncates string/bytes values to the configured maximum length.

        Handles nested structures (lists and dicts) by truncating string/bytes
        values within them.

        Args:
            value: The value to truncate.

        Returns:
            The truncated value, or the original value if truncation fails.
        """
        try:
            return self._do_truncate(value)
        except Exception:
            return value

    def _do_truncate(self, value: Any) -> Any:
        """Performs the actual truncation logic.

        Args:
            value: The value to truncate.

        Returns:
            The truncated value.
        """
        max_len = Config.ATTRIBUTE_MAX_LEN

        if isinstance(value, str):
            return value[:max_len] if len(value) > max_len else value

        if isinstance(value, (bytes, bytearray)):
            return value[:max_len] if len(value) > max_len else value

        if isinstance(value, list):
            return self._truncate_list(value)

        if isinstance(value, dict):
            return self._truncate_dict(value)

        return value

    def _truncate_list(self, items: list[Any]) -> list[Any]:
        """Truncates truncatable values within a list.

        Args:
            items: The list to process.

        Returns:
            A new list with truncated values.
        """
        return [self._do_truncate(item) if isinstance(item, (str, bytes, bytearray)) else item for item in items]

    def _truncate_dict(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Truncates truncatable values within a dict.

        Args:
            mapping: The dict to process.

        Returns:
            A new dict with truncated values.
        """
        return {
            key: self._do_truncate(val) if isinstance(val, (str, bytes, bytearray)) else val
            for key, val in mapping.items()
        }
