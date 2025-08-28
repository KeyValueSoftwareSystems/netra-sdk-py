import logging
import re
from typing import Any, Dict, Optional, Union

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)


class ScrubbingSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """OpenTelemetry span processor that scrubs sensitive data from span attributes using pydantic logfire patterns."""

    # Common patterns for sensitive data detection (based on pydantic logfire scrubbing)
    SENSITIVE_PATTERNS = {
        # API keys first to avoid other patterns interfering
        "api_key": re.compile(
            r"(?:Token:\s*\S{32,})"  # scrub entire "Token: <value>" where value is 32+ non-space
            r"|(?:sk-[A-Za-z0-9]{16,})"  # scrub only the sk-... token (keep labels like "API Key:")
        ),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"),
        # Run credit card BEFORE SSN to avoid SSN partially matching inside card numbers
        "credit_card": re.compile(r"(?<!\d)(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})(?!\d)"),
        "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
        "password": re.compile(r"(?i)(?:password|passwd|pwd|secret|token)\s*[:=]\s*\S+"),
        "bearer_token": re.compile(r"(?i)(?:authorization:\s*)?bearer\s+[A-Za-z0-9\-._~+/]+=*"),
        "authorization": re.compile(r"(?i)authorization\s*:\s*\S+"),
    }

    # Sensitive attribute keys that should be scrubbed
    SENSITIVE_KEYS = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "key",
        "api_key",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "private_key",
        "access_token",
        "refresh_token",
        "session_token",
        "x-api-key",
        "x-auth-token",
        "cookie",
        "set-cookie",
    }

    def __init__(self):  # type: ignore[no-untyped-def]
        """Initialize the scrubbing span processor."""
        self.scrub_replacement = "[SCRUBBED]"

    def on_start(self, span: trace.Span, parent_context: Optional[otel_context.Context] = None) -> None:
        """Process span when it starts - no scrubbing needed here."""

    def on_end(self, span: trace.Span) -> None:
        """Scrub sensitive data from span attributes when span ends."""
        try:
            # Get span attributes
            if hasattr(span, "_attributes") and span._attributes:
                scrubbed_attributes = {}
                for key, value in span._attributes.items():
                    scrubbed_key, scrubbed_value = self._scrub_key_value(key, value)
                    scrubbed_attributes[scrubbed_key] = scrubbed_value

                # Replace the attributes with scrubbed versions
                span._attributes = scrubbed_attributes

        except Exception as e:
            logger.exception(f"Error scrubbing span attributes: {e}")

    def _scrub_key_value(self, key: str, value: Any) -> tuple[str, Any]:
        """Scrub sensitive data from a key-value pair.

        Args:
            key: The attribute key
            value: The attribute value

        Returns:
            Tuple of (scrubbed_key, scrubbed_value)
        """
        # Check if key itself is sensitive and value is a simple type (string, number, etc.)
        if self._is_sensitive_key(key) and not isinstance(value, (dict, list, tuple)):
            return key, self.scrub_replacement

        # Scrub value based on its type
        if isinstance(value, str):
            scrubbed_value = self._scrub_string_value(value)
            return key, scrubbed_value
        elif isinstance(value, dict):
            return key, self._scrub_dict_value(value)
        elif isinstance(value, (list, tuple)):
            return key, self._scrub_list_value(value)

        return key, value

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key is considered sensitive.

        Args:
            key: The key to check

        Returns:
            True if the key is sensitive, False otherwise
        """
        key_lower = key.lower()
        return any(sensitive_key in key_lower for sensitive_key in self.SENSITIVE_KEYS)

    def _scrub_string_value(self, value: str) -> str:
        """Scrub sensitive patterns from a string value.

        Args:
            value: The string value to scrub

        Returns:
            The scrubbed string value
        """
        scrubbed_value = value

        # Early catch-all for contiguous 13-19 digit sequences (credit/debit cards)
        scrubbed_value = re.sub(r"(?<!\d)\d{13,19}(?!\d)", self.scrub_replacement, scrubbed_value)

        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern.search(scrubbed_value):
                scrubbed_value = pattern.sub(self.scrub_replacement, scrubbed_value)

        # No extra fallback required now that we pre-scrub 13-19 digit sequences

        return scrubbed_value

    def _scrub_dict_value(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub sensitive data from a dictionary value.

        Args:
            value: The dictionary value to scrub

        Returns:
            The scrubbed dictionary value
        """
        scrubbed_dict = {}
        for k, v in value.items():
            scrubbed_k, scrubbed_v = self._scrub_key_value(k, v)
            scrubbed_dict[scrubbed_k] = scrubbed_v
        return scrubbed_dict

    def _scrub_list_value(self, value: Union[list, tuple]) -> Union[list, tuple] | None:  # type: ignore[type-arg]
        """Recursively scrub sensitive data from a list/tuple value.

        Args:
            value: The list/tuple value to scrub

        Returns:
            The scrubbed list/tuple value
        """
        scrubbed_items = []
        for item in value:
            if isinstance(item, str):
                scrubbed_items.append(self._scrub_string_value(item))
            elif isinstance(item, dict):
                scrubbed_items.append(self._scrub_dict_value(item))  # type: ignore[arg-type]
            elif isinstance(item, (list, tuple)):
                scrubbed_items.append(self._scrub_list_value(item))  # type: ignore[arg-type]
            else:
                scrubbed_items.append(item)

        return type(value)(scrubbed_items)

    def force_flush(self, timeout_millis: int = 30000) -> None:
        """Force flush - no-op for scrubbing processor."""

    def shutdown(self) -> None:
        """Shutdown - no-op for scrubbing processor."""
