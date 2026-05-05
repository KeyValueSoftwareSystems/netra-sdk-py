"""Utility functions for FastAPI instrumentation.

Provides span naming, header sanitization, body parsing, and structured
``input`` / ``output`` attribute serialization mirroring the conventions
used by the httpx instrumentation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.util.http import sanitize_method
from starlette.routing import Match

logger = logging.getLogger(__name__)

_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed.

    Returns:
        True if the OpenTelemetry suppression key is active in the current
        context, False otherwise.
    """
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def get_route_details(scope: Dict[str, Any]) -> Optional[str]:
    """Retrieve the Starlette route path from an ASGI scope.

    Iterates over the application's registered routes and returns the
    first full match, falling back to a partial match if no full match
    is found.

    Args:
        scope: An ASGI/Starlette scope dictionary containing the ``app`` key.

    Returns:
        The matched route path string, or None if no route matches.
    """
    app = scope.get("app")
    if app is None:
        return None

    route = None
    for starlette_route in getattr(app, "routes", []):
        match, _ = starlette_route.matches(scope)
        if match == Match.FULL:
            route = starlette_route.path
            break
        if match == Match.PARTIAL:
            route = starlette_route.path

    return route


def get_default_span_details(scope: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Derive a span name and attributes from an ASGI scope.

    Uses the sanitized HTTP method to produce a descriptive span name.

    Args:
        scope: An ASGI/Starlette scope dictionary.

    Returns:
        A tuple of ``(span_name, attributes_dict)``.
    """
    method = sanitize_method(scope.get("method", "").strip())
    attributes: Dict[str, Any] = {}

    if method == "_OTHER":
        method = "HTTP"
    return method, attributes


def sanitize_headers(raw_headers: List[Tuple[bytes, bytes]]) -> Dict[str, str]:
    """Convert ASGI raw header pairs to a dict with sensitive values redacted.

    Args:
        raw_headers: List of ``(name_bytes, value_bytes)`` tuples from the
            ASGI scope ``headers`` key.

    Returns:
        A dict mapping lower-cased header names to their values, with
        sensitive headers replaced by ``"[REDACTED]"``.
    """
    result: Dict[str, str] = {}
    for name_bytes, value_bytes in raw_headers:
        name = name_bytes.decode("latin-1").lower()
        if name in _SENSITIVE_HEADERS:
            result[name] = "[REDACTED]"
        else:
            result[name] = value_bytes.decode("latin-1")
    return result


def build_request_url(scope: Dict[str, Any]) -> str:
    """Reconstruct the full request URL from an ASGI scope.

    Args:
        scope: An ASGI scope dictionary with ``scheme``, ``server``, ``path``,
            and ``query_string`` keys.

    Returns:
        The reconstructed URL string.
    """
    scheme = scope.get("scheme", "http")
    server = scope.get("server")
    path = scope.get("path", "/")
    query_string = scope.get("query_string", b"")

    if server:
        host, port = server
        # Wrap IPv6 addresses in brackets per RFC 3986
        if ":" in host:
            host = f"[{host}]"
        default_port = 443 if scheme == "https" else 80
        if port == default_port:
            url = f"{scheme}://{host}{path}"
        else:
            url = f"{scheme}://{host}:{port}{path}"
        url = path

    if query_string:
        url = f"{url}?{query_string.decode('latin-1')}"

    return url


def parse_body(raw: bytes) -> Any:
    """Parse raw bytes into a structured value.

    Attempts JSON first, then UTF-8 text, and falls back to a binary
    placeholder string.

    Args:
        raw: The raw body bytes.

    Returns:
        The parsed JSON object, a decoded string, a binary placeholder,
        or None if the input is empty.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    try:
        text = raw.decode("utf-8")
        if text:
            return text
    except UnicodeDecodeError:
        pass
    return f"<binary content: {len(raw)} bytes>"


def set_span_input(span: Span, scope: Dict[str, Any], body: bytes) -> None:
    """Serialize request data and set it as the span ``input`` attribute.

    Follows the same JSON structure as the httpx instrumentation:
    ``{"url": ..., "headers": ..., "body": ...}``.

    Args:
        span: The active OpenTelemetry span.
        scope: The ASGI scope dictionary.
        body: The raw request body bytes.
    """
    if not span.is_recording():
        return
    try:
        input_data: Dict[str, Any] = {
            "url": build_request_url(scope),
            "headers": sanitize_headers(scope.get("headers", [])),
        }
        parsed_body = parse_body(body)
        if parsed_body is not None:
            input_data["body"] = parsed_body
        span.set_attribute("input", json.dumps(input_data))
    except Exception as e:
        logger.error("Failed to set input attribute on FastAPI span: %s", e)


def set_span_output(
    span: Span,
    status_code: int,
    headers: List[Tuple[bytes, bytes]],
    body: bytes,
) -> None:
    """Serialize response data and set it as the span ``output`` attribute.

    Follows the same JSON structure as the httpx instrumentation:
    ``{"status_code": ..., "headers": ..., "body": ...}``.

    Args:
        span: The active OpenTelemetry span.
        status_code: The HTTP response status code.
        headers: Raw ASGI response header pairs.
        body: The raw response body bytes.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": status_code,
            "headers": sanitize_headers(headers),
        }
        parsed_body = parse_body(body)
        if parsed_body is not None:
            output_data["body"] = parsed_body
        span.set_attribute("output", json.dumps(output_data))
    except Exception as e:
        logger.error("Failed to set output attribute on FastAPI span: %s", e)


def get_error_message(
    status_code: int,
    error_messages: Dict[Union[int, range], str],
) -> str:
    """Resolve the error message for a given HTTP status code.

    Checks for an exact match first, then range matches, and finally
    falls back to a default based on the status code class.

    Args:
        status_code: The HTTP status code.
        error_messages: Mapping of status codes (or ranges) to custom messages.

    Returns:
        A descriptive error message string.
    """
    if status_code in error_messages:
        return error_messages[status_code]

    for key, message in error_messages.items():
        if isinstance(key, range) and status_code in key:
            return message

    if 400 <= status_code < 500:
        return f"Client Error: HTTP {status_code}"
    elif 500 <= status_code < 600:
        return f"Server Error: HTTP {status_code}"

    return f"HTTP {status_code} Error"
