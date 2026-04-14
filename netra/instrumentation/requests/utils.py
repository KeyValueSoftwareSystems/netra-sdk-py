import json
import logging
from typing import Any, Dict, List

import requests as requests_lib  # type: ignore[import-untyped]
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.util.http import remove_url_credentials, sanitize_method

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
    """Check if instrumentation should be suppressed."""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def get_default_span_name(method: str) -> str:
    """Derive a span name from the HTTP method.

    Args:
        method: The raw HTTP method string.

    Returns:
        The sanitized method (e.g. "GET") or "HTTP" for non-standard methods.
    """
    if not method:
        return "HTTP"
    method = sanitize_method(method.strip())
    if method == "_OTHER":
        return "HTTP"
    return method


def _sanitize_headers(headers: Any) -> Dict[str, str]:
    """Redact sensitive header values.

    Args:
        headers: A mapping of header names to values.

    Returns:
        A new dict with sensitive values replaced by "[REDACTED]".
    """
    return {k: "[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v for k, v in headers.items()}


def _get_request_body(request: requests_lib.PreparedRequest) -> Any:
    """Extract and deserialize the request body.

    Args:
        request: The requests PreparedRequest object.

    Returns:
        The parsed JSON, decoded string, streaming placeholder, or None.
    """
    body = request.body
    if body is None:
        return None
    if isinstance(body, bytes):
        if not body:
            return None
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary content: {len(body)} bytes>"
    if isinstance(body, str):
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    return "<streaming body>"


def _get_response_body(response: requests_lib.Response) -> Any:
    """Extract and deserialize the response body.

    Skips body capture for streaming responses whose content has not yet been
    consumed, to avoid forcing a full download and breaking downstream readers.

    Args:
        response: The requests Response object.

    Returns:
        The parsed JSON, text content, or None.
    """
    if not getattr(response, "_content_consumed", True):
        return "<streaming response>"
    try:
        return response.json()
    except Exception:
        pass
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass
    return None


def set_span_input(span: Span, request: requests_lib.PreparedRequest) -> None:
    """Serialize request data and set it as the span ``input`` attribute.

    Args:
        span: The active OpenTelemetry span.
        request: The outgoing PreparedRequest.
    """
    if not span.is_recording():
        return
    try:
        input_data: Dict[str, Any] = {
            "url": remove_url_credentials(request.url or ""),
            "headers": _sanitize_headers(request.headers),
        }
        body = _get_request_body(request)
        if body is not None:
            input_data["body"] = body
        span.set_attribute("input", json.dumps(input_data))
    except Exception:
        logger.debug("Failed to set input attribute on requests span", exc_info=True)


def set_span_output(span: Span, response: requests_lib.Response) -> None:
    """Serialize response data and set it as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The received Response.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": _sanitize_headers(response.headers),
        }
        body = _get_response_body(response)
        if body is not None:
            output_data["body"] = body
        span.set_attribute("output", json.dumps(output_data))
    except Exception:
        logger.debug("Failed to set output attribute on requests span", exc_info=True)


def set_streaming_span_output(span: Span, response: requests_lib.Response, chunks: List[bytes]) -> None:
    """Serialize accumulated streaming chunks and set them as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The requests Response whose headers/status are used.
        chunks: Raw bytes chunks accumulated during iteration.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": _sanitize_headers(response.headers),
        }
        if chunks:
            accumulated = b"".join(chunks)
            try:
                body: Any = json.loads(accumulated)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    body = accumulated.decode("utf-8")
                except UnicodeDecodeError:
                    body = f"<binary content: {len(accumulated)} bytes>"
            output_data["body"] = body
        span.set_attribute("output", json.dumps(output_data))
    except Exception:
        logger.debug("Failed to set streaming output attribute on requests span", exc_info=True)
