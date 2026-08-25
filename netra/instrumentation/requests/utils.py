import json
import logging
from typing import Any, Dict

import requests as requests_lib  # type: ignore[import-untyped]
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.util.http import remove_url_credentials, sanitize_method

from netra.config import get_attribute_max_len
from netra.instrumentation.utils import (
    BoundedBodyBuffer,
    parse_streaming_body,
    serialize_bounded_output,
)

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


def set_streaming_span_output(span: Span, response: requests_lib.Response, body_buffer: BoundedBodyBuffer) -> None:
    """Serialize accumulated streaming body bytes and set them as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The requests Response whose headers/status are used.
        body_buffer: Body bytes accumulated during iteration, capped at the
            configured attribute length.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": _sanitize_headers(response.headers),
        }
        if body_buffer:
            max_len = get_attribute_max_len()
            parsed = parse_streaming_body(
                body_buffer.getvalue(),
                body_buffer.total_bytes,
                truncated=body_buffer.truncated,
                budget=max_len,
            )
            span.set_attribute(
                "output",
                serialize_bounded_output(
                    output_data,
                    parsed,
                    total_bytes=body_buffer.total_bytes,
                    truncated=body_buffer.truncated or parsed.truncated,
                    max_len=max_len,
                ),
            )
            return
        else:
            # Fallback: body was accessed via .content/.text rather than iterators
            body = _get_response_body(response)
            if body is not None:
                output_data["body"] = body
        span.set_attribute("output", json.dumps(output_data))
    except Exception:
        logger.debug("Failed to set streaming output attribute on requests span", exc_info=True)
