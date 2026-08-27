import logging
from typing import Any, Dict, Optional

import httpx
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.util.http import remove_url_credentials, sanitize_method

from netra.instrumentation.capture.bounded_capture import BoundedStreamBuffer
from netra.instrumentation.http.body import build_response_output, build_streaming_output
from netra.instrumentation.http.headers import sanitize_header_mapping

logger = logging.getLogger(__name__)


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
    method = sanitize_method(method.strip())
    if method == "_OTHER":
        return "HTTP"
    return method


def _get_request_body(request: httpx.Request) -> Optional[bytes]:
    """Return the raw request body bytes, or None when the request carries none.

    Parsing is left to the body pipeline so it happens under a size bound; a
    request body is application-controlled and can be arbitrarily large.

    Args:
        request: The httpx Request object.

    Returns:
        The raw body bytes, or None.
    """
    return request.content or None


def _get_response_body(response: httpx.Response) -> Optional[bytes]:
    """Return the raw response body bytes, or None when there are none to record.

    Args:
        response: The httpx Response object. Must already be read -- callers
            only reach this on the non-streaming path.

    Returns:
        The raw body bytes, or None when the response has no body or its
        content is not available.
    """
    try:
        return response.content or None
    except httpx.ResponseNotRead:
        # A streaming response the caller has not consumed. Reading it here
        # would drain the stream out from under them.
        logger.debug("netra.instrumentation.libraries.httpx: response not read, skipping body capture")
        return None


def set_span_input(span: Span, request: httpx.Request) -> None:
    """Serialize request data and set it as the span ``input`` attribute.

    Args:
        span: The active OpenTelemetry span.
        request: The outgoing httpx Request.
    """
    if not span.is_recording():
        return
    try:
        input_data: Dict[str, Any] = {
            "url": remove_url_credentials(str(request.url)),
            "headers": sanitize_header_mapping(request.headers),
        }
        span.set_attribute("input", build_response_output(input_data, _get_request_body(request)))
    except Exception as e:
        logger.error(f"Failed to set input attribute on httpx span: {e}")


def set_span_output(span: Span, response: httpx.Response) -> None:
    """Serialize response data and set it as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The received httpx Response.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": sanitize_header_mapping(response.headers),
        }
        span.set_attribute("output", build_response_output(output_data, _get_response_body(response)))
    except Exception as e:
        logger.error(f"Failed to set output attribute on httpx span: {e}")


def set_streaming_span_output(span: Span, response: httpx.Response, body_buffer: BoundedStreamBuffer) -> None:
    """Serialize accumulated streaming body bytes and set them as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The httpx Response whose headers/status are used.
        body_buffer: Body bytes accumulated during iteration, capped at the
            configured attribute length.
    """
    if not span.is_recording():
        return
    try:
        output_data: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": sanitize_header_mapping(response.headers),
        }
        span.set_attribute("output", build_streaming_output(output_data, body_buffer))
    except Exception as e:
        logger.error(f"Failed to set streaming output attribute on httpx span: {e}")
