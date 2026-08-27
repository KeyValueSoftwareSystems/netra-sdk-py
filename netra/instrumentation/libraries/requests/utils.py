import logging
from typing import Any, Dict, Union

import requests as requests_lib  # type: ignore[import-untyped]
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.util.http import remove_url_credentials, sanitize_method

from netra.instrumentation.capture.bounded_capture import BoundedStreamBuffer
from netra.instrumentation.http.body import build_response_output, build_streaming_output
from netra.instrumentation.http.headers import sanitize_header_mapping

logger = logging.getLogger(__name__)

# A ``PreparedRequest`` body may be a generator or file object, and a streaming
# response's body has not arrived yet. Both are recorded as a description
# rather than read, because reading would consume what the caller is about to
# send or receive.
_UNREAD_REQUEST_BODY = "<streaming body>"
_UNREAD_RESPONSE_BODY = "<streaming response>"


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


def _get_request_body(request: requests_lib.PreparedRequest) -> Union[bytes, str, None]:
    """Return the raw request body, or None when the request carries none.

    Parsing is left to the body pipeline so it happens under a size bound; a
    request body is application-controlled and can be arbitrarily large.

    Args:
        request: The requests PreparedRequest object.

    Returns:
        The raw body bytes or text, :data:`_UNREAD_REQUEST_BODY` for a body
        that would have to be consumed to read, or None when there is no body.
    """
    body = request.body
    if body is None:
        return None
    if isinstance(body, (bytes, str)):
        return body or None
    return _UNREAD_REQUEST_BODY


def _get_response_body(response: requests_lib.Response) -> Union[bytes, str, None]:
    """Return the raw response body, or None when there is none to record.

    Args:
        response: The requests Response object.

    Returns:
        The raw body bytes, :data:`_UNREAD_RESPONSE_BODY` for a streaming
        response the caller has not consumed -- reading it here would force a
        full download and break their iterator -- or None when there is no body.
    """
    if not getattr(response, "_content_consumed", True):
        return _UNREAD_RESPONSE_BODY
    try:
        return response.content or None
    except requests_lib.RequestException:
        # The body failed at the HTTP layer (connection dropped mid-read, bad
        # chunked encoding). The status and headers are still worth recording,
        # so report no body rather than losing the whole attribute.
        logger.debug("Failed to read response content for span body", exc_info=True)
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
            "headers": sanitize_header_mapping(request.headers),
        }
        span.set_attribute("input", build_response_output(input_data, _get_request_body(request)))
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
            "headers": sanitize_header_mapping(response.headers),
        }
        span.set_attribute("output", build_response_output(output_data, _get_response_body(response)))
    except Exception:
        logger.debug("Failed to set output attribute on requests span", exc_info=True)


def set_streaming_span_output(span: Span, response: requests_lib.Response, body_buffer: BoundedStreamBuffer) -> None:
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
            "headers": sanitize_header_mapping(response.headers),
        }
        if body_buffer.total_bytes:
            span.set_attribute("output", build_streaming_output(output_data, body_buffer))
            return

        # Fallback: body was accessed via .content/.text rather than iterators
        span.set_attribute("output", build_response_output(output_data, _get_response_body(response)))
    except Exception:
        logger.debug("Failed to set streaming output attribute on requests span", exc_info=True)
