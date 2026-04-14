import json
import logging
from typing import Any, Dict, List

import httpx
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


def _sanitize_headers(headers: httpx.Headers) -> Dict[str, str]:
    """Redact sensitive header values.

    Args:
        headers: The httpx Headers mapping.

    Returns:
        A new dict with sensitive values replaced by "[REDACTED]".
    """
    return {k: "[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v for k, v in headers.items()}


def _get_request_body(request: httpx.Request) -> Any:
    """Extract and deserialize the request body.

    Args:
        request: The httpx Request object.

    Returns:
        The parsed JSON, decoded string, binary placeholder, or None.
    """
    content = request.content
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"Request body is not JSON, falling back to text: {e}")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary content: {len(content)} bytes>"


def _get_response_body(response: httpx.Response) -> Any:
    """Extract and deserialize the response body.

    Args:
        response: The httpx Response object.

    Returns:
        The parsed JSON, text content, or None.
    """
    try:
        return response.json()
    except Exception as e:
        logger.debug(f"Failed to parse response body: {e}")
    try:
        text = response.text
        if text:
            return text
    except Exception as e:
        logger.debug(f"Failed to parse response body: {e}")
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
            "headers": _sanitize_headers(request.headers),
        }
        body = _get_request_body(request)
        if body is not None:
            input_data["body"] = body
        span.set_attribute("input", json.dumps(input_data))
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
            "headers": _sanitize_headers(response.headers),
        }
        body = _get_response_body(response)
        if body is not None:
            output_data["body"] = body
        span.set_attribute("output", json.dumps(output_data))
    except Exception as e:
        logger.error(f"Failed to set output attribute on httpx span: {e}")


def _parse_streaming_body(accumulated: bytes) -> Any:
    """Parse accumulated streaming response bytes into a structured value.

    Handles SSE (``data: {...}``), NDJSON, plain concatenated JSON objects,
    and falls back to a decoded string or a binary placeholder.

    Args:
        accumulated: Raw bytes collected from the streaming response chunks.

    Returns:
        A parsed JSON object or list, a plain string, or a binary-size
        placeholder string if the bytes cannot be decoded as UTF-8.
    """
    try:
        text = accumulated.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary content: {len(accumulated)} bytes>"

    # SSE: any line starts with "data:"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(ln.startswith("data:") for ln in lines):
        parsed: List[Any] = []
        for ln in lines:
            if ln.startswith("data:"):
                data = ln[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    parsed.append(json.loads(data))
                except json.JSONDecodeError:
                    parsed.append(data)
        if parsed:
            return parsed[0] if len(parsed) == 1 else parsed

    # Sequential JSON decoding: handles single JSON, NDJSON, and bare concatenated objects
    decoder = json.JSONDecoder()
    results: List[Any] = []
    idx = 0
    stripped = text.strip()
    try:
        while idx < len(stripped):
            obj, end_idx = decoder.raw_decode(stripped, idx)
            results.append(obj)
            idx = end_idx
            while idx < len(stripped) and stripped[idx] in " \t\n\r":
                idx += 1
        if results and idx == len(stripped):
            return results[0] if len(results) == 1 else results
    except json.JSONDecodeError:
        pass

    return text


def set_streaming_span_output(span: Span, response: httpx.Response, chunks: List[bytes]) -> None:
    """Serialize accumulated streaming chunks and set them as the span ``output`` attribute.

    Args:
        span: The active OpenTelemetry span.
        response: The httpx Response whose headers/status are used.
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
            output_data["body"] = _parse_streaming_body(b"".join(chunks))
        span.set_attribute("output", json.dumps(output_data))
    except Exception as e:
        logger.error(f"Failed to set streaming output attribute on httpx span: {e}")
