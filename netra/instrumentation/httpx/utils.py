from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import httpx
from opentelemetry.instrumentation._semconv import (
    _client_duration_attrs_new,
    _client_duration_attrs_old,
    _filter_semconv_duration_attrs,
    _report_new,
    _report_old,
    _set_http_host_client,
    _set_http_method,
    _set_http_net_peer_name_client,
    _set_http_peer_port_client,
    _set_http_scheme,
    _set_http_url,
    _set_status,
    _StabilityMode,
)
from opentelemetry.metrics import Histogram
from opentelemetry.semconv.attributes.network_attributes import (
    NETWORK_PEER_ADDRESS,
    NETWORK_PEER_PORT,
)
from opentelemetry.trace.span import Span
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


def get_default_span_name(method: str) -> str:
    method = sanitize_method(method.strip())
    if method == "_OTHER":
        return "HTTP"
    return method


def _sanitize_headers(headers: httpx.Headers) -> Dict[str, str]:
    return {k: "[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v for k, v in headers.items()}


def _get_request_body(request: httpx.Request) -> Any:
    content = request.content
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary content: {len(content)} bytes>"


def _get_response_body(response: httpx.Response) -> Any:
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


def set_span_input(span: Span, request: httpx.Request) -> None:
    if not span.is_recording():
        return
    try:
        input_data: Dict[str, Any] = {
            "method": request.method,
            "url": remove_url_credentials(str(request.url)),
            "headers": _sanitize_headers(request.headers),
        }
        body = _get_request_body(request)
        if body is not None:
            input_data["body"] = body
        span.set_attribute("input", json.dumps(input_data))
    except Exception:
        logger.debug("Failed to set input attribute on httpx span", exc_info=True)


def set_span_output(span: Span, response: httpx.Response) -> None:
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
        logger.debug("Failed to set output attribute on httpx span", exc_info=True)


def set_span_attributes(
    span_attributes: Dict[str, Any],
    metric_labels: Dict[str, Any],
    method: str,
    url: str,
    sem_conv_opt_in_mode: _StabilityMode,
) -> None:
    _set_http_method(span_attributes, method, sanitize_method(method), sem_conv_opt_in_mode)
    _set_http_url(span_attributes, url, sem_conv_opt_in_mode)
    _set_http_method(metric_labels, method, sanitize_method(method), sem_conv_opt_in_mode)

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme and _report_old(sem_conv_opt_in_mode):
            _set_http_scheme(metric_labels, parsed_url.scheme, sem_conv_opt_in_mode)
        if parsed_url.hostname:
            _set_http_host_client(metric_labels, parsed_url.hostname, sem_conv_opt_in_mode)
            _set_http_net_peer_name_client(metric_labels, parsed_url.hostname, sem_conv_opt_in_mode)
            if _report_new(sem_conv_opt_in_mode):
                _set_http_host_client(span_attributes, parsed_url.hostname, sem_conv_opt_in_mode)
                span_attributes[NETWORK_PEER_ADDRESS] = parsed_url.hostname
        if parsed_url.port:
            _set_http_peer_port_client(metric_labels, parsed_url.port, sem_conv_opt_in_mode)
            if _report_new(sem_conv_opt_in_mode):
                _set_http_peer_port_client(span_attributes, parsed_url.port, sem_conv_opt_in_mode)
                span_attributes[NETWORK_PEER_PORT] = parsed_url.port
    except ValueError as error:
        logger.error(error)


def set_http_status_code_attribute(
    span: Span,
    status_code: Union[int, str],
    metric_attributes: Optional[Dict[str, Any]] = None,
    sem_conv_opt_in_mode: _StabilityMode = _StabilityMode.DEFAULT,
) -> None:
    status_code_str = str(status_code)
    try:
        status_code_int = int(status_code)
    except ValueError:
        status_code_int = -1
    if metric_attributes is None:
        metric_attributes = {}
    _set_status(
        span,
        metric_attributes,
        status_code_int,
        status_code_str,
        server_span=False,
        sem_conv_opt_in_mode=sem_conv_opt_in_mode,
    )


def record_duration_metrics(
    duration_histogram_old: Optional[Histogram],
    duration_histogram_new: Optional[Histogram],
    elapsed_time: float,
    metric_labels: Dict[str, Any],
    sem_conv_opt_in_mode: _StabilityMode,
) -> None:
    if duration_histogram_old is not None:
        duration_attrs_old = _filter_semconv_duration_attrs(
            metric_labels,
            _client_duration_attrs_old,
            _client_duration_attrs_new,
            _StabilityMode.DEFAULT,
        )
        duration_histogram_old.record(max(round(elapsed_time * 1000), 0), attributes=duration_attrs_old)
    if duration_histogram_new is not None:
        duration_attrs_new = _filter_semconv_duration_attrs(
            metric_labels,
            _client_duration_attrs_old,
            _client_duration_attrs_new,
            _StabilityMode.HTTP,
        )
        duration_histogram_new.record(elapsed_time, attributes=duration_attrs_new)
