"""ASGI middleware for FastAPI instrumentation.

Provides :class:`NetraFastAPIMiddleware`, a single-span ASGI middleware that
creates one SERVER span per request and captures structured ``input`` / ``output``
attributes matching the conventions used by the httpx instrumentation.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from fastapi import HTTPException
from opentelemetry import context as context_api
from opentelemetry.propagate import extract
from opentelemetry.propagators import textmap
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
from starlette.types import ASGIApp

from netra.instrumentation.fastapi.utils import (
    build_request_url,
    get_default_span_details,
    get_error_message,
    set_span_input,
    set_span_output,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

DEFAULT_ERROR_STATUS_CODE_RANGE = range(400, 600)


class _ASGIGetter(textmap.Getter[Dict[str, Any]]):  # type:ignore[misc]
    """W3C Trace Context-compliant header getter for ASGI scopes.

    Reads header values directly from the ASGI scope ``headers`` list,
    properly handling multi-value headers (e.g. multiple ``tracestate``
    entries) as required by RFC 7230 Section 3.2.2 and the W3C Trace
    Context specification.
    """

    def get(self, carrier: Dict[str, Any], key: str) -> Optional[List[str]]:
        """Return all values for the given header key.

        Args:
            carrier: The ASGI scope dictionary.
            key: The header name to look up (case-insensitive).

        Returns:
            A list of decoded header values, or None if the key is absent.
        """
        key_lower = key.lower()
        values = [
            value_bytes.decode("latin-1")
            for name_bytes, value_bytes in carrier.get("headers", [])
            if name_bytes.decode("latin-1").lower() == key_lower
        ]
        return values if values else None

    def keys(self, carrier: Dict[str, Any]) -> Sequence[str]:
        """Return all unique header names from the ASGI scope.

        Args:
            carrier: The ASGI scope dictionary.

        Returns:
            A list of unique lower-cased header names.
        """
        return list({name_bytes.decode("latin-1").lower() for name_bytes, _ in carrier.get("headers", [])})


_asgi_getter = _ASGIGetter()


class NetraFastAPIMiddleware:
    """ASGI middleware that creates a single SERVER span per HTTP request.

    Unlike ``OpenTelemetryMiddleware`` (which produces multiple child spans for
    receive/send events), this middleware produces exactly **one** span and
    populates structured ``input`` / ``output`` attributes in the same JSON
    format as the Netra httpx instrumentation.

    Trace context is extracted from incoming request headers so that the span
    correctly becomes a child of any upstream trace.

    Args:
        app: The inner ASGI application.
        tracer: The OpenTelemetry tracer to use for span creation.
        excluded_urls: An ``ExcludeList`` instance (from ``opentelemetry.util.http``)
            used to skip tracing for certain URL paths. May be ``None``.
        error_status_codes: HTTP status codes to treat as errors.
            Defaults to 400-599.
        error_messages: Optional mapping of status codes (or ranges) to
            custom error messages.
    """

    def __init__(
        self,
        app: ASGIApp,
        tracer: Tracer,
        excluded_urls: Any = None,
        error_status_codes: Optional[Iterable[int]] = None,
        error_messages: Optional[Dict[Union[int, range], str]] = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The inner ASGI application.
            tracer: The OpenTelemetry tracer for creating spans.
            excluded_urls: URL exclusion list, or None.
            error_status_codes: HTTP status codes considered errors.
            error_messages: Custom error message mapping.
        """
        self.app = app
        self.tracer = tracer
        self.excluded_urls = excluded_urls
        self.error_status_codes = set(error_status_codes or DEFAULT_ERROR_STATUS_CODE_RANGE)
        self.error_messages = error_messages or {}

    def _is_url_excluded(self, scope: Dict[str, Any]) -> Any:
        """Check whether the request path is in the exclusion list.

        Args:
            scope: The ASGI connection scope.

        Returns:
            True if the URL should be excluded from tracing.
        """
        if not self.excluded_urls:
            return False
        path = scope.get("path", "")
        return self.excluded_urls.url_disabled(path)

    @staticmethod
    def _build_span_attributes(scope: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble standard HTTP span attributes from the ASGI scope.

        Args:
            scope: The ASGI connection scope.

        Returns:
            A dict of span attributes.
        """
        attrs: Dict[str, Any] = {
            "http.method": scope.get("method", ""),
            "http.url": build_request_url(scope),
            "http.scheme": scope.get("scheme", ""),
            "http.target": scope.get("path", ""),
            "http.flavor": scope.get("http_version", ""),
        }

        server = scope.get("server")
        if server:
            host, port = server
            attrs["http.host"] = f"{host}:{port}"
            attrs["net.host.port"] = str(port)
            attrs["http.server_name"] = f"{host}:{port}"

        client = scope.get("client")
        if client:
            peer_ip, peer_port = client
            attrs["net.peer.ip"] = str(peer_ip)
            attrs["net.peer.port"] = str(peer_port)

        for name_bytes, value_bytes in scope.get("headers", []):
            if name_bytes.lower() == b"user-agent":
                attrs["http.user_agent"] = value_bytes.decode("latin-1")
                break

        return attrs

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        """ASGI interface entry point.

        For HTTP requests, creates a single SERVER span, captures the request
        body as ``input`` and the response body as ``output``, and handles
        error status codes.

        Non-HTTP scopes and excluded URLs are passed through without tracing.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self._is_url_excluded(scope):
            await self.app(scope, receive, send)
            return

        if should_suppress_instrumentation():
            await self.app(scope, receive, send)
            return

        ctx = extract(carrier=scope, getter=_asgi_getter)
        token = context_api.attach(ctx)

        try:
            await self._handle_request(scope, receive, send)
        finally:
            context_api.detach(token)

    async def _handle_request(
        self,
        scope: Dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Create a span and process the request/response cycle.

        Args:
            scope: The ASGI connection scope.
            receive: The ASGI receive callable.
            send: The ASGI send callable.
        """
        span_name, route_attrs = get_default_span_details(scope)
        span_attributes = self._build_span_attributes(scope)
        span_attributes.update(route_attrs)

        request_body_parts: List[bytes] = []
        response_status_code: List[int] = []
        response_headers: List[List[Tuple[bytes, bytes]]] = [[]]
        response_body_parts: List[bytes] = []

        async def capture_receive() -> Any:
            """Intercept ASGI receive to accumulate request body chunks."""
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body:
                    request_body_parts.append(body)
            return message

        async def capture_send(message: Dict[str, Any]) -> None:
            """Intercept ASGI send to capture response status, headers, and body."""
            if message["type"] == "http.response.start":
                response_status_code.append(message["status"])
                response_headers[0] = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_body_parts.append(body)
            await send(message)

        with self.tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER,
            attributes=span_attributes,
        ) as span:
            app_exception: Optional[Exception] = None

            try:
                await self.app(scope, capture_receive, capture_send)
            except Exception as exc:
                app_exception = exc

            try:
                set_span_input(span, scope, b"".join(request_body_parts))
            except Exception as e:
                logger.debug("netra.instrumentation.fastapi: failed to set span input: %s", e)

            if response_status_code:
                status_code = response_status_code[0]
                try:
                    span.set_attribute("http.status_code", status_code)
                    set_span_output(
                        span,
                        status_code,
                        response_headers[0],
                        b"".join(response_body_parts),
                    )
                except Exception as e:
                    logger.debug("netra.instrumentation.fastapi: failed to set span output: %s", e)

                self._finalize_span_status(span, status_code, app_exception)
            elif app_exception:
                span.record_exception(app_exception)
                span.set_status(Status(StatusCode.ERROR, str(app_exception)))

            if app_exception:
                raise app_exception

    def _finalize_span_status(
        self,
        span: Any,
        status_code: int,
        app_exception: Optional[Exception],
    ) -> None:
        """Set the final span status and record error exceptions.

        Args:
            span: The active OpenTelemetry span.
            status_code: The HTTP response status code.
            app_exception: An exception raised by the application, if any.
        """
        if app_exception:
            span.record_exception(app_exception)
            span.set_status(Status(StatusCode.ERROR, str(app_exception)))
            return

        if status_code in self.error_status_codes:
            error_msg = get_error_message(status_code, self.error_messages)
            exception = HTTPException(status_code=status_code, detail=error_msg)
            span.record_exception(exception)

            if status_code >= 500:
                span.set_status(Status(StatusCode.ERROR, error_msg))
            else:
                span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.OK))
