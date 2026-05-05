"""FastAPI instrumentation for Netra SDK.

Provides :class:`NetraFastAPIInstrumentor`, an OpenTelemetry instrumentor that
produces a single SERVER span per request with structured ``input`` / ``output``
attributes and automatic HTTP error status-code monitoring.
"""

from __future__ import annotations

import functools
import logging
import types
import weakref
from typing import Any, Collection, Dict, Iterable, Optional, Union

import fastapi
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import TracerProvider, get_tracer
from opentelemetry.util.http import get_excluded_urls, parse_excluded_urls
from starlette.applications import Starlette
from starlette.types import ASGIApp

from netra.instrumentation.fastapi.middleware import NetraFastAPIMiddleware
from netra.instrumentation.fastapi.utils import (
    get_default_span_details,
    get_route_details,
)
from netra.instrumentation.fastapi.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("fastapi >= 0.58.0",)
_excluded_urls_from_env = get_excluded_urls("FASTAPI")


class NetraFastAPIInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """OpenTelemetry instrumentor for FastAPI.

    Produces a **single** SERVER span per request (no child receive/send spans)
    with structured ``input`` / ``output`` attributes matching the Netra httpx
    instrumentation conventions. Automatically records ``HTTPException`` events
    for configurable error status codes.
    """

    _original_fastapi: Optional[type[fastapi.FastAPI]] = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the package dependencies required by this instrumentor.

        Returns:
            A collection of pip requirement strings.
        """
        return _instruments

    @staticmethod
    def instrument_app(
        app: fastapi.FastAPI,
        tracer_provider: Optional[TracerProvider] = None,
        excluded_urls: Optional[str] = None,
        error_status_codes: Optional[Iterable[int]] = None,
        error_messages: Optional[Dict[Union[int, range], str]] = None,
    ) -> None:
        """Instrument a FastAPI application.

        Monkey-patches the application's middleware stack to insert a
        :class:`~netra.instrumentation.fastapi.middleware.NetraFastAPIMiddleware`
        that creates a single SERVER span with ``input`` / ``output`` capture.

        Args:
            app: The FastAPI application to instrument.
            tracer_provider: Tracer provider to use. Falls back to the global provider.
            excluded_urls: Comma-delimited regexes of URLs to exclude from tracing.
            error_status_codes: HTTP status codes to treat as errors.
                Defaults to 400-599.
            error_messages: Mapping of status codes or ranges to custom error messages.
        """
        if not hasattr(app, "_is_instrumented_by_opentelemetry"):
            app._is_instrumented_by_opentelemetry = False

        if getattr(app, "_is_instrumented_by_opentelemetry", False):
            logger.warning("Attempting to instrument FastAPI app while already instrumented")
            return

        if excluded_urls is None:
            excluded_urls = _excluded_urls_from_env
        else:
            excluded_urls = parse_excluded_urls(excluded_urls)

        try:
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error("Failed to initialize tracer: %s", e)
            return

        def build_middleware_stack(self_app: Starlette) -> ASGIApp:
            """Build the middleware stack with Netra FastAPI middleware."""
            inner: ASGIApp = self_app._original_build_middleware_stack()

            return NetraFastAPIMiddleware(
                inner,
                tracer=tracer,
                excluded_urls=excluded_urls,
                error_status_codes=error_status_codes,
                error_messages=error_messages,
            )

        app._original_build_middleware_stack = app.build_middleware_stack
        app.build_middleware_stack = types.MethodType(
            functools.wraps(app.build_middleware_stack)(build_middleware_stack),
            app,
        )

        app._is_instrumented_by_opentelemetry = True
        if app not in _InstrumentedFastAPI._instrumented_fastapi_apps:
            _InstrumentedFastAPI._instrumented_fastapi_apps.add(app)

    @staticmethod
    def uninstrument_app(app: fastapi.FastAPI) -> None:
        """Remove instrumentation from a FastAPI application.

        Args:
            app: The FastAPI application to uninstrument.
        """
        original_build_middleware_stack = getattr(app, "_original_build_middleware_stack", None)
        if original_build_middleware_stack:
            app.build_middleware_stack = original_build_middleware_stack
            del app._original_build_middleware_stack
        app.middleware_stack = app.build_middleware_stack()
        app._is_instrumented_by_opentelemetry = False

    def _instrument(self, **kwargs: Any) -> None:
        """Replace ``fastapi.FastAPI`` with an auto-instrumenting subclass.

        Args:
            **kwargs: Accepts ``tracer_provider``, ``excluded_urls``,
                ``error_status_codes``, and ``error_messages``.
        """
        self._original_fastapi = fastapi.FastAPI

        _InstrumentedFastAPI._tracer_provider = kwargs.get("tracer_provider")
        _InstrumentedFastAPI._excluded_urls = kwargs.get("excluded_urls")
        _InstrumentedFastAPI._error_status_codes = kwargs.get("error_status_codes")
        _InstrumentedFastAPI._error_messages = kwargs.get("error_messages")

        fastapi.FastAPI = _InstrumentedFastAPI

    def _uninstrument(self, **kwargs: Any) -> None:
        """Restore the original ``fastapi.FastAPI`` class and uninstrument all apps.

        Args:
            **kwargs: Unused; present for interface compatibility.
        """
        for instance in list(_InstrumentedFastAPI._instrumented_fastapi_apps):
            self.uninstrument_app(instance)
        _InstrumentedFastAPI._instrumented_fastapi_apps.clear()
        if self._original_fastapi is not None:
            fastapi.FastAPI = self._original_fastapi
            self._original_fastapi = None


# Backward-compatible alias
FastAPIInstrumentor = NetraFastAPIInstrumentor


class _InstrumentedFastAPI(fastapi.FastAPI):  # type: ignore[misc]
    """FastAPI subclass that auto-instruments on construction.

    When :meth:`NetraFastAPIInstrumentor._instrument` replaces
    ``fastapi.FastAPI`` with this class, every new FastAPI application is
    automatically instrumented with the configured options.
    """

    _tracer_provider: Optional[TracerProvider] = None
    _excluded_urls: Optional[str] = None
    _error_status_codes: Optional[Iterable[int]] = None
    _error_messages: Optional[Dict[Union[int, range], str]] = None

    _instrumented_fastapi_apps: weakref.WeakSet[fastapi.FastAPI] = weakref.WeakSet()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize and auto-instrument the FastAPI application.

        Args:
            *args: Positional arguments forwarded to ``fastapi.FastAPI``.
            **kwargs: Keyword arguments forwarded to ``fastapi.FastAPI``.
        """
        super().__init__(*args, **kwargs)
        NetraFastAPIInstrumentor.instrument_app(
            self,
            tracer_provider=self._tracer_provider,
            excluded_urls=self._excluded_urls,
            error_status_codes=self._error_status_codes,
            error_messages=self._error_messages,
        )
        _InstrumentedFastAPI._instrumented_fastapi_apps.add(self)


# Backward-compatible aliases for utility functions
_get_route_details = get_route_details
_get_default_span_details = get_default_span_details
