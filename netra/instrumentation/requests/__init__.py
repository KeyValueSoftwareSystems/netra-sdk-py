from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry.instrumentation._semconv import (
    HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
    HTTP_DURATION_HISTOGRAM_BUCKETS_OLD,
    _get_schema_url,
    _OpenTelemetrySemanticConventionStability,
    _OpenTelemetryStabilitySignalType,
    _report_new,
    _report_old,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.metrics import MetricInstruments
from opentelemetry.semconv.metrics.http_metrics import HTTP_CLIENT_REQUEST_DURATION
from opentelemetry.trace import get_tracer
from opentelemetry.util.http import get_excluded_urls, parse_excluded_urls

from netra.instrumentation.requests.version import __version__
from netra.instrumentation.requests.wrappers import instrument, uninstrument

logger = logging.getLogger(__name__)

_instruments = ("requests >= 2.0.0",)

_excluded_urls_from_env = get_excluded_urls("REQUESTS")


class RequestsInstrumentor(BaseInstrumentor):  # type: ignore
    """An instrumentor for the requests library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        semconv_opt_in_mode = _OpenTelemetrySemanticConventionStability._get_opentelemetry_stability_opt_in_mode(
            _OpenTelemetryStabilitySignalType.HTTP,
        )
        schema_url = _get_schema_url(semconv_opt_in_mode)
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider, schema_url=schema_url)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider, schema_url=schema_url)

        duration_histogram_boundaries = kwargs.get("duration_histogram_boundaries")

        duration_histogram_old = None
        if _report_old(semconv_opt_in_mode):
            duration_histogram_old = meter.create_histogram(
                name=MetricInstruments.HTTP_CLIENT_DURATION,
                unit="ms",
                description="measures the duration of the outbound HTTP request",
                explicit_bucket_boundaries_advisory=duration_histogram_boundaries
                or HTTP_DURATION_HISTOGRAM_BUCKETS_OLD,
            )

        duration_histogram_new = None
        if _report_new(semconv_opt_in_mode):
            duration_histogram_new = meter.create_histogram(
                name=HTTP_CLIENT_REQUEST_DURATION,
                unit="s",
                description="Duration of HTTP client requests.",
                explicit_bucket_boundaries_advisory=duration_histogram_boundaries
                or HTTP_DURATION_HISTOGRAM_BUCKETS_NEW,
            )

        excluded_urls = kwargs.get("excluded_urls")
        instrument(
            tracer,
            duration_histogram_old,
            duration_histogram_new,
            request_hook=kwargs.get("request_hook"),
            response_hook=kwargs.get("response_hook"),
            excluded_urls=(_excluded_urls_from_env if excluded_urls is None else parse_excluded_urls(excluded_urls)),
            sem_conv_opt_in_mode=semconv_opt_in_mode,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        uninstrument()
