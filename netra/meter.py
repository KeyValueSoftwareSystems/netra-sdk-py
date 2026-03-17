import json
import logging
import threading
from typing import Any, Optional

from google.protobuf.json_format import MessageToDict
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter, encode_metrics
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    MeterProvider,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExportResult,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource

from netra.config import Config

logger = logging.getLogger(__name__)

_provider_install_lock = threading.Lock()

# Map every OTel instrument type to DELTA so the backend receives
# incremental values on each export cycle, matching standard
# observability platform behavior (Datadog, Prometheus pull model, etc.)
# NOTE: Keys must be the SDK instrument classes (opentelemetry.sdk.metrics),
# not the public API classes (opentelemetry.metrics).
_DELTA_TEMPORALITY: dict = {
    Counter: AggregationTemporality.DELTA,
    UpDownCounter: AggregationTemporality.DELTA,
    Histogram: AggregationTemporality.DELTA,
    ObservableCounter: AggregationTemporality.DELTA,
    ObservableUpDownCounter: AggregationTemporality.DELTA,
    ObservableGauge: AggregationTemporality.DELTA,
}


class _JsonOTLPMetricExporter(OTLPMetricExporter):
    """Thin wrapper that sends OTLP metrics as JSON instead of protobuf.

    The upstream ``OTLPMetricExporter`` serialises to protobuf and sets
    ``Content-Type: application/x-protobuf``.  The Netra backend currently
    only reliably parses the JSON encoding (matching the JS SDK), so this
    subclass converts the protobuf ``ExportMetricsServiceRequest`` to its
    JSON representation before posting.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session.headers["Content-Type"] = "application/json"

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: Optional[float] = 10_000,
        **kwargs: Any,
    ) -> MetricExportResult:
        if self._shutdown:
            logger.warning("Exporter already shutdown, ignoring batch")
            return MetricExportResult.FAILURE

        pb_message = encode_metrics(metrics_data)
        payload = MessageToDict(pb_message, preserving_proto_field_name=True)
        serialized = json.dumps(payload).encode("utf-8")

        try:
            resp = self._session.post(
                url=self._endpoint,
                data=serialized,
                verify=self._certificate_file,
                timeout=self._timeout,
                cert=getattr(self, "_client_cert", None),
            )
        except ConnectionError:
            resp = self._session.post(
                url=self._endpoint,
                data=serialized,
                verify=self._certificate_file,
                timeout=self._timeout,
                cert=getattr(self, "_client_cert", None),
            )

        if resp.ok:
            return MetricExportResult.SUCCESS

        logger.error(
            "Failed to export metrics batch code: %s, reason: %s",
            resp.status_code,
            resp.text,
        )
        return MetricExportResult.FAILURE


class MetricsSetup:
    """
    Configures Netra's OpenTelemetry metrics pipeline.

    Sets up a MeterProvider backed by an OTLPMetricExporter that sends
    OTLP/HTTP JSON payloads to ``{otlp_endpoint}/v1/metrics`` at a
    configurable interval.  Delta temporality is used for all instrument
    types so the backend receives incremental values per export cycle.

    Usage::

        # Inside Netra.init() when enable_metrics=True
        MetricsSetup(cfg)

        # In application code
        meter = Netra.get_meter("my_service")
        counter = meter.create_counter("request_count")
        counter.add(1, {"route": "/api/health"})
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._setup_meter()

    def _setup_meter(self) -> None:
        """Install a global MeterProvider with an OTLP exporter."""
        if not self.cfg.otlp_endpoint:
            logger.warning(
                "OTLP endpoint not configured; metrics pipeline will not be started. "
                "Set NETRA_OTLP_ENDPOINT or pass otlp_endpoint to Netra.init()."
            )
            return

        with _provider_install_lock:
            current_provider = metrics.get_meter_provider()
            if isinstance(current_provider, MeterProvider):
                logger.info("Reusing existing MeterProvider; skipping metrics setup.")
                return

            resource_attrs = {
                SERVICE_NAME: self.cfg.app_name,
                DEPLOYMENT_ENVIRONMENT: self.cfg.environment,
            }
            if self.cfg.resource_attributes:
                resource_attrs.update(self.cfg.resource_attributes)

            resource = Resource(attributes=resource_attrs)
            metrics_endpoint = _format_metrics_endpoint(self.cfg.otlp_endpoint)

            exporter = _JsonOTLPMetricExporter(
                endpoint=metrics_endpoint,
                headers=self.cfg.headers,
                preferred_temporality=_DELTA_TEMPORALITY,
            )

            reader = PeriodicExportingMetricReader(
                exporter=exporter,
                export_interval_millis=self.cfg.metrics_export_interval_ms,
            )

            provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(provider)

            logger.info(
                "Netra metrics pipeline started (JSON): endpoint=%s, interval=%dms",
                metrics_endpoint,
                self.cfg.metrics_export_interval_ms,
            )


def _format_metrics_endpoint(endpoint: str) -> str:
    """Append ``/v1/metrics`` to the base OTLP endpoint if not already present."""
    if not endpoint.endswith("/v1/metrics"):
        return endpoint.rstrip("/") + "/v1/metrics"
    return endpoint


def get_meter(name: str = "netra", version: Optional[str] = None) -> metrics.Meter:
    """
    Return an OpenTelemetry ``Meter`` from the global metrics API.

    This is the standard OTel pattern used by Datadog, New Relic, and other
    platforms: the global API dispatches to whichever MeterProvider was
    installed by ``MetricsSetup``.  If metrics were not enabled, a no-op
    Meter is returned, so instrumented code never needs to check for ``None``.

    Args:
        name:    Instrumentation scope name — typically your module or service
                 name, e.g. ``"order_service"`` or ``"netra"``.
        version: Optional instrumentation scope version string.

    Returns:
        An OTel ``Meter`` instance.

    Example::

        meter = Netra.get_meter("payment_service")
        latency = meter.create_histogram("payment.latency_ms", unit="ms")
        latency.record(42, {"provider": "stripe"})
    """
    return metrics.get_meter(name, version or "")
