import logging
import threading
from typing import Optional

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource

from netra.config import Config

logger = logging.getLogger(__name__)

_provider_install_lock = threading.Lock()

# Map every OTel instrument type to DELTA so the backend receives
# incremental values on each export cycle, matching standard
# observability platform behavior (Datadog, Prometheus pull model, etc.)
_DELTA_TEMPORALITY: dict = {
    metrics.Counter: AggregationTemporality.DELTA,
    metrics.UpDownCounter: AggregationTemporality.DELTA,
    metrics.Histogram: AggregationTemporality.DELTA,
    metrics.ObservableCounter: AggregationTemporality.DELTA,
    metrics.ObservableUpDownCounter: AggregationTemporality.DELTA,
    metrics.ObservableGauge: AggregationTemporality.DELTA,
}


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

            exporter = OTLPMetricExporter(
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
                "Netra metrics pipeline started: endpoint=%s, interval=%dms",
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
