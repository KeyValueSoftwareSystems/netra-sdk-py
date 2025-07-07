"""
Unit tests for Tracer class.
Tests the happy path scenarios for OpenTelemetry tracer configuration.
"""

from unittest.mock import Mock, patch

from netra.config import Config
from netra.tracer import Tracer


class TestTracerInitialization:
    """Test tracer initialization and setup."""

    @patch("netra.tracer.trace")
    @patch("netra.tracer.TracerProvider")
    @patch("netra.tracer.Resource")
    @patch("netra.tracer.OTLPSpanExporter")
    @patch("netra.tracer.BatchSpanProcessor")
    @patch("netra.processors.SessionSpanProcessor")
    @patch("netra.processors.SpanAggregationProcessor")
    def test_tracer_initialization_with_otlp_endpoint(
        self,
        mock_span_agg_processor,
        mock_session_processor,
        mock_batch_processor,
        mock_otlp_exporter,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test tracer initialization with OTLP endpoint."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318"
        mock_config.headers = {"Authorization": "Bearer token"}
        mock_config.resource_attributes = {"custom.attr": "value"}
        mock_config.disable_batch = False

        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance

        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider

        mock_exporter = Mock()
        mock_otlp_exporter.return_value = mock_exporter

        mock_batch_proc = Mock()
        mock_batch_processor.return_value = mock_batch_proc

        mock_session_proc = Mock()
        mock_session_processor.return_value = mock_session_proc

        mock_agg_proc = Mock()
        mock_span_agg_processor.return_value = mock_agg_proc

        # Act
        Tracer(mock_config)

        # Assert
        # Verify resource creation with correct attributes
        expected_attrs = {"service.name": "test-app", "deployment.environment": "test", "custom.attr": "value"}
        mock_resource.assert_called_once_with(attributes=expected_attrs)

        # Verify tracer provider creation
        mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)

        # Verify OTLP exporter creation
        mock_otlp_exporter.assert_called_once_with(
            endpoint="http://localhost:4318/v1/traces", headers={"Authorization": "Bearer token"}
        )

        # Verify span processors are added
        mock_provider.add_span_processor.assert_any_call(mock_session_proc)
        mock_provider.add_span_processor.assert_any_call(mock_agg_proc)
        mock_provider.add_span_processor.assert_any_call(mock_batch_proc)

        # Verify batch processor creation
        mock_batch_processor.assert_called_once_with(mock_exporter)

        # Verify global tracer provider is set
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)

    @patch("netra.tracer.trace")
    @patch("netra.tracer.TracerProvider")
    @patch("netra.tracer.Resource")
    @patch("netra.tracer.ConsoleSpanExporter")
    @patch("netra.tracer.SimpleSpanProcessor")
    @patch("netra.processors.SessionSpanProcessor")
    @patch("netra.processors.SpanAggregationProcessor")
    def test_tracer_initialization_with_console_fallback(
        self,
        mock_span_agg_processor,
        mock_session_processor,
        mock_simple_processor,
        mock_console_exporter,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test tracer initialization with console exporter fallback."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "production"
        mock_config.otlp_endpoint = None  # No OTLP endpoint
        mock_config.headers = {}
        mock_config.resource_attributes = None
        mock_config.disable_batch = True  # Use simple processor

        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance

        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider

        mock_exporter = Mock()
        mock_console_exporter.return_value = mock_exporter

        mock_simple_proc = Mock()
        mock_simple_processor.return_value = mock_simple_proc

        # Act
        Tracer(mock_config)

        # Assert
        # Verify resource creation with basic attributes only
        expected_attrs = {"service.name": "test-app", "deployment.environment": "production"}
        mock_resource.assert_called_once_with(attributes=expected_attrs)

        # Verify console exporter is used
        mock_console_exporter.assert_called_once()

        # Verify simple processor is used instead of batch
        mock_simple_processor.assert_called_once_with(mock_exporter)
        mock_provider.add_span_processor.assert_any_call(mock_simple_proc)

    @patch("netra.tracer.trace")
    @patch("netra.tracer.TracerProvider")
    @patch("netra.tracer.Resource")
    @patch("netra.tracer.OTLPSpanExporter")
    @patch("netra.tracer.BatchSpanProcessor")
    @patch("netra.processors.SessionSpanProcessor")
    @patch("netra.processors.SpanAggregationProcessor")
    def test_tracer_initialization_with_minimal_config(
        self,
        mock_span_agg_processor,
        mock_session_processor,
        mock_batch_processor,
        mock_otlp_exporter,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test tracer initialization with minimal configuration."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "minimal-app"
        mock_config.environment = "dev"
        mock_config.otlp_endpoint = "http://jaeger:14268"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = False

        mock_resource_instance = Mock()
        mock_resource.return_value = mock_resource_instance

        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider

        # Act
        Tracer(mock_config)

        # Assert
        # Verify resource creation with minimal attributes
        expected_attrs = {"service.name": "minimal-app", "deployment.environment": "dev"}
        mock_resource.assert_called_once_with(attributes=expected_attrs)

        # Verify OTLP exporter with empty headers
        mock_otlp_exporter.assert_called_once_with(endpoint="http://jaeger:14268/v1/traces", headers={})


class TestTracerEndpointFormatting:
    """Test endpoint URL formatting functionality."""

    def test_format_endpoint_adds_traces_suffix(self):
        """Test that endpoint formatting adds /v1/traces suffix."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = False

        with (
            patch("netra.tracer.trace"),
            patch("netra.tracer.TracerProvider"),
            patch("netra.tracer.Resource"),
            patch("netra.tracer.OTLPSpanExporter"),
            patch("netra.tracer.BatchSpanProcessor"),
            patch("netra.processors.SessionSpanProcessor"),
            patch("netra.processors.SpanAggregationProcessor"),
        ):

            tracer = Tracer(mock_config)

            # Act
            result = tracer._format_endpoint("http://localhost:4318")

            # Assert
            assert result == "http://localhost:4318/v1/traces"

    def test_format_endpoint_preserves_existing_traces_suffix(self):
        """Test that endpoint formatting preserves existing /v1/traces suffix."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318/v1/traces"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = False

        with (
            patch("netra.tracer.trace"),
            patch("netra.tracer.TracerProvider"),
            patch("netra.tracer.Resource"),
            patch("netra.tracer.OTLPSpanExporter"),
            patch("netra.tracer.BatchSpanProcessor"),
            patch("netra.processors.SessionSpanProcessor"),
            patch("netra.processors.SpanAggregationProcessor"),
        ):

            tracer = Tracer(mock_config)

            # Act
            result = tracer._format_endpoint("http://localhost:4318/v1/traces")

            # Assert
            assert result == "http://localhost:4318/v1/traces"

    def test_format_endpoint_removes_trailing_slash(self):
        """Test that endpoint formatting removes trailing slash before adding suffix."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318/"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = False

        with (
            patch("netra.tracer.trace"),
            patch("netra.tracer.TracerProvider"),
            patch("netra.tracer.Resource"),
            patch("netra.tracer.OTLPSpanExporter"),
            patch("netra.tracer.BatchSpanProcessor"),
            patch("netra.processors.SessionSpanProcessor"),
            patch("netra.processors.SpanAggregationProcessor"),
        ):

            tracer = Tracer(mock_config)

            # Act
            result = tracer._format_endpoint("http://localhost:4318/")

            # Assert
            assert result == "http://localhost:4318/v1/traces"


class TestTracerConfiguration:
    """Test tracer configuration scenarios."""

    @patch("netra.tracer.trace")
    @patch("netra.tracer.TracerProvider")
    @patch("netra.tracer.Resource")
    @patch("netra.tracer.OTLPSpanExporter")
    @patch("netra.tracer.BatchSpanProcessor")
    @patch("netra.processors.SessionSpanProcessor")
    @patch("netra.processors.SpanAggregationProcessor")
    def test_tracer_with_custom_resource_attributes(
        self,
        mock_span_agg_processor,
        mock_session_processor,
        mock_batch_processor,
        mock_otlp_exporter,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test tracer initialization with custom resource attributes."""
        # Arrange
        custom_attrs = {"service.version": "1.0.0", "service.instance.id": "instance-123", "custom.team": "ai-team"}

        mock_config = Mock(spec=Config)
        mock_config.app_name = "custom-app"
        mock_config.environment = "staging"
        mock_config.otlp_endpoint = "https://api.honeycomb.io"
        mock_config.headers = {"x-honeycomb-team": "team-key"}
        mock_config.resource_attributes = custom_attrs
        mock_config.disable_batch = False

        # Act
        Tracer(mock_config)

        # Assert
        expected_attrs = {
            "service.name": "custom-app",
            "deployment.environment": "staging",
            "service.version": "1.0.0",
            "service.instance.id": "instance-123",
            "custom.team": "ai-team",
        }
        mock_resource.assert_called_once_with(attributes=expected_attrs)

    @patch("netra.tracer.trace")
    @patch("netra.tracer.TracerProvider")
    @patch("netra.tracer.Resource")
    @patch("netra.tracer.OTLPSpanExporter")
    @patch("netra.tracer.SimpleSpanProcessor")
    @patch("netra.processors.SessionSpanProcessor")
    @patch("netra.processors.SpanAggregationProcessor")
    def test_tracer_with_batch_disabled(
        self,
        mock_span_agg_processor,
        mock_session_processor,
        mock_simple_processor,
        mock_otlp_exporter,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test tracer initialization with batch processing disabled."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "sync-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = True

        mock_provider = Mock()
        mock_tracer_provider.return_value = mock_provider

        mock_exporter = Mock()
        mock_otlp_exporter.return_value = mock_exporter

        mock_simple_proc = Mock()
        mock_simple_processor.return_value = mock_simple_proc

        # Act
        Tracer(mock_config)

        # Assert
        # Verify simple processor is used
        mock_simple_processor.assert_called_once_with(mock_exporter)
        mock_provider.add_span_processor.assert_any_call(mock_simple_proc)

    def test_tracer_stores_config_reference(self):
        """Test that tracer stores reference to configuration."""
        # Arrange
        mock_config = Mock(spec=Config)
        mock_config.app_name = "test-app"
        mock_config.environment = "test"
        mock_config.otlp_endpoint = "http://localhost:4318"
        mock_config.headers = {}
        mock_config.resource_attributes = {}
        mock_config.disable_batch = False

        with (
            patch("netra.tracer.trace"),
            patch("netra.tracer.TracerProvider"),
            patch("netra.tracer.Resource"),
            patch("netra.tracer.OTLPSpanExporter"),
            patch("netra.tracer.BatchSpanProcessor"),
            patch("netra.processors.SessionSpanProcessor"),
            patch("netra.processors.SpanAggregationProcessor"),
        ):

            # Act
            tracer = Tracer(mock_config)

            # Assert
            assert tracer.cfg == mock_config
