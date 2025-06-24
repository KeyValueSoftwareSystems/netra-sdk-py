from typing import Collection

from opentelemetry.instrumentation.openai.utils import is_metrics_enabled
from opentelemetry.instrumentation.openai.v1 import OpenAIV1Instrumentor
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.openai.batch_wrapper import (
    batch_create_wrapper,
    batch_retrieve_wrapper,
    batch_list_wrapper,
    batch_cancel_wrapper
)
from netra.instrumentation.openai.file_wrapper import (
    file_create_wrapper
)

_instruments = ("openai >= 1.0.0",)


class OpenAIV1BatchInstrumentor(OpenAIV1Instrumentor):
    """Extended OpenAI instrumentor that includes batch API instrumentation."""
    
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        # First, call the parent class instrumentation to set up all existing instrumentations
        super()._instrument(**kwargs)
        
        # Now add batch-specific instrumentation
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Set up batch-specific metrics if metrics are enabled
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            # Batch-specific metrics
            batch_operation_duration = meter.create_histogram(
                name="llm.batch.operation.duration",
                unit="s",
                description="Batch operation duration",
            )
            
            batch_exception_counter = meter.create_counter(
                name="llm.batch.exceptions",
                unit="time",
                description="Number of exceptions occurred during batch operations",
            )
            
            batch_status_counter = meter.create_counter(
                name="llm.batch.status",
                unit="batch",
                description="Number of batches by status",
            )
            
            batch_requests_counter = meter.create_counter(
                name="llm.batch.requests",
                unit="request",
                description="Number of requests in batches",
            )
            
            # File operation metrics
            file_operation_duration = meter.create_histogram(
                name="llm.file.operation.duration",
                unit="s",
                description="File operation duration",
            )
            
            file_exception_counter = meter.create_counter(
                name="llm.file.exceptions",
                unit="time",
                description="Number of exceptions occurred during file operations",
            )
            
            file_requests_counter = meter.create_counter(
                name="llm.file.requests",
                unit="request",
                description="Number of requests in files",
            )
        else:
            (
                batch_operation_duration,
                batch_exception_counter,
                batch_status_counter,
                batch_requests_counter,
                file_operation_duration,
                file_exception_counter,
                file_requests_counter,
            ) = (None, None, None, None, None, None, None)

        # Instrument batch API endpoints
        self._instrument_batch_operations(
            tracer,
            batch_operation_duration,
            batch_exception_counter,
            batch_status_counter,
            batch_requests_counter,
        )

        # Instrument file API endpoints
        self._instrument_file_operations(
            tracer,
            file_operation_duration,
            file_exception_counter,
            file_requests_counter,
        )

    def _instrument_batch_operations(
        self,
        tracer,
        duration_histogram,
        exception_counter,
        status_counter,
        requests_counter,
    ):
        """Instrument batch-specific operations."""
        
        # Batch create
        wrap_function_wrapper(
            "openai.resources.batches",
            "Batches.create",
            batch_create_wrapper(
                tracer,
                duration_histogram,
                exception_counter,
                requests_counter,
            ),
        )

        # Batch retrieve
        wrap_function_wrapper(
            "openai.resources.batches",
            "Batches.retrieve",
            batch_retrieve_wrapper(
                tracer,
                duration_histogram,
                exception_counter,
                status_counter,
            ),
        )

        # Batch list
        wrap_function_wrapper(
            "openai.resources.batches",
            "Batches.list",
            batch_list_wrapper(
                tracer,
                duration_histogram,
                exception_counter,
            ),
        )

        # Batch cancel
        wrap_function_wrapper(
            "openai.resources.batches",
            "Batches.cancel",
            batch_cancel_wrapper(
                tracer,
                duration_histogram,
                exception_counter,
                status_counter,
            ),
        )

    def _instrument_file_operations(
        self,
        tracer,
        duration_histogram,
        exception_counter,
        requests_counter,
    ):
        """Instrument file-specific operations."""
        
        # File create
        wrap_function_wrapper(
            "openai.resources.files",
            "Files.create",
            file_create_wrapper(
                tracer,
                duration_histogram,
                exception_counter,
                requests_counter,
            ),
        )

    def _uninstrument(self, **kwargs):
        # Call parent uninstrument method
        super()._uninstrument(**kwargs)
        # Add any batch-specific uninstrumentation if needed
        pass
