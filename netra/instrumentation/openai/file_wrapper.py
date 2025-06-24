import json
import logging
import time
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry import context as context_api
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.openai.utils import (
    dont_throw,
)
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_span_attribute,
    model_as_dict,
    _get_openai_base_url,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.openai.utils import _with_batch_telemetry_wrapper

logger = logging.getLogger(__name__)

# Span names for file operations
FILE_CREATE_SPAN_NAME = "openai.files.create"

# Use a generic request type for files operations
LLM_REQUEST_TYPE = "files"


@_with_batch_telemetry_wrapper
def file_create_wrapper(
        tracer: Tracer,
        duration_histogram: Histogram,
        exception_counter: Counter,
        requests_counter: Counter,
        wrapped,
        instance,
        args,
        kwargs,
):
    """Wrapper for file create operations."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    with tracer.start_span(
            FILE_CREATE_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE},
    ) as span:
        _handle_file_create_request(span, kwargs, instance)

        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
                "operation": "files.create",
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise e

        duration = end_time - start_time

        _handle_file_create_response(
            response,
            span,
            instance,
            duration_histogram,
            requests_counter,
            duration,
        )

        return response


def _handle_file_create_request(span, kwargs, instance):
    """Handle request attributes for file create."""
    _set_client_attributes(span, instance)
    
    # Set file-specific request attributes
    if "file" in kwargs:
        file_obj = kwargs["file"]
        if hasattr(file_obj, "name"):
            _set_span_attribute(span, "llm.request.file.name", file_obj.name)
    
    if "purpose" in kwargs:
        _set_span_attribute(span, "llm.request.file.purpose", kwargs["purpose"])


def _handle_file_create_response(
        response,
        span,
        instance=None,
        duration_histogram=None,
        requests_counter=None,
        duration=None,
):
    """Handle response attributes and metrics for file create."""
    response_dict = model_as_dict(response)
    
    _set_file_response_attributes(span, response_dict)
    
    if duration_histogram and duration is not None:
        _set_file_metrics(
            instance, duration_histogram, requests_counter, response_dict, duration, "files.create"
        )


def _set_file_metrics(
        instance, duration_histogram, counter, response_dict, duration, operation
):
    """Set file-specific metrics."""
    attributes = _get_file_shared_attributes(instance, operation)
    
    if response_dict:
        # Add file-specific attributes to metrics
        if "purpose" in response_dict:
            attributes["llm.file.purpose"] = response_dict["purpose"]
        if "bytes" in response_dict:
            attributes["llm.file.size_bytes"] = response_dict["bytes"]
    
    if duration_histogram:
        duration_histogram.record(duration, attributes=attributes)
    
    if counter:
        counter.add(1, attributes=attributes)


def _get_file_shared_attributes(instance, operation):
    """Get shared attributes for file operations."""
    return {
        "gen_ai.system": "openai",
        "gen_ai.operation.name": operation,
        "server.address": _get_openai_base_url(instance),
    }


def _set_file_response_attributes(span, response_dict):
    """Set common file response attributes."""
    if not response_dict:
        return
    
    # Set file response attributes
    _set_span_attribute(span, "llm.response.file.id", response_dict.get("id"))
    _set_span_attribute(span, "llm.response.file.object", response_dict.get("object"))
    _set_span_attribute(span, "llm.response.file.bytes", response_dict.get("bytes"))
    _set_span_attribute(span, "llm.response.file.created_at", response_dict.get("created_at"))
    _set_span_attribute(span, "llm.response.file.filename", response_dict.get("filename"))
    _set_span_attribute(span, "llm.response.file.purpose", response_dict.get("purpose"))
    _set_span_attribute(span, "llm.response.file.status", response_dict.get("status"))
    
    # Set status as OK for successful responses
    span.set_status(Status(StatusCode.OK))
