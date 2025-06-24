import json
import logging
import time

from opentelemetry import context as context_api
from opentelemetry.instrumentation.openai.shared import (
    _set_client_attributes,
    _set_span_attribute,
    model_as_dict,
    _get_openai_base_url,
)
from opentelemetry.instrumentation.openai.utils import (
    dont_throw,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.openai.utils import _with_batch_telemetry_wrapper

logger = logging.getLogger(__name__)

# Span names for different batch operations
BATCH_CREATE_SPAN_NAME = "openai.batch.create"
BATCH_RETRIEVE_SPAN_NAME = "openai.batch.retrieve"
BATCH_LIST_SPAN_NAME = "openai.batch.list"
BATCH_CANCEL_SPAN_NAME = "openai.batch.cancel"

LLM_REQUEST_TYPE = LLMRequestTypeValues.BATCH if hasattr(LLMRequestTypeValues, 'BATCH') else "batch"


@_with_batch_telemetry_wrapper
def batch_create_wrapper(
        tracer: Tracer,
        duration_histogram: Histogram,
        exception_counter: Counter,
        requests_counter: Counter,
        wrapped,
        instance,
        args,
        kwargs,
):
    """Wrapper for batch create operations."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    with tracer.start_span(
            BATCH_CREATE_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE},
    ) as span:
        _handle_batch_create_request(span, kwargs, instance)

        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
                "operation": "batch.create",
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise e

        duration = end_time - start_time

        _handle_batch_create_response(
            response,
            span,
            instance,
            duration_histogram,
            requests_counter,
            duration,
        )

        return response


@_with_batch_telemetry_wrapper
def batch_retrieve_wrapper(
        tracer: Tracer,
        duration_histogram: Histogram,
        exception_counter: Counter,
        status_counter: Counter,
        wrapped,
        instance,
        args,
        kwargs,
):
    """Wrapper for batch retrieve operations."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    with tracer.start_span(
            BATCH_RETRIEVE_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE},
    ) as span:
        _handle_batch_retrieve_request(span, kwargs, args, instance)

        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
                "operation": "batch.retrieve",
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise e

        duration = end_time - start_time

        _handle_batch_retrieve_response(
            response,
            span,
            instance,
            duration_histogram,
            status_counter,
            duration,
        )

        return response


@_with_batch_telemetry_wrapper
def batch_list_wrapper(
        tracer: Tracer,
        duration_histogram: Histogram,
        exception_counter: Counter,
        wrapped,
        instance,
        args,
        kwargs,
):
    """Wrapper for batch list operations."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    with tracer.start_span(
            BATCH_LIST_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE},
    ) as span:
        _handle_batch_list_request(span, kwargs, instance)

        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
                "operation": "batch.list",
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise e

        duration = end_time - start_time

        _handle_batch_list_response(
            response,
            span,
            instance,
            duration_histogram,
            duration,
        )

        return response


@_with_batch_telemetry_wrapper
def batch_cancel_wrapper(
        tracer: Tracer,
        duration_histogram: Histogram,
        exception_counter: Counter,
        status_counter: Counter,
        wrapped,
        instance,
        args,
        kwargs,
):
    """Wrapper for batch cancel operations."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    with tracer.start_span(
            BATCH_CANCEL_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE},
    ) as span:
        _handle_batch_cancel_request(span, kwargs, args, instance)

        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            attributes = {
                "error.type": e.__class__.__name__,
                "operation": "batch.cancel",
            }

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes=attributes)
            if exception_counter:
                exception_counter.add(1, attributes=attributes)

            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise e

        duration = end_time - start_time

        _handle_batch_cancel_response(
            response,
            span,
            instance,
            duration_histogram,
            status_counter,
            duration,
        )

        return response


@dont_throw
def _handle_batch_create_request(span, kwargs, instance):
    """Handle request attributes for batch create."""
    _set_client_attributes(span, instance)

    # Set batch create specific attributes
    if kwargs.get("input_file_id"):
        _set_span_attribute(span, "gen_ai.batch.input_file_id", kwargs.get("input_file_id"))

    if kwargs.get("endpoint"):
        _set_span_attribute(span, "gen_ai.batch.endpoint", kwargs.get("endpoint"))

    if kwargs.get("completion_window"):
        _set_span_attribute(span, "gen_ai.batch.completion_window", kwargs.get("completion_window"))

    if kwargs.get("metadata"):
        _set_span_attribute(span, "gen_ai.batch.metadata", json.dumps(kwargs.get("metadata")))


@dont_throw
def _handle_batch_retrieve_request(span, kwargs, args, instance):
    """Handle request attributes for batch retrieve."""
    _set_client_attributes(span, instance)

    # Extract batch ID from kwargs or args
    batch_id = kwargs.get("batch_id") or (args[0] if args else None)
    if batch_id:
        _set_span_attribute(span, "gen_ai.batch.id", batch_id)


@dont_throw
def _handle_batch_list_request(span, kwargs, instance):
    """Handle request attributes for batch list."""
    _set_client_attributes(span, instance)

    # Set list parameters
    if kwargs.get("limit"):
        _set_span_attribute(span, "gen_ai.batch.list.limit", kwargs.get("limit"))

    if kwargs.get("after"):
        _set_span_attribute(span, "gen_ai.batch.list.after", kwargs.get("after"))


@dont_throw
def _handle_batch_cancel_request(span, kwargs, args, instance):
    """Handle request attributes for batch cancel."""
    _set_client_attributes(span, instance)

    # Extract batch ID from kwargs or args
    batch_id = kwargs.get("batch_id") or (args[0] if args else None)
    if batch_id:
        _set_span_attribute(span, "gen_ai.batch.id", batch_id)


@dont_throw
def _handle_batch_create_response(
        response,
        span,
        instance=None,
        duration_histogram=None,
        requests_counter=None,
        duration=None,
):
    """Handle response attributes and metrics for batch create."""
    response_dict = model_as_dict(response) if hasattr(response, '__dict__') else response

    # Set batch metrics
    _set_batch_metrics(
        instance,
        duration_histogram,
        requests_counter,
        response_dict,
        duration,
        "batch.create",
    )

    # Set response attributes
    _set_batch_response_attributes(span, response_dict)

    span.set_status(Status(StatusCode.OK))


@dont_throw
def _handle_batch_retrieve_response(
        response,
        span,
        instance=None,
        duration_histogram=None,
        status_counter=None,
        duration=None,
):
    """Handle response attributes and metrics for batch retrieve."""
    response_dict = model_as_dict(response) if hasattr(response, '__dict__') else response

    # Set batch metrics
    _set_batch_metrics(
        instance,
        duration_histogram,
        status_counter,
        response_dict,
        duration,
        "batch.retrieve",
    )

    # Set response attributes
    _set_batch_response_attributes(span, response_dict)

    # Record status if available
    if status_counter and response_dict.get("status"):
        shared_attributes = _get_batch_shared_attributes(instance, "batch.retrieve")
        status_counter.add(1, attributes={**shared_attributes, "batch.status": response_dict["status"]})

    span.set_status(Status(StatusCode.OK))


@dont_throw
def _handle_batch_list_response(
        response,
        span,
        instance=None,
        duration_histogram=None,
        duration=None,
):
    """Handle response attributes and metrics for batch list."""
    response_dict = model_as_dict(response) if hasattr(response, '__dict__') else response

    # Set batch metrics
    _set_batch_metrics(
        instance,
        duration_histogram,
        None,
        response_dict,
        duration,
        "batch.list",
    )

    # Set response attributes
    _set_batch_list_response_attributes(span, response_dict)

    span.set_status(Status(StatusCode.OK))


@dont_throw
def _handle_batch_cancel_response(
        response,
        span,
        instance=None,
        duration_histogram=None,
        status_counter=None,
        duration=None,
):
    """Handle response attributes and metrics for batch cancel."""
    response_dict = model_as_dict(response) if hasattr(response, '__dict__') else response

    # Set batch metrics
    _set_batch_metrics(
        instance,
        duration_histogram,
        status_counter,
        response_dict,
        duration,
        "batch.cancel",
    )

    # Set response attributes
    _set_batch_response_attributes(span, response_dict)

    # Record status change if available
    if status_counter and response_dict.get("status"):
        shared_attributes = _get_batch_shared_attributes(instance, "batch.cancel")
        status_counter.add(1, attributes={**shared_attributes, "batch.status": response_dict["status"]})

    span.set_status(Status(StatusCode.OK))


def _set_batch_metrics(
        instance, duration_histogram, counter, response_dict, duration, operation
):
    """Set batch-specific metrics."""
    shared_attributes = _get_batch_shared_attributes(instance, operation)

    # Duration metrics
    if duration and isinstance(duration, (float, int)) and duration_histogram:
        duration_histogram.record(duration, attributes=shared_attributes)

    # Counter metrics (requests for create, status for retrieve/cancel)
    if counter and response_dict:
        if operation == "batch.create" and response_dict.get("request_counts", {}).get("total"):
            counter.add(
                response_dict["request_counts"]["total"],
                attributes={**shared_attributes, "batch.operation": "create"}
            )
        elif operation in ["batch.retrieve", "batch.cancel"] and response_dict.get("status"):
            counter.add(
                1,
                attributes={**shared_attributes, "batch.status": response_dict["status"]}
            )


def _get_batch_shared_attributes(instance, operation):
    """Get shared attributes for batch operations."""
    return {
        "gen_ai.system": "openai",
        "gen_ai.operation.name": operation,
        "server.address": _get_openai_base_url(instance),
    }


@dont_throw
def _set_batch_response_attributes(span, response_dict):
    """Set common batch response attributes."""
    if not response_dict:
        return

    # Common batch attributes
    if response_dict.get("id"):
        _set_span_attribute(span, "gen_ai.batch.id", response_dict["id"])

    if response_dict.get("object"):
        _set_span_attribute(span, "gen_ai.batch.object", response_dict["object"])

    if response_dict.get("endpoint"):
        _set_span_attribute(span, "gen_ai.batch.endpoint", response_dict["endpoint"])

    if response_dict.get("status"):
        _set_span_attribute(span, "gen_ai.batch.status", response_dict["status"])

    if response_dict.get("created_at"):
        _set_span_attribute(span, "gen_ai.batch.created_at", response_dict["created_at"])

    if response_dict.get("completed_at"):
        _set_span_attribute(span, "gen_ai.batch.completed_at", response_dict["completed_at"])

    if response_dict.get("failed_at"):
        _set_span_attribute(span, "gen_ai.batch.failed_at", response_dict["failed_at"])

    if response_dict.get("expired_at"):
        _set_span_attribute(span, "gen_ai.batch.expired_at", response_dict["expired_at"])

    if response_dict.get("finalizing_at"):
        _set_span_attribute(span, "gen_ai.batch.finalizing_at", response_dict["finalizing_at"])

    if response_dict.get("in_progress_at"):
        _set_span_attribute(span, "gen_ai.batch.in_progress_at", response_dict["in_progress_at"])

    if response_dict.get("cancelled_at"):
        _set_span_attribute(span, "gen_ai.batch.cancelled_at", response_dict["cancelled_at"])

    # Request counts
    if response_dict.get("request_counts"):
        counts = response_dict["request_counts"]
        if counts.get("total"):
            _set_span_attribute(span, "gen_ai.batch.requests.total", counts["total"])
        if counts.get("completed"):
            _set_span_attribute(span, "gen_ai.batch.requests.completed", counts["completed"])
        if counts.get("failed"):
            _set_span_attribute(span, "gen_ai.batch.requests.failed", counts["failed"])

    # Files
    if response_dict.get("input_file_id"):
        _set_span_attribute(span, "gen_ai.batch.input_file_id", response_dict["input_file_id"])

    if response_dict.get("output_file_id"):
        _set_span_attribute(span, "gen_ai.batch.output_file_id", response_dict["output_file_id"])

    if response_dict.get("error_file_id"):
        _set_span_attribute(span, "gen_ai.batch.error_file_id", response_dict["error_file_id"])

    # Metadata
    if response_dict.get("metadata"):
        _set_span_attribute(span, "gen_ai.batch.metadata", json.dumps(response_dict["metadata"]))

    # Completion window
    if response_dict.get("completion_window"):
        _set_span_attribute(span, "gen_ai.batch.completion_window", response_dict["completion_window"])


@dont_throw
def _set_batch_list_response_attributes(span, response_dict):
    """Set batch list specific response attributes."""
    if not response_dict:
        return

    # List specific attributes
    if response_dict.get("object"):
        _set_span_attribute(span, "gen_ai.batch.list.object", response_dict["object"])

    if response_dict.get("data") and isinstance(response_dict["data"], list):
        _set_span_attribute(span, "gen_ai.batch.list.count", len(response_dict["data"]))

        # Set attributes for each batch in the list
        for i, batch in enumerate(response_dict["data"]):
            prefix = f"gen_ai.batch.list.{i}"
            if batch.get("id"):
                _set_span_attribute(span, f"{prefix}.id", batch["id"])
            if batch.get("status"):
                _set_span_attribute(span, f"{prefix}.status", batch["status"])
            if batch.get("endpoint"):
                _set_span_attribute(span, f"{prefix}.endpoint", batch["endpoint"])

    if response_dict.get("first_id"):
        _set_span_attribute(span, "gen_ai.batch.list.first_id", response_dict["first_id"])

    if response_dict.get("last_id"):
        _set_span_attribute(span, "gen_ai.batch.list.last_id", response_dict["last_id"])

    if response_dict.get("has_more"):
        _set_span_attribute(span, "gen_ai.batch.list.has_more", response_dict["has_more"])