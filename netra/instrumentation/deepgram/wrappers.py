import logging
import time
from typing import Any, Callable, Dict, Tuple

from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.deepgram.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

TRANSCRIBE_URL_SPAN_NAME = "deepgram.transcribe_url"
TRANSCRIBE_FILE_SPAN_NAME = "deepgram.transcribe_file"


def _wrap_transcribe(
    tracer: Tracer,
    span_name: str,
    source_type: str,
) -> Callable[..., Any]:
    """
    Wrap the transcribe_url method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
        span_name: The name of the span to create.
        source_type: The type of the source (e.g. "url" or "file").
    """

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
            try:
                set_request_attributes(span, kwargs, source_type)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                span.set_attribute("deepgram.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.deepgram: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def transcribe_url_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the transcribe_url method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_transcribe(tracer, TRANSCRIBE_URL_SPAN_NAME, "url")


def transcribe_file_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Wrap the transcribe_file method with OpenTelemetry instrumentation.

    Args:
        tracer: The OpenTelemetry tracer to use for instrumentation.
    """
    return _wrap_transcribe(tracer, TRANSCRIBE_FILE_SPAN_NAME, "file")
