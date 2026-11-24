import time
from typing import Any, Callable, Dict, Tuple

from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.typesafe.utils import (
    ERROR_TYPE_ATTR,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_TYPESAFE,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.span_utils import record_span_timing

SYSTEM_ONE_SPAN_NAME = "typesafe.system_one"
LLM_RESPONSE_DURATION = "llm.response.duration"

# "decision", as for the OpenAI Decisions API: system_one answers questions about an input
# rather than generating text.
_SPAN_ATTRIBUTES = {GEN_AI_SYSTEM: GEN_AI_SYSTEM_TYPESAFE, "llm.request.type": "decision"}


def _record_success(span: Span, response: Any) -> None:
    record_span_timing(span, LLM_RESPONSE_DURATION, time.time())
    set_response_attributes(span, response)
    span.set_status(Status(StatusCode.OK))


def _record_failure(span: Span, error: BaseException) -> None:
    span.set_attribute(ERROR_TYPE_ATTR, type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def system_one_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrap ``TypeSafeClient.system_one`` in a CLIENT span covering the call and its retries."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            SYSTEM_ONE_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes=_SPAN_ATTRIBUTES,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            set_request_attributes(span, instance, args, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                _record_failure(span, e)
                raise
            _record_success(span, response)
            return response

    return wrapper


def async_system_one_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrap ``AsyncTypeSafeClient.system_one`` in a CLIENT span covering the call and its retries."""

    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            SYSTEM_ONE_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes=_SPAN_ATTRIBUTES,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            set_request_attributes(span, instance, args, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                _record_failure(span, e)
                raise
            _record_success(span, response)
            return response

    return wrapper
