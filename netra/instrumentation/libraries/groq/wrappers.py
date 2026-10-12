import logging
import time
from typing import Any, Callable, Dict, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.groq.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.llm_common import AsyncLLMStream, BaseLLMStream, LLMStream
from netra.instrumentation.span_utils import record_span_timing

logger = logging.getLogger(__name__)

CHAT_SPAN_NAME = "groq.chat"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"


class _GroqStreamMixin(BaseLLMStream):
    """Chunk handling shared by the sync and async Groq stream wrappers (OpenAI-style chunks)."""

    def _process_chunk(self, chunk: Any) -> None:
        self._accumulate_chat_chunk(model_as_dict(chunk))

    def _apply_response_attributes(self) -> None:
        set_response_attributes(self._self_span, self._self_complete_response)


class StreamingWrapper(_GroqStreamMixin, LLMStream):
    """Wrapper for streaming responses (OpenAI-style)."""


class AsyncStreamingWrapper(_GroqStreamMixin, AsyncLLMStream):
    """Async wrapper for streaming responses (OpenAI-style)."""


# Only exceptions raised by the wrapped Groq call may reach the caller. Failures while
# recording telemetry are logged at DEBUG and swallowed.
def _safe_set_request_attributes(span: Span, kwargs: Dict[str, Any]) -> None:
    try:
        set_request_attributes(span, kwargs, "chat")
    except Exception:
        logger.debug("Failed to set Groq request attributes", exc_info=True)


def _record_response(span: Span, response: Any) -> None:
    end_time = time.time()
    try:
        set_response_attributes(span, model_as_dict(response))
        record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
        record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time, record_event_timestamp=True)
        record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
    except Exception:
        logger.debug("Failed to set Groq response attributes", exc_info=True)


def _start_stream_span(tracer: Tracer, kwargs: Dict[str, Any]) -> Span:
    span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
    _safe_set_request_attributes(span, kwargs)
    return span


def _fail_stream_span(span: Span, error: Exception) -> None:
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)
    span.end()


def _wrap_stream(stream_cls: Any, span: Span, response: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return stream_cls(span=span, response=response, request_kwargs=kwargs)
    except Exception:
        logger.debug("Failed to wrap Groq stream; returning it uninstrumented", exc_info=True)
        span.end()
        return response


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Groq chat completions (OpenAI-style)."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        if kwargs.get("stream"):
            span = _start_stream_span(tracer, kwargs)
            context = context_api.attach(set_span_in_context(span))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                _fail_stream_span(span, e)
                raise
            finally:
                context_api.detach(context)
            return _wrap_stream(StreamingWrapper, span, response, kwargs)

        with tracer.start_as_current_span(
            CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
        ) as span:
            _safe_set_request_attributes(span, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            _record_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for Groq chat completions (OpenAI-style)."""

    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        if kwargs.get("stream"):
            span = _start_stream_span(tracer, kwargs)
            context = context_api.attach(set_span_in_context(span))
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                _fail_stream_span(span, e)
                raise
            finally:
                context_api.detach(context)
            return _wrap_stream(AsyncStreamingWrapper, span, response, kwargs)

        with tracer.start_as_current_span(
            CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
        ) as span:
            _safe_set_request_attributes(span, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            _record_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper
