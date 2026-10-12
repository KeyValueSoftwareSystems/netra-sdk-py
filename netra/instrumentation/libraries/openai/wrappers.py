import logging
import time
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Optional, Tuple, Type

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.openai.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.llm_common import AsyncLLMStream, BaseLLMStream, LLMStream
from netra.instrumentation.span_utils import record_span_timing

logger = logging.getLogger(__name__)

# Span names
CHAT_SPAN_NAME = "openai.chat"
EMBEDDING_SPAN_NAME = "openai.embedding"
RESPONSE_SPAN_NAME = "openai.response"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"


# Only exceptions raised by the wrapped OpenAI call may reach the caller. Failures while
# recording telemetry are logged at DEBUG and swallowed.
def _safe_set_request_attributes(span: Span, kwargs: Dict[str, Any], operation_type: str) -> None:
    try:
        set_request_attributes(span, kwargs, operation_type)
    except Exception:
        logger.debug("Failed to set OpenAI request attributes", exc_info=True)


def _record_response(span: Span, response: Any) -> None:
    end_time = time.time()
    try:
        set_response_attributes(span, model_as_dict(response))
        record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
        record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time, record_event_timestamp=True)
        record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
    except Exception:
        logger.debug("Failed to set OpenAI response attributes", exc_info=True)


def _start_stream_span(tracer: Tracer, span_name: str, operation_type: str, kwargs: Dict[str, Any]) -> Span:
    span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": operation_type})
    _safe_set_request_attributes(span, kwargs, operation_type)
    return span


def _fail_stream_span(span: Span, error: Exception) -> None:
    logger.error("netra.instrumentation.libraries.openai: %s", error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)
    span.end()


def _wrap_stream(stream_cls: Type[BaseLLMStream], span: Span, response: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return stream_cls(span=span, response=response, request_kwargs=kwargs)
    except Exception:
        logger.debug("Failed to wrap OpenAI stream; returning it uninstrumented", exc_info=True)
        span.end()
        return response


def _traced_call(
    tracer: Tracer,
    span_name: str,
    operation_type: str,
    stream_cls: Optional[Type[BaseLLMStream]],
    wrapped: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if stream_cls is not None and kwargs.get("stream", False):
        span = _start_stream_span(tracer, span_name, operation_type, kwargs)
        context = context_api.attach(set_span_in_context(span))
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            _fail_stream_span(span, e)
            raise
        finally:
            context_api.detach(context)
        return _wrap_stream(stream_cls, span, response, kwargs)

    with tracer.start_as_current_span(
        span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": operation_type}
    ) as span:
        _safe_set_request_attributes(span, kwargs, operation_type)
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.openai: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        _record_response(span, response)
        span.set_status(Status(StatusCode.OK))
        return response


async def _atraced_call(
    tracer: Tracer,
    span_name: str,
    operation_type: str,
    stream_cls: Optional[Type[BaseLLMStream]],
    wrapped: Callable[..., Awaitable[Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if stream_cls is not None and kwargs.get("stream", False):
        span = _start_stream_span(tracer, span_name, operation_type, kwargs)
        context = context_api.attach(set_span_in_context(span))
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            _fail_stream_span(span, e)
            raise
        finally:
            context_api.detach(context)
        return _wrap_stream(stream_cls, span, response, kwargs)

    with tracer.start_as_current_span(
        span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": operation_type}
    ) as span:
        _safe_set_request_attributes(span, kwargs, operation_type)
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.openai: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        _record_response(span, response)
        span.set_status(Status(StatusCode.OK))
        return response


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for chat completions"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        return _traced_call(tracer, CHAT_SPAN_NAME, "chat", StreamingWrapper, wrapped, args, kwargs)

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for chat completions"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)
        return await _atraced_call(tracer, CHAT_SPAN_NAME, "chat", AsyncStreamingWrapper, wrapped, args, kwargs)

    return wrapper


def embeddings_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for embeddings"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        return _traced_call(tracer, EMBEDDING_SPAN_NAME, "embedding", None, wrapped, args, kwargs)

    return wrapper


def aembeddings_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for embeddings"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)
        return await _atraced_call(tracer, EMBEDDING_SPAN_NAME, "embedding", None, wrapped, args, kwargs)

    return wrapper


def responses_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for responses.create (new OpenAI API)"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        return _traced_call(tracer, RESPONSE_SPAN_NAME, "response", StreamingWrapper, wrapped, args, kwargs)

    return wrapper


def aresponses_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for responses.create (new OpenAI API)"""

    async def wrapper(wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Any, kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)
        return await _atraced_call(tracer, RESPONSE_SPAN_NAME, "response", AsyncStreamingWrapper, wrapped, args, kwargs)

    return wrapper


class _OpenAIStreamMixin(BaseLLMStream):
    """Chunk handling shared by the sync and async OpenAI stream wrappers.

    Handles both Chat Completions chunks and Responses API stream events.
    """

    def _process_chunk(self, chunk: Any) -> None:
        chunk_dict = model_as_dict(chunk)
        if not isinstance(chunk_dict, dict):
            return
        self._accumulate_chat_chunk(chunk_dict)
        self._accumulate_responses_event(chunk_dict)

    def _apply_response_attributes(self) -> None:
        set_response_attributes(self._self_span, self._self_complete_response)


class StreamingWrapper(_OpenAIStreamMixin, LLMStream):
    """Wrapper for streaming responses"""


class AsyncStreamingWrapper(_OpenAIStreamMixin, AsyncLLMStream):
    """Async wrapper for streaming responses"""
