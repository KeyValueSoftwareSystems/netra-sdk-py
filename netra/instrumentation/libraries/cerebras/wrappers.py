import logging
import time
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Tuple, Type

from opentelemetry import context as context_api
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.cerebras.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.llm_common import AsyncLLMStream, BaseLLMStream, LLMStream
from netra.instrumentation.span_utils import record_span_timing

logger = logging.getLogger(__name__)

CHAT_SPAN_NAME = "cerebras.chat.completions"
COMPLETION_SPAN_NAME = "cerebras.completions"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"


def _detect_streaming(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    is_streaming = bool(kwargs.get("stream", False))
    try:
        if not is_streaming and args:
            last_pos = args[-1]
            if isinstance(last_pos, bool):
                is_streaming = is_streaming or last_pos is True
    except Exception:
        pass
    return is_streaming


class _CerebrasStreamMixin(BaseLLMStream):
    """Chunk handling shared by the sync and async Cerebras stream wrappers (chat and completions)."""

    def _process_chunk(self, chunk: Any) -> None:
        self._accumulate_chat_chunk(model_as_dict(chunk))

    def _apply_response_attributes(self) -> None:
        set_response_attributes(self._self_span, self._self_complete_response)


class StreamingWrapper(_CerebrasStreamMixin, LLMStream):
    """Wrapper for streaming responses"""


class AsyncStreamingWrapper(_CerebrasStreamMixin, AsyncLLMStream):
    """Async wrapper for streaming responses"""


# Only exceptions raised by the wrapped Cerebras call may reach the caller. Failures while
# recording telemetry are logged at DEBUG and swallowed.
def _safe_set_request_attributes(span: Span, request_type: LLMRequestTypeValues, kwargs: Dict[str, Any]) -> None:
    try:
        set_request_attributes(span, request_type, kwargs)
    except Exception:
        logger.debug("Failed to set Cerebras request attributes", exc_info=True)


def _record_response(span: Span, response: Any) -> None:
    end_time = time.time()
    try:
        set_response_attributes(span, response)
        record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
        record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time, record_event_timestamp=True)
        record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
    except Exception:
        logger.debug("Failed to set Cerebras response attributes", exc_info=True)


def _wrap_stream(stream_cls: Type[BaseLLMStream], span: Span, response: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return stream_cls(span=span, response=response, request_kwargs=kwargs)
    except Exception:
        logger.debug("Failed to wrap Cerebras stream; returning it uninstrumented", exc_info=True)
        span.end()
        return response


def _traced_call(
    tracer: Tracer,
    span_name: str,
    request_type: LLMRequestTypeValues,
    attributes: Dict[str, Any],
    wrapped: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if _detect_streaming(args, kwargs):
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attributes)
        _safe_set_request_attributes(span, request_type, kwargs)
        ctx = context_api.attach(set_span_in_context(span))
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.cerebras: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(ctx)
        return _wrap_stream(StreamingWrapper, span, response, kwargs)

    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
        _safe_set_request_attributes(span, request_type, kwargs)
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.cerebras: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        _record_response(span, response)
        span.set_status(Status(StatusCode.OK))
        return response


async def _atraced_call(
    tracer: Tracer,
    span_name: str,
    request_type: LLMRequestTypeValues,
    attributes: Dict[str, Any],
    wrapped: Callable[..., Awaitable[Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    if _detect_streaming(args, kwargs):
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes=attributes)
        _safe_set_request_attributes(span, request_type, kwargs)
        ctx = context_api.attach(set_span_in_context(span))
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.cerebras: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(ctx)
        return _wrap_stream(AsyncStreamingWrapper, span, response, kwargs)

    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attributes) as span:
        _safe_set_request_attributes(span, request_type, kwargs)
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            logger.error("netra.instrumentation.libraries.cerebras: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        _record_response(span, response)
        span.set_status(Status(StatusCode.OK))
        return response


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        return _traced_call(
            tracer,
            CHAT_SPAN_NAME,
            LLMRequestTypeValues.CHAT,
            {"gen_ai.system": "Cerebras", "llm.request.type": "chat"},
            wrapped,
            args,
            kwargs,
        )

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)
        return await _atraced_call(
            tracer,
            CHAT_SPAN_NAME,
            LLMRequestTypeValues.CHAT,
            {"gen_ai.system": "Cerebras", "llm.request.type": "chat"},
            wrapped,
            args,
            kwargs,
        )

    return wrapper


def completions_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)
        return _traced_call(
            tracer,
            COMPLETION_SPAN_NAME,
            LLMRequestTypeValues.COMPLETION,
            {"gen_ai.system": "Cerebras", "llm.request.type": "completion"},
            wrapped,
            args,
            kwargs,
        )

    return wrapper


def acompletions_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)
        return await _atraced_call(
            tracer,
            COMPLETION_SPAN_NAME,
            LLMRequestTypeValues.COMPLETION,
            {"gen_ai.system": "Cerebras", "llm.request.type": "completion"},
            wrapped,
            args,
            kwargs,
        )

    return wrapper
