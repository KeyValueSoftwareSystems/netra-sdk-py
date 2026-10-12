import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.libraries.google_genai.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.llm_common import AsyncLLMStream, BaseLLMStream, LLMStream
from netra.instrumentation.span_utils import record_span_timing

logger = logging.getLogger(__name__)


CONTENT_SPAN_NAME = "genai.generate_content"
CONTENT_STREAM_SPAN_NAME = "genai.generate_content_stream"
IMAGES_SPAN_NAME = "genai.generate_images"
VIDEOS_SPAN_NAME = "genai.generate_videos"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"
GEN_AI_RESPONSE_DURATION = "gen_ai.response.duration"


# Only exceptions raised by the wrapped GenAI call may reach the caller. Failures while
# recording telemetry are logged at DEBUG and swallowed.
def _safe_set_request_attributes(span: Span, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    try:
        set_request_attributes(span, args, kwargs)
    except Exception:
        logger.debug("Failed to set GenAI request attributes", exc_info=True)


def _record_content_response(span: Span, response: Any) -> None:
    end_time = time.time()
    try:
        set_response_attributes(span, response)
        record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
        record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time, record_event_timestamp=True)
        record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
    except Exception:
        logger.debug("Failed to set GenAI response attributes", exc_info=True)


def _record_media_response(span: Span, response: Any) -> None:
    try:
        record_span_timing(span, GEN_AI_RESPONSE_DURATION)
        set_response_attributes(span, response)
    except Exception:
        logger.debug("Failed to set GenAI response attributes", exc_info=True)


def _fail_stream_span(span: Span, error: Exception) -> None:
    logger.error("netra.instrumentation.libraries.google_genai: %s", error)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)
    span.end()


def _wrap_stream(stream_cls: Type[BaseLLMStream], span: Span, response: Any) -> Any:
    try:
        return stream_cls(span=span, response=response)
    except Exception:
        logger.debug("Failed to wrap GenAI stream; returning it uninstrumented", exc_info=True)
        span.end()
        return response


def content_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            CONTENT_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Gemini", "llm.request.type": "completion"},
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_content_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def acontent_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            CONTENT_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Gemini", "llm.request.type": "completion"},
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_content_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def content_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span = tracer.start_span(
            CONTENT_STREAM_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Gemini", "llm.request.type": "completion"},
        )
        _safe_set_request_attributes(span, args, kwargs)
        context = context_api.attach(set_span_in_context(span))
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            _fail_stream_span(span, e)
            raise
        finally:
            context_api.detach(context)
        return _wrap_stream(StreamingWrapper, span, response)

    return wrapper


def acontent_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        span = tracer.start_span(
            CONTENT_STREAM_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Gemini", "llm.request.type": "completion"},
        )
        _safe_set_request_attributes(span, args, kwargs)
        context = context_api.attach(set_span_in_context(span))
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            _fail_stream_span(span, e)
            raise
        finally:
            context_api.detach(context)
        return _wrap_stream(AsyncStreamingWrapper, span, response)

    return wrapper


def images_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            IMAGES_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "image"}
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_media_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def aimages_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            IMAGES_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "image"}
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_media_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def videos_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            VIDEOS_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "video"}
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_media_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


def avideos_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            VIDEOS_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "video"}
        ) as span:
            _safe_set_request_attributes(span, args, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.libraries.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            _record_media_response(span, response)
            span.set_status(Status(StatusCode.OK))
            return response

    return wrapper


class _GenAIStreamMixin(BaseLLMStream):
    """Chunk handling shared by the sync and async Gemini stream wrappers.

    Keeps the latest chunk (it carries the final usage metadata) and the concatenated text.
    """

    def __init__(self, span: Span, response: Any, request_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(span=span, response=response, request_kwargs=request_kwargs)
        self._self_buffer: Dict[str, Any] = {"chunk": None, "content": ""}

    def _process_chunk(self, chunk: Any) -> None:
        self._self_buffer["chunk"] = chunk
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            if text:
                self._record_first_token()
            self._self_buffer["content"] += text

    def _apply_response_attributes(self) -> None:
        set_response_attributes(self._self_span, self._self_buffer)

    def _extract_output_text(self) -> str:
        content: str = self._self_buffer.get("content", "")
        return content


class StreamingWrapper(_GenAIStreamMixin, LLMStream):
    """Wrapper for ``generate_content_stream`` responses."""


class AsyncStreamingWrapper(_GenAIStreamMixin, AsyncLLMStream):
    """Wrapper for async ``generate_content_stream`` responses."""
