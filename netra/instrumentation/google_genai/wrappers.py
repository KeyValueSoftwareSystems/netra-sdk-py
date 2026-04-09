import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.google_genai.utils import (
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.utils import record_span_timing

logger = logging.getLogger(__name__)


CONTENT_SPAN_NAME = "genai.generate_content"
CONTENT_STREAM_SPAN_NAME = "genai.generate_content_stream"
IMAGES_SPAN_NAME = "genai.generate_images"
VIDEOS_SPAN_NAME = "genai.generate_videos"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"
GEN_AI_RESPONSE_DURATION = "gen_ai.response.duration"


def content_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            CONTENT_SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={"gen_ai.system": "Gemini", "llm.request.type": "completion"},
        ) as span:
            try:
                set_request_attributes(span, args, kwargs)
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

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
            try:
                set_request_attributes(span, args, kwargs)
                response = await wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

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
        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, args, kwargs)
            response = wrapped(*args, **kwargs)
            return StreamingWrapper(span=span, response=response)
        except Exception as e:
            logger.error("netra.instrumentation.google_genai: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(context)

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
        try:
            context = context_api.attach(set_span_in_context(span))
            set_request_attributes(span, args, kwargs)
            response = await wrapped(*args, **kwargs)
            return AsyncStreamingWrapper(span=span, response=response)
        except Exception as e:
            logger.error("netra.instrumentation.google_genai: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise
        finally:
            context_api.detach(context)

    return wrapper


def images_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            IMAGES_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "image"}
        ) as span:
            try:
                set_request_attributes(span, args, kwargs)
                response = wrapped(*args, **kwargs)
                record_span_timing(span, GEN_AI_RESPONSE_DURATION)
                set_response_attributes(span, response)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def aimages_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            IMAGES_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "image"}
        ) as span:
            try:
                set_request_attributes(span, args, kwargs)
                response = await wrapped(*args, **kwargs)
                record_span_timing(span, GEN_AI_RESPONSE_DURATION)
                set_response_attributes(span, response)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def videos_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            VIDEOS_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "video"}
        ) as span:
            try:
                set_request_attributes(span, args, kwargs)
                response = wrapped(*args, **kwargs)
                record_span_timing(span, GEN_AI_RESPONSE_DURATION)
                set_response_attributes(span, response)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def avideos_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            VIDEOS_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"gen_ai.system": "Gemini", "gen_ai.type": "video"}
        ) as span:
            try:
                set_request_attributes(span, args, kwargs)
                response = await wrapped(*args, **kwargs)
                record_span_timing(span, GEN_AI_RESPONSE_DURATION)
                set_response_attributes(span, response)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.google_genai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


class StreamingWrapper:
    def __init__(self, span: Span, response: Iterator[Any]) -> None:
        self._span = span
        self._buffer: dict[Any, Any] = {"chunk": None, "content": ""}
        self._chunk: Any = None
        self._response = response
        self._first_content_recorded: bool = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._response)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize_span()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _process_chunk(self, chunk: Any) -> None:
        text = getattr(chunk, "text", None)
        self._buffer["chunk"] = chunk
        if isinstance(text, str):
            if text and not self._first_content_recorded:
                self._first_content_recorded = True
                first_token_time = time.time()
                record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                record_span_timing(self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)
            self._buffer["content"] += text
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._buffer)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class AsyncStreamingWrapper:
    def __init__(self, span: Span, response: AsyncIterator[Any]) -> None:
        self._span = span
        self._buffer: dict[Any, Any] = {"chunk": None, "content": ""}
        self._response = response
        self._first_content_recorded: bool = False

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._response.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize_span()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _process_chunk(self, chunk: Any) -> None:
        text = getattr(chunk, "text", None)
        self._buffer["chunk"] = chunk
        if isinstance(text, str):
            if text and not self._first_content_recorded:
                self._first_content_recorded = True
                first_token_time = time.time()
                record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                record_span_timing(self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)
            self._buffer["content"] += text
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._buffer)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
