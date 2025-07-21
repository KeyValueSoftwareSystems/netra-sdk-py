"""
OpenAI API wrappers for Netra SDK instrumentation.

This module contains wrapper functions for different OpenAI API endpoints with
proper span handling for streaming vs non-streaming operations.
"""

import logging
import time
from collections.abc import Awaitable
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import (
    SpanAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)

# Span names
CHAT_SPAN_NAME = "openai.chat"
COMPLETION_SPAN_NAME = "openai.completion"
EMBEDDING_SPAN_NAME = "openai.embedding"
RESPONSE_SPAN_NAME = "openai.response"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def is_streaming_response(response: Any) -> bool:
    """Check if response is a streaming response"""
    return hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict))


def model_as_dict(obj: Any) -> Dict[str, Any]:
    """Convert OpenAI model object to dictionary"""
    if hasattr(obj, "model_dump"):
        result = obj.model_dump()
        return result if isinstance(result, dict) else {}
    elif hasattr(obj, "to_dict"):
        result = obj.to_dict()
        return result if isinstance(result, dict) else {}
    elif isinstance(obj, dict):
        return obj
    else:
        return {}


def set_request_attributes(span: Span, kwargs: Dict[str, Any], operation_type: str) -> None:
    """Set request attributes on span"""
    if not span.is_recording():
        return

    # Set operation type
    span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TYPE}", operation_type)

    # Common attributes
    if kwargs.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MODEL}", kwargs["model"])

    if kwargs.get("temperature") is not None:
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", kwargs["temperature"])

    if kwargs.get("max_tokens") is not None:
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs["max_tokens"])

    if kwargs.get("stream") is not None:
        span.set_attribute("gen_ai.stream", kwargs["stream"])

    # Chat-specific attributes
    if operation_type == "chat" and kwargs.get("messages"):
        messages = kwargs["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            for index, message in enumerate(messages):
                if hasattr(message, "content"):
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "user")
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", message.content)
                elif isinstance(message, dict):
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", message.get("role", "user"))
                    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(message.get("content", "")))

    # Response-specific attributes
    if operation_type == "response":
        if kwargs.get("instructions"):
            span.set_attribute("gen_ai.instructions", kwargs["instructions"])
        if kwargs.get("input"):
            span.set_attribute("gen_ai.input", kwargs["input"])


def set_response_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        return

    if response_dict.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_RESPONSE_MODEL}", response_dict["model"])

    if response_dict.get("id"):
        span.set_attribute("gen_ai.response.id", response_dict["id"])

    # Usage information
    usage = response_dict.get("usage", {})
    if usage:
        if usage.get("prompt_tokens"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", usage["prompt_tokens"])
        if usage.get("completion_tokens"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", usage["completion_tokens"])
        if usage.get("cache_read_input_token"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", usage["cache_read_input_token"])
        if usage.get("total_tokens"):
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", usage["total_tokens"])

    # Response content
    choices = response_dict.get("choices", [])
    for index, choice in enumerate(choices):
        if choice.get("message", {}).get("role"):
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", choice["message"]["role"])
        if choice.get("message", {}).get("content"):
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", choice["message"]["content"])
        if choice.get("finish_reason"):
            span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason", choice["finish_reason"])

    # For responses.create
    if response_dict.get("output_text"):
        span.set_attribute("gen_ai.response.output_text", response_dict["output_text"])


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for chat completions"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Check if streaming
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})

            set_request_attributes(span, kwargs, "chat")

            try:
                start_time = time.time()
                response = wrapped(*args, **kwargs)

                return StreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:
            # Use start_as_current_span for non-streaming - returns context manager
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                set_request_attributes(span, kwargs, "chat")

                try:
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for chat completions"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Check if streaming
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})

            set_request_attributes(span, kwargs, "chat")

            try:
                start_time = time.time()
                response = await wrapped(*args, **kwargs)

                return AsyncStreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:
            # Use start_as_current_span for non-streaming - returns context manager
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                set_request_attributes(span, kwargs, "chat")

                try:
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def completion_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for text completions"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            )

            set_request_attributes(span, kwargs, "completion")

            try:
                start_time = time.time()
                response = wrapped(*args, **kwargs)

                return StreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:
            # Use start_as_current_span for non-streaming - returns context manager
            with tracer.start_as_current_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            ) as span:
                set_request_attributes(span, kwargs, "completion")

                try:
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def acompletion_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for text completions"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            )

            set_request_attributes(span, kwargs, "completion")

            try:
                start_time = time.time()
                response = await wrapped(*args, **kwargs)

                return AsyncStreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
        else:
            # Use start_as_current_span for non-streaming - returns context manager
            with tracer.start_as_current_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            ) as span:
                set_request_attributes(span, kwargs, "completion")

                try:
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def embeddings_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for embeddings"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Embeddings are never streaming, always use start_as_current_span
        with tracer.start_as_current_span(
            EMBEDDING_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "embedding"}
        ) as span:
            set_request_attributes(span, kwargs, "embedding")

            try:
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def aembeddings_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for embeddings"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Embeddings are never streaming, always use start_as_current_span
        with tracer.start_as_current_span(
            EMBEDDING_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "embedding"}
        ) as span:
            set_request_attributes(span, kwargs, "embedding")

            try:
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def responses_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for responses.create (new OpenAI API)"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # responses.create is typically not streaming, use start_as_current_span
        with tracer.start_as_current_span(
            RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
        ) as span:
            set_request_attributes(span, kwargs, "response")

            try:
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def aresponses_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for responses.create (new OpenAI API)"""

    async def wrapper(wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Any, kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # responses.create is typically not streaming, use start_as_current_span
        with tracer.start_as_current_span(
            RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
        ) as span:
            set_request_attributes(span, kwargs, "response")

            try:
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


class StreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Wrapper for streaming responses"""

    def __init__(self, span: Span, response: Iterator[Any], start_time: float, request_kwargs: Dict[str, Any]) -> None:
        super().__init__(response)
        self._span = span
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize_span()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process streaming chunk"""
        chunk_dict = model_as_dict(chunk)

        # Accumulate response data
        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        # Add chunk event
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time

        set_response_attributes(self._span, self._complete_response)
        self._span.set_attribute("llm.response.duration", duration)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class AsyncStreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Async wrapper for streaming responses"""

    def __init__(
        self, span: Span, response: AsyncIterator[Any], start_time: float, request_kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(response)
        self._span = span
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize_span()
            raise

    def _process_chunk(self, chunk: Any) -> None:
        """Process streaming chunk"""
        chunk_dict = model_as_dict(chunk)

        # Accumulate response data
        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        # Add chunk event
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time

        set_response_attributes(self._span, self._complete_response)
        self._span.set_attribute("llm.response.duration", duration)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
