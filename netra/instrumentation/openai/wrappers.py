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


def extract_output_text(response_dict: Dict[str, Any]) -> str:
    """Best-effort extraction of output_text from OpenAI Responses object shape."""
    # Direct property if present
    direct = response_dict.get("output_text")
    if isinstance(direct, str) and direct:
        return direct

    # Attempt to reconstruct from structured 'output' list
    out = response_dict.get("output")
    parts: list[str] = []
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    # c may have shape {"type": "output_text", "text": {"value": "..."}} or {"text": "..."}
                    text = c.get("text")
                    value = None
                    if isinstance(text, dict):
                        value = text.get("value")
                    elif isinstance(text, str):
                        value = text
                    if isinstance(value, str) and value:
                        parts.append(value)
            # Some shapes may place text directly on the item
            text = item.get("text")
            if isinstance(text, dict) and isinstance(text.get("value"), str):
                parts.append(text["value"])
            elif isinstance(text, str):
                parts.append(text)

    return "".join(parts)


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
        idx = 0
        if kwargs.get("instructions"):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", "system")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", kwargs["instructions"])
            idx += 1
        if kwargs.get("input"):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", "user")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", kwargs["input"])


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
    if isinstance(usage, dict) and usage:
        # Support both classic and Responses API naming
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        cache_read_input = usage.get("cache_read_input_token", usage.get("cache_read_input_tokens"))
        total_tokens = usage.get("total_tokens")

        if prompt_tokens is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", prompt_tokens)
        if completion_tokens is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", completion_tokens)
        if cache_read_input is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", cache_read_input)
        if total_tokens is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", total_tokens)

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
    output_text = extract_output_text(response_dict)
    if output_text:
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", output_text)


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


def responses_stream_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for responses.stream (OpenAI Responses streaming API)"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Always treated as streaming
        span = tracer.start_span(RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"})
        set_request_attributes(span, kwargs, "response")

        try:
            start_time = time.time()
            stream_obj = wrapped(*args, **kwargs)
            return StreamingWrapper(span=span, response=stream_obj, start_time=start_time, request_kwargs=kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
            raise

    return wrapper


def aresponses_stream_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for responses.stream (OpenAI Responses streaming API)"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Always treated as streaming
        span = tracer.start_span(RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"})
        set_request_attributes(span, kwargs, "response")

        try:
            start_time = time.time()
            stream_obj = await wrapped(*args, **kwargs)
            return AsyncStreamingWrapper(span=span, response=stream_obj, start_time=start_time, request_kwargs=kwargs)
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.end()
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

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            )

            set_request_attributes(span, kwargs, "response")

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
            # Non-streaming
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

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            )

            set_request_attributes(span, kwargs, "response")

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
            # Non-streaming
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
        self._complete_response: Dict[str, Any] = {"choices": [], "model": "", "output_text": "", "usage": {}}

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

    def __enter__(self):  # type: ignore[no-untyped-def]
        # Support context manager pattern from SDK
        if hasattr(self.__wrapped__, "__enter__"):
            self.__wrapped__.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):  # type:ignore[no-untyped-def]
        try:
            if hasattr(self.__wrapped__, "__exit__"):
                self.__wrapped__.__exit__(exc_type, exc, tb)
        finally:
            # Ensure span is finalized regardless of errors
            self._finalize_span()
        return False

    def _process_chunk(self, chunk: Any) -> None:
        """Process streaming chunk"""
        chunk_dict = model_as_dict(chunk)

        # Accumulate response data
        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        # Accumulate output_text when available (Responses API)
        try:
            text_piece = extract_output_text(chunk_dict)
            if text_piece:
                self._complete_response["output_text"] = self._complete_response.get("output_text", "") + text_piece
        except Exception:
            pass

        # Merge usage if provided in streaming chunks
        usage = chunk_dict.get("usage")
        if isinstance(usage, dict):
            # Last-write-wins merge for simplicity
            stored = self._complete_response.get("usage")
            if not isinstance(stored, dict):
                stored = {}
            stored.update({k: v for k, v in usage.items() if v is not None})
            self._complete_response["usage"] = stored

        # Handle nested response (Responses API 'response.completed' events)
        resp = chunk_dict.get("response")
        if isinstance(resp, dict):
            if resp.get("model"):
                self._complete_response["model"] = resp["model"]
            nested_usage = resp.get("usage")
            if isinstance(nested_usage, dict):
                stored = self._complete_response.get("usage")
                if not isinstance(stored, dict):
                    stored = {}
                stored.update({k: v for k, v in nested_usage.items() if v is not None})
                self._complete_response["usage"] = stored
            try:
                text_piece = extract_output_text(resp)
                if text_piece:
                    self._complete_response["output_text"] = self._complete_response.get("output_text", "") + text_piece
            except Exception:
                pass

        # Accumulate direct delta if present in streaming events
        delta = chunk_dict.get("delta")
        if isinstance(delta, str) and delta:
            self._complete_response["output_text"] = self._complete_response.get("output_text", "") + delta

        # Some SDKs nest payload under 'data'
        data = chunk_dict.get("data")
        if isinstance(data, dict):
            # Merge nested response usage/model
            dresp = data.get("response")
            if isinstance(dresp, dict):
                if dresp.get("model"):
                    self._complete_response["model"] = dresp["model"]
                dusage = dresp.get("usage")
                if isinstance(dusage, dict):
                    stored = self._complete_response.get("usage")
                    if not isinstance(stored, dict):
                        stored = {}
                    stored.update({k: v for k, v in dusage.items() if v is not None})
                    self._complete_response["usage"] = stored
                try:
                    text_piece = extract_output_text(dresp)
                    if text_piece:
                        self._complete_response["output_text"] = (
                            self._complete_response.get("output_text", "") + text_piece
                        )
                except Exception:
                    pass
            # Also check for direct delta in data
            ddelta = data.get("delta")
            if isinstance(ddelta, str) and ddelta:
                self._complete_response["output_text"] = self._complete_response.get("output_text", "") + ddelta

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
        self._complete_response: Dict[str, Any] = {"choices": [], "model": "", "output_text": "", "usage": {}}

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

    async def __aenter__(self):  # type:ignore[no-untyped-def]
        if hasattr(self.__wrapped__, "__aenter__"):
            await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):  # type:ignore[no-untyped-def]
        try:
            if hasattr(self.__wrapped__, "__aexit__"):
                await self.__wrapped__.__aexit__(exc_type, exc, tb)
        finally:
            self._finalize_span()
        return False

    def _process_chunk(self, chunk: Any) -> None:
        """Process streaming chunk"""
        chunk_dict = model_as_dict(chunk)

        # Accumulate response data
        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        # Accumulate output_text when available (Responses API)
        try:
            text_piece = extract_output_text(chunk_dict)
            if text_piece:
                self._complete_response["output_text"] = self._complete_response.get("output_text", "") + text_piece
        except Exception:
            pass

        # Merge usage if provided in streaming chunks
        usage = chunk_dict.get("usage")
        if isinstance(usage, dict):
            stored = self._complete_response.get("usage")
            if not isinstance(stored, dict):
                stored = {}
            stored.update({k: v for k, v in usage.items() if v is not None})
            self._complete_response["usage"] = stored

        # Handle nested response (Responses API 'response.completed' events)
        resp = chunk_dict.get("response")
        if isinstance(resp, dict):
            if resp.get("model"):
                self._complete_response["model"] = resp["model"]
            nested_usage = resp.get("usage")
            if isinstance(nested_usage, dict):
                stored = self._complete_response.get("usage")
                if not isinstance(stored, dict):
                    stored = {}
                stored.update({k: v for k, v in nested_usage.items() if v is not None})
                self._complete_response["usage"] = stored
            try:
                text_piece = extract_output_text(resp)
                if text_piece:
                    self._complete_response["output_text"] = self._complete_response.get("output_text", "") + text_piece
            except Exception:
                pass

        # Accumulate direct delta if present in streaming events
        delta = chunk_dict.get("delta")
        if isinstance(delta, str) and delta:
            self._complete_response["output_text"] = self._complete_response.get("output_text", "") + delta

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
