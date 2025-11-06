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

    # Chat Completion API
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

    # Response API attributes
    if operation_type == "response":
        if kwargs.get("instructions"):
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.role", "system")
            span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs["instructions"])
        if kwargs.get("input"):
            has_instructions = kwargs.get("instructions") is not None
            if isinstance(kwargs["input"], list) and len(kwargs["input"]) > 0:
                start_index = 1 if has_instructions else 0
                for index, message in enumerate(kwargs["input"]):
                    idx = start_index + index
                    if hasattr(message, "content"):
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.role)
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", message.content)
                    elif isinstance(message, dict):
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{idx}.role", message.get("role", "user"))
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{idx}.content", str(message.get("content", ""))
                        )
            elif isinstance(kwargs["input"], str):
                target_index = 1 if has_instructions else 0
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{target_index}.role", "user")
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{target_index}.content", kwargs["input"])


def set_response_attributes(span: Span, response_dict: Dict[str, Any]) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        return
    if response_dict.get("model"):
        span.set_attribute(f"{SpanAttributes.LLM_RESPONSE_MODEL}", response_dict["model"])

    if response_dict.get("id"):
        span.set_attribute("gen_ai.response.id", response_dict["id"])

    # Usage information (support both chat/completions and responses APIs)
    usage = response_dict.get("usage", {})
    if usage:
        # Legacy keys (chat/completions)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens_details = usage.get("prompt_tokens_details", {})
        # Newer keys (responses API)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        input_tokens_details = usage.get("input_tokens_details", {})

        # Normalize prompt/input tokens
        normalized_prompt = prompt_tokens if prompt_tokens is not None else input_tokens
        normalized_completion = completion_tokens if completion_tokens is not None else output_tokens
        cache_read_input_tokens = (
            prompt_tokens_details.get("cached_tokens") or input_tokens_details.get("cached_tokens") or 0
        )

        if normalized_prompt is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", normalized_prompt)
        if normalized_completion is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", normalized_completion)
        if cache_read_input_tokens:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", cache_read_input_tokens)
        if usage.get("total_tokens") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", usage["total_tokens"])
        if usage.get("output_tokens_details"):
            span.set_attribute("gen_ai.usage.output_tokens_details", usage["output_tokens_details"])

    # Response content
    choices = response_dict.get("choices", [])
    if choices:
        for index, choice in enumerate(choices):
            if choice.get("message", {}).get("role"):
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", choice["message"]["role"])
            # Prefer chat message content when present, else fallback to text completions
            message_content = choice.get("message", {}).get("content")
            if message_content:
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", message_content)
            elif choice.get("text"):
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", choice["text"])
            if choice.get("finish_reason"):
                span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason", choice["finish_reason"])

    if response_dict.get("output"):
        span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
        try:
            span.set_attribute(
                f"{SpanAttributes.LLM_COMPLETIONS}.0.content",
                response_dict["output"][0]["content"][0]["text"],
            )
        except Exception:
            pass

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

    def _is_chat(self) -> bool:
        """Best-effort detection for chat vs completion request."""
        # Presence of "messages" strongly indicates chat endpoints
        return isinstance(self._request_kwargs, dict) and "messages" in self._request_kwargs

    def _ensure_choice(self, index: int) -> None:
        """Ensure choices list has an entry at index."""
        while len(self._complete_response["choices"]) <= index:
            if self._is_chat():
                self._complete_response["choices"].append({"message": {"role": "assistant", "content": ""}})
            else:
                self._complete_response["choices"].append({"text": ""})

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

        # Aggregate choices/content for chat and completion streams
        choices = chunk_dict.get("choices") or []
        if isinstance(choices, list):
            for choice in choices:
                try:
                    idx = int(choice.get("index", 0))
                except Exception:
                    idx = 0
                self._ensure_choice(idx)

                # Chat streaming: content in choices[].delta.content
                delta = choice.get("delta") or {}
                content_piece = None
                if isinstance(delta, dict) and delta.get("content"):
                    content_piece = str(delta.get("content", ""))
                    self._complete_response["choices"][idx].setdefault("message", {"role": "assistant", "content": ""})
                    self._complete_response["choices"][idx]["message"]["content"] += content_piece

                # Legacy/text completion streaming: content in choices[].text
                if content_piece is None and choice.get("text"):
                    text_piece = str(choice.get("text", ""))
                    self._complete_response["choices"][idx]["text"] = (
                        self._complete_response["choices"][idx].get("text", "") + text_piece
                    )

                # Finish reason when available
                if choice.get("finish_reason"):
                    self._complete_response["choices"][idx]["finish_reason"] = choice.get("finish_reason")

        # Usage sometimes appears in the final chunk if include_usage=True
        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

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

    def _is_chat(self) -> bool:
        """Best-effort detection for chat vs completion request."""
        return isinstance(self._request_kwargs, dict) and "messages" in self._request_kwargs

    def _ensure_choice(self, index: int) -> None:
        """Ensure choices list has an entry at index."""
        while len(self._complete_response["choices"]) <= index:
            if self._is_chat():
                self._complete_response["choices"].append({"message": {"role": "assistant", "content": ""}})
            else:
                self._complete_response["choices"].append({"text": ""})

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

        # Aggregate choices/content for chat and completion streams
        choices = chunk_dict.get("choices") or []
        if isinstance(choices, list):
            for choice in choices:
                try:
                    idx = int(choice.get("index", 0))
                except Exception:
                    idx = 0
                self._ensure_choice(idx)

                # Chat streaming: content in choices[].delta.content
                delta = choice.get("delta") or {}
                content_piece = None
                if isinstance(delta, dict) and delta.get("content"):
                    content_piece = str(delta.get("content", ""))
                    self._complete_response["choices"][idx].setdefault("message", {"role": "assistant", "content": ""})
                    self._complete_response["choices"][idx]["message"]["content"] += content_piece

                # Legacy/text completion streaming: content in choices[].text
                if content_piece is None and choice.get("text"):
                    text_piece = str(choice.get("text", ""))
                    self._complete_response["choices"][idx]["text"] = (
                        self._complete_response["choices"][idx].get("text", "") + text_piece
                    )

                # Finish reason when available
                if choice.get("finish_reason"):
                    self._complete_response["choices"][idx]["finish_reason"] = choice.get("finish_reason")

        # Usage sometimes appears in the final chunk if include_usage=True
        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

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
