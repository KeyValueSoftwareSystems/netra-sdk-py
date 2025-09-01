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

COMPLETION_SPAN_NAME = "litellm.completion"
EMBEDDING_SPAN_NAME = "litellm.embedding"
IMAGE_GENERATION_SPAN_NAME = "litellm.image_generation"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def is_streaming_response(response: Any) -> bool:
    """Check if response is a streaming response"""
    return hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict))


def model_as_dict(obj: Any) -> Dict[str, Any]:
    """Convert LiteLLM model object to dictionary"""
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
    try:
        # Set operation type
        span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TYPE}", operation_type)
        span.set_attribute(f"{SpanAttributes.LLM_SYSTEM}", "LiteLLM")

        # Common attributes
        if kwargs.get("model"):
            span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MODEL}", kwargs["model"])

        if kwargs.get("temperature") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_REQUEST_TEMPERATURE}", kwargs["temperature"])

        if kwargs.get("max_tokens") is not None:
            span.set_attribute(f"{SpanAttributes.LLM_REQUEST_MAX_TOKENS}", kwargs["max_tokens"])

        if kwargs.get("stream") is not None:
            span.set_attribute("gen_ai.stream", kwargs["stream"])

        # Chat completion specific attributes
        if operation_type == "chat" and kwargs.get("messages"):
            messages = kwargs["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                for index, message in enumerate(messages):
                    if isinstance(message, dict):
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.role", message.get("role", "user"))
                        span.set_attribute(
                            f"{SpanAttributes.LLM_PROMPTS}.{index}.content", str(message.get("content", ""))
                        )

        # Embedding specific attributes
        if operation_type == "embedding" and kwargs.get("input"):
            input_data = kwargs["input"]
            if isinstance(input_data, str):
                span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.0.content", input_data)
            elif isinstance(input_data, list):
                for index, text in enumerate(input_data):
                    if isinstance(text, str):
                        span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.{index}.content", text)

        # Image generation specific attributes
        if operation_type == "image_generation":
            if kwargs.get("prompt"):
                span.set_attribute("gen_ai.prompt", kwargs["prompt"])
            if kwargs.get("n"):
                span.set_attribute("gen_ai.request.n", kwargs["n"])
            if kwargs.get("size"):
                span.set_attribute("gen_ai.request.size", kwargs["size"])
            if kwargs.get("quality"):
                span.set_attribute("gen_ai.request.quality", kwargs["quality"])
            if kwargs.get("style"):
                span.set_attribute("gen_ai.request.style", kwargs["style"])
    except Exception as e:
        logger.error(f"Failed to set attributes for LiteLLM span: {e}")


def set_response_attributes(span: Span, response_dict: Dict[str, Any], operation_type: str) -> None:
    """Set response attributes on span"""
    if not span.is_recording():
        return
    try:
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
            if usage.get("cache_read_input_tokens"):
                span.set_attribute(
                    f"{SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS}", usage["cache_read_input_tokens"]
                )
            if usage.get("cache_creation_input_tokens"):
                span.set_attribute("gen_ai.usage.cache_creation_input_tokens", usage["cache_creation_input_tokens"])
            if usage.get("total_tokens"):
                span.set_attribute(f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", usage["total_tokens"])

        # Chat completion response content
        if operation_type == "chat":
            choices = response_dict.get("choices", [])
            for index, choice in enumerate(choices):
                if choice.get("message", {}).get("role"):
                    span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", choice["message"]["role"])
                if choice.get("message", {}).get("content"):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", choice["message"]["content"]
                    )
                if choice.get("finish_reason"):
                    span.set_attribute(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{index}.finish_reason", choice["finish_reason"]
                    )

        # Embedding response content
        elif operation_type == "embedding":
            data = response_dict.get("data", [])
            for index, embedding_data in enumerate(data):
                if embedding_data.get("index") is not None:
                    span.set_attribute(f"gen_ai.response.embeddings.{index}.index", embedding_data["index"])
                if embedding_data.get("embedding"):
                    # Don't log the actual embedding vector, just its dimensions
                    embedding_vector = embedding_data["embedding"]
                    if isinstance(embedding_vector, list):
                        span.set_attribute(f"gen_ai.response.embeddings.{index}.dimensions", len(embedding_vector))

        # Image generation response content
        elif operation_type == "image_generation":
            data = response_dict.get("data", [])
            for index, image_data in enumerate(data):
                if image_data.get("url"):
                    span.set_attribute(f"gen_ai.response.images.{index}.url", image_data["url"])
                if image_data.get("b64_json"):
                    span.set_attribute(f"gen_ai.response.images.{index}.has_b64_json", True)
                if image_data.get("revised_prompt"):
                    span.set_attribute(f"gen_ai.response.images.{index}.revised_prompt", image_data["revised_prompt"])
    except Exception as e:
        logger.error(f"Failed to set attributes for LiteLLM span: {e}")


def completion_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for LiteLLM completion function"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        logger.debug(f"LiteLLM completion wrapper called with model: {kwargs.get('model')}")

        if should_suppress_instrumentation():
            logger.debug("LiteLLM instrumentation suppressed")
            return wrapped(*args, **kwargs)

        # Check if streaming
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            )

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
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                set_request_attributes(span, kwargs, "chat")

                try:
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict, "chat")

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def acompletion_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for LiteLLM acompletion function"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Check if streaming
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Use start_span for streaming - returns span directly
            span = tracer.start_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            )

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
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                set_request_attributes(span, kwargs, "chat")

                try:
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()

                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict, "chat")

                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))

                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def embedding_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for LiteLLM embedding function"""

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
                set_response_attributes(span, response_dict, "embedding")

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def aembedding_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for LiteLLM aembedding function"""

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
                set_response_attributes(span, response_dict, "embedding")

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def image_generation_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for LiteLLM image_generation function"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Image generation is never streaming, always use start_as_current_span
        with tracer.start_as_current_span(
            IMAGE_GENERATION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "image_generation"}
        ) as span:
            set_request_attributes(span, kwargs, "image_generation")

            try:
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict, "image_generation")

                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))

                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def aimage_generation_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for LiteLLM aimage_generation function"""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Image generation is never streaming, always use start_as_current_span
        with tracer.start_as_current_span(
            IMAGE_GENERATION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "image_generation"}
        ) as span:
            set_request_attributes(span, kwargs, "image_generation")

            try:
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                end_time = time.time()

                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict, "image_generation")

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
        self._content_parts: list[str] = []

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

        # Accumulate usage information from chunks
        if chunk_dict.get("usage"):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Collect content from delta
        choices = chunk_dict.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            if delta.get("content"):
                self._content_parts.append(delta["content"])

            # Collect finish_reason from choices
            if choice.get("finish_reason"):
                if "choices" not in self._complete_response:
                    self._complete_response["choices"] = []
                # Ensure we have enough choice entries
                while len(self._complete_response["choices"]) <= len(choices) - 1:
                    self._complete_response["choices"].append(
                        {"message": {"role": "assistant", "content": ""}, "finish_reason": None}
                    )

                choice_index = choice.get("index", 0)
                if choice_index < len(self._complete_response["choices"]):
                    self._complete_response["choices"][choice_index]["finish_reason"] = choice["finish_reason"]

        # Add chunk event
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time

        # Set accumulated content
        if self._content_parts:
            full_content = "".join(self._content_parts)
            self._span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", full_content)
            self._span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")

        set_response_attributes(self._span, self._complete_response, "chat")
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
        self._content_parts: list[str] = []

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

        # Accumulate usage information from chunks
        if chunk_dict.get("usage"):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Collect content from delta
        choices = chunk_dict.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            if delta.get("content"):
                self._content_parts.append(delta["content"])

            # Collect finish_reason from choices
            if choice.get("finish_reason"):
                if "choices" not in self._complete_response:
                    self._complete_response["choices"] = []
                # Ensure we have enough choice entries
                while len(self._complete_response["choices"]) <= len(choices) - 1:
                    self._complete_response["choices"].append(
                        {"message": {"role": "assistant", "content": ""}, "finish_reason": None}
                    )

                choice_index = choice.get("index", 0)
                if choice_index < len(self._complete_response["choices"]):
                    self._complete_response["choices"][choice_index]["finish_reason"] = choice["finish_reason"]

        # Add chunk event
        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time

        # Set accumulated content
        if self._content_parts:
            full_content = "".join(self._content_parts)
            self._span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.content", full_content)
            self._span.set_attribute(f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")

        set_response_attributes(self._span, self._complete_response, "chat")
        self._span.set_attribute("llm.response.duration", duration)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
