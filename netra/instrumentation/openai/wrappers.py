import logging
import time
from collections.abc import Awaitable
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

from netra.instrumentation.openai.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

# Span names
CHAT_SPAN_NAME = "openai.chat"
EMBEDDING_SPAN_NAME = "openai.embedding"
RESPONSE_SPAN_NAME = "openai.response"


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for chat completions"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "chat")
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
            finally:
                context_api.detach(context)

        else:
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                try:
                    set_request_attributes(span, kwargs, "chat")
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.openai: %s", e)
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

        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "chat")
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
            finally:
                context_api.detach(context)
        else:
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                try:
                    set_request_attributes(span, kwargs, "chat")
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.openai: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def embeddings_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for embeddings"""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            EMBEDDING_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "embedding"}
        ) as span:
            try:
                set_request_attributes(span, kwargs, "embedding")
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)
                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
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

        with tracer.start_as_current_span(
            EMBEDDING_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "embedding"}
        ) as span:
            try:
                set_request_attributes(span, kwargs, "embedding")
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                end_time = time.time()
                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)
                span.set_attribute("llm.response.duration", end_time - start_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
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
            span = tracer.start_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            )
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "response")
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
            finally:
                context_api.detach(context)
        else:
            with tracer.start_as_current_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            ) as span:
                try:
                    set_request_attributes(span, kwargs, "response")
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.openai: %s", e)
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
            span = tracer.start_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            )
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "response")
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.openai: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
            finally:
                context_api.detach(context)
        else:
            with tracer.start_as_current_span(
                RESPONSE_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "response"}
            ) as span:
                try:
                    set_request_attributes(span, kwargs, "response")
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.openai: %s", e)
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
        """Determine if the request is a chat request."""
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

        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        choices = chunk_dict.get("choices") or []

        # Completion API
        if isinstance(choices, list):
            for choice in choices:
                index = int(choice.get("index", 0))
                self._ensure_choice(index)
                delta = choice.get("delta") or {}
                content_piece = None
                if isinstance(delta, dict) and delta.get("content"):
                    content_piece = str(delta.get("content", ""))
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Response API
        if chunk_dict.get("response"):
            response = chunk_dict.get("response", {})
            if response.get("status") == "completed":
                response_output = response.get("output", {})
                for output in response_output:
                    content = output.get("content")
                    for index, chunk in enumerate(content):
                        assistant_text = chunk.get("text", "")
                        self._complete_response["choices"] = [
                            {"message": {"role": "assistant", "content": assistant_text}}
                        ]

                usage = response.get("usage", {})
                self._complete_response["usage"] = usage

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
        """Determine if the request is a chat request."""
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

        if chunk_dict.get("model"):
            self._complete_response["model"] = chunk_dict["model"]

        choices = chunk_dict.get("choices") or []

        # Completion API
        if isinstance(choices, list):
            for choice in choices:
                index = int(choice.get("index", 0))
                self._ensure_choice(index)
                delta = choice.get("delta") or {}
                content_piece = None
                if isinstance(delta, dict) and delta.get("content"):
                    content_piece = str(delta.get("content", ""))
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Response API
        if chunk_dict.get("response"):
            response = chunk_dict.get("response", {})
            if response.get("status") == "completed":
                response_output = response.get("output", {})
                for output in response_output:
                    content = output.get("content")
                    for index, chunk in enumerate(content):
                        assistant_text = chunk.get("text", "")
                        self._complete_response["choices"] = [
                            {"message": {"role": "assistant", "content": assistant_text}}
                        ]

                usage = response.get("usage", {})
                self._complete_response["usage"] = usage

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time
        set_response_attributes(self._span, self._complete_response)
        self._span.set_attribute("llm.response.duration", duration)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
