import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.semconv_ai import LLMRequestTypeValues
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

from netra.instrumentation.cerebras.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)

logger = logging.getLogger(__name__)

CHAT_SPAN_NAME = "cerebras.chat.completions"
COMPLETION_SPAN_NAME = "cerebras.completions"


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
                elif choice.get("text"):
                    content_piece = str(choice.get("text", ""))
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

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

                elif choice.get("text"):
                    content_piece = str(choice.get("text", ""))
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        end_time = time.time()
        duration = end_time - self._start_time
        set_response_attributes(self._span, self._complete_response)
        self._span.set_attribute("llm.response.duration", duration)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        is_streaming = _detect_streaming(args, kwargs)
        if is_streaming:
            span = tracer.start_span(
                CHAT_SPAN_NAME,
                kind=SpanKind.CLIENT,
                attributes={"gen_ai.system": "Cerebras", "llm.request.type": "chat"},
            )
            try:
                ctx = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, LLMRequestTypeValues.CHAT, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.cerebras: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            finally:
                context_api.detach(ctx)

        else:
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME,
                kind=SpanKind.CLIENT,
                attributes={"gen_ai.system": "Cerebras", "llm.request.type": "chat"},
            ) as span:
                try:
                    set_request_attributes(span, LLMRequestTypeValues.CHAT, kwargs)
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.cerebras: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        is_streaming = _detect_streaming(args, kwargs)
        if is_streaming:
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
            try:
                ctx = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, LLMRequestTypeValues.CHAT, kwargs)
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, start_time=start_time, request_kwargs=kwargs)
            except Exception as e:
                logger.error("netra.instrumentation.cerebras: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            finally:
                context_api.detach(ctx)
        else:
            with tracer.start_as_current_span(
                CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"}
            ) as span:
                try:
                    set_request_attributes(span, LLMRequestTypeValues.CHAT, kwargs)
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


def completions_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        is_streaming = _detect_streaming(args, kwargs)
        if is_streaming:
            span = tracer.start_span(
                COMPLETION_SPAN_NAME,
                kind=SpanKind.CLIENT,
                attributes={"gen_ai.system": "Cerebras", "llm.request.type": "completion"},
            )
            try:
                ctx = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, LLMRequestTypeValues.COMPLETION, kwargs)
                start_time = time.time()
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(
                    span=span,
                    response=response,
                    start_time=start_time,
                    request_kwargs=kwargs,
                )
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            finally:
                context_api.detach(ctx)
        else:
            with tracer.start_as_current_span(
                COMPLETION_SPAN_NAME,
                kind=SpanKind.CLIENT,
                attributes={"gen_ai.system": "Cerebras", "llm.request.type": "completion"},
            ) as span:
                try:
                    set_request_attributes(span, LLMRequestTypeValues.COMPLETION, kwargs)
                    start_time = time.time()
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

    return wrapper


def acompletions_wrapper(tracer: Tracer) -> Callable[..., Any]:
    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        is_streaming = _detect_streaming(args, kwargs)
        if is_streaming:
            span = tracer.start_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            )
            try:
                ctx = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, LLMRequestTypeValues.COMPLETION, kwargs)
                start_time = time.time()
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(
                    span=span,
                    response=response,
                    start_time=start_time,
                    request_kwargs=kwargs,
                )
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                context_api.detach(ctx)
        else:
            with tracer.start_as_current_span(
                COMPLETION_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "completion"}
            ) as span:
                try:
                    set_request_attributes(span, LLMRequestTypeValues.COMPLETION, kwargs)
                    start_time = time.time()
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    span.set_attribute("llm.response.duration", end_time - start_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

    return wrapper
