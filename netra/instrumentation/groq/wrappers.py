import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import ObjectProxy

from netra.instrumentation.groq.utils import (
    model_as_dict,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.utils import record_span_timing

logger = logging.getLogger(__name__)

CHAT_SPAN_NAME = "groq.chat"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"


class StreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Wrapper for streaming responses (OpenAI-style)."""

    def __init__(self, span: Span, response: Iterator[Any], request_kwargs: Dict[str, Any]) -> None:
        super().__init__(response)
        self._span = span
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}
        self._first_content_recorded: bool = False

    def _is_chat(self) -> bool:
        return isinstance(self._request_kwargs, dict) and "messages" in self._request_kwargs

    def _ensure_choice(self, index: int) -> None:
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
        chunk_dict = model_as_dict(chunk)

        if "model" in chunk_dict:
            self._complete_response["model"] = chunk_dict["model"]

        choices = chunk_dict.get("choices")
        if choices:
            for choice in choices:
                index = int(choice.get("index", 0))
                self._ensure_choice(index)

                delta = choice.get("delta")
                if delta:
                    content = delta.get("content")
                    if content:
                        if not self._first_content_recorded:
                            self._first_content_recorded = True
                            first_token_time = time.time()
                            record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                            record_span_timing(
                                self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True
                            )
                        message = self._complete_response["choices"][index].setdefault(
                            "message", {"role": "assistant", "content": ""}
                        )
                        message["content"] += str(content)

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    self._complete_response["choices"][index]["finish_reason"] = finish_reason

        usage = chunk_dict.get("usage")
        if usage and isinstance(usage, dict):
            self._complete_response["usage"] = usage

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._complete_response)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class AsyncStreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Async wrapper for streaming responses (OpenAI-style)."""

    def __init__(self, span: Span, response: AsyncIterator[Any], request_kwargs: Dict[str, Any]) -> None:
        super().__init__(response)
        self._span = span
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}
        self._first_content_recorded: bool = False

    def _is_chat(self) -> bool:
        return isinstance(self._request_kwargs, dict) and "messages" in self._request_kwargs

    def _ensure_choice(self, index: int) -> None:
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
        chunk_dict = model_as_dict(chunk)

        if "model" in chunk_dict:
            self._complete_response["model"] = chunk_dict["model"]

        choices = chunk_dict.get("choices")
        if choices:
            for choice in choices:
                index = int(choice.get("index", 0))
                self._ensure_choice(index)

                delta = choice.get("delta")
                if delta:
                    content = delta.get("content")
                    if content:
                        if not self._first_content_recorded:
                            self._first_content_recorded = True
                            first_token_time = time.time()
                            record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                            record_span_timing(
                                self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True
                            )
                        message = self._complete_response["choices"][index].setdefault(
                            "message", {"role": "assistant", "content": ""}
                        )
                        message["content"] += str(content)

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    self._complete_response["choices"][index]["finish_reason"] = finish_reason

        usage = chunk_dict.get("usage")
        if usage and isinstance(usage, dict):
            self._complete_response["usage"] = usage

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._complete_response)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


def chat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Groq chat completions (OpenAI-style)."""

    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        is_stream = bool(kwargs.get("stream"))
        if is_stream:
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "chat")
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, request_kwargs=kwargs)
            except Exception as e:  # pylint: disable=broad-except
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
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
                    span.set_status(Status(StatusCode.OK))
                except Exception:
                    logger.warning("Failed to set response attributes for Groq span", exc_info=True)
                return response

    return wrapper


def achat_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Async wrapper for Groq chat completions (OpenAI-style)."""

    async def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        is_stream = bool(kwargs.get("stream"))
        if is_stream:
            span = tracer.start_span(CHAT_SPAN_NAME, kind=SpanKind.CLIENT, attributes={"llm.request.type": "chat"})
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, kwargs, "chat")
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, request_kwargs=kwargs)
            except Exception as e:  # pylint: disable=broad-except
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
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
                    span.set_status(Status(StatusCode.OK))
                except Exception:
                    logger.warning("Failed to set response attributes for Groq span", exc_info=True)
                return response

    return wrapper
