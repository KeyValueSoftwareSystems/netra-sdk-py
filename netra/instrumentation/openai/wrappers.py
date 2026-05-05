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
from netra.instrumentation.utils import record_span_timing

logger = logging.getLogger(__name__)

# Span names
CHAT_SPAN_NAME = "openai.chat"
EMBEDDING_SPAN_NAME = "openai.embedding"
RESPONSE_SPAN_NAME = "openai.response"
TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"
LLM_RESPONSE_DURATION = "llm.response.duration"


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
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, request_kwargs=kwargs)
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
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
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
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, request_kwargs=kwargs)
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
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
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
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
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
                response = await wrapped(*args, **kwargs)
                end_time = time.time()
                response_dict = model_as_dict(response)
                set_response_attributes(span, response_dict)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
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
                response = wrapped(*args, **kwargs)
                return StreamingWrapper(span=span, response=response, request_kwargs=kwargs)
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
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
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
                response = await wrapped(*args, **kwargs)
                return AsyncStreamingWrapper(span=span, response=response, request_kwargs=kwargs)
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
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    response_dict = model_as_dict(response)
                    set_response_attributes(span, response_dict)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    record_span_timing(span, TIME_TO_FIRST_TOKEN, end_time)
                    record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, end_time, use_root_span=True)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    logger.error("netra.instrumentation.openai: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

    return wrapper


class StreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Wrapper for streaming responses"""

    def __init__(self, span: Span, response: Iterator[Any], request_kwargs: Dict[str, Any]) -> None:
        super().__init__(response)
        self._span = span
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}
        self._first_content_recorded: bool = False

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

    def __enter__(self) -> "StreamingWrapper":
        if hasattr(self.__wrapped__, "__enter__"):
            self.__wrapped__.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.__wrapped__, "__exit__"):
            self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self._span.record_exception(exc_val)
            self._span.end()

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
                    if content_piece and not self._first_content_recorded:
                        self._first_content_recorded = True
                        first_token_time = time.time()
                        record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                        record_span_timing(
                            self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True
                        )
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if isinstance(delta, dict) and delta.get("tool_calls"):
                    msg = self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    tc_acc = msg.setdefault("tool_calls", {})
                    for tc_delta in delta["tool_calls"]:
                        tc_index = tc_delta.get("index", 0)
                        if tc_index not in tc_acc:
                            tc_acc[tc_index] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        if tc_delta.get("id"):
                            tc_acc[tc_index]["id"] = tc_delta["id"]
                        if func := tc_delta.get("function"):
                            tc_acc[tc_index]["function"]["name"] += func.get("name") or ""
                            tc_acc[tc_index]["function"]["arguments"] += func.get("arguments") or ""

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Response API
        if chunk_dict.get("delta") and not self._first_content_recorded:
            self._first_content_recorded = True
            first_token_time = time.time()
            record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
            record_span_timing(self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)
        if chunk_dict.get("response"):
            response = chunk_dict.get("response", {})
            if response.get("status") == "completed":
                response_output = response.get("output") or []
                for output in response_output:
                    if output.get("type") == "function_call":
                        self._complete_response.setdefault("output", []).append(output)
                    else:
                        assistant_text = ""
                        for content_chunk in output.get("content") or []:
                            assistant_text += content_chunk.get("text", "")
                        self._complete_response["choices"] = [
                            {"message": {"role": "assistant", "content": assistant_text}}
                        ]

                usage = response.get("usage", {})
                self._complete_response["usage"] = usage

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        for choice in self._complete_response.get("choices", []):
            msg = choice.get("message", {})
            if isinstance(msg.get("tool_calls"), dict):
                msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"].keys())]
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._complete_response)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class AsyncStreamingWrapper(ObjectProxy):  # type: ignore[misc]
    """Async wrapper for streaming responses"""

    def __init__(self, span: Span, response: AsyncIterator[Any], request_kwargs: Dict[str, Any]) -> None:
        super().__init__(response)
        self._span = span
        self._request_kwargs = request_kwargs
        self._complete_response: Dict[str, Any] = {"choices": [], "model": ""}
        self._first_content_recorded: bool = False

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

    async def __aenter__(self) -> "AsyncStreamingWrapper":
        if hasattr(self.__wrapped__, "__aenter__"):
            await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.__wrapped__, "__aexit__"):
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self._span.record_exception(exc_val)
            self._span.end()

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
                    if content_piece and not self._first_content_recorded:
                        self._first_content_recorded = True
                        first_token_time = time.time()
                        record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
                        record_span_timing(
                            self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True
                        )
                    self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    self._complete_response["choices"][index]["message"]["content"] += content_piece

                if isinstance(delta, dict) and delta.get("tool_calls"):
                    msg = self._complete_response["choices"][index].setdefault(
                        "message", {"role": "assistant", "content": ""}
                    )
                    tc_acc = msg.setdefault("tool_calls", {})
                    for tc_delta in delta["tool_calls"]:
                        tc_index = tc_delta.get("index", 0)
                        if tc_index not in tc_acc:
                            tc_acc[tc_index] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        if tc_delta.get("id"):
                            tc_acc[tc_index]["id"] = tc_delta["id"]
                        if func := tc_delta.get("function"):
                            tc_acc[tc_index]["function"]["name"] += func.get("name") or ""
                            tc_acc[tc_index]["function"]["arguments"] += func.get("arguments") or ""

                if choice.get("finish_reason"):
                    self._complete_response["choices"][index]["finish_reason"] = choice.get("finish_reason")

        if chunk_dict.get("usage") and isinstance(chunk_dict["usage"], dict):
            self._complete_response["usage"] = chunk_dict["usage"]

        # Response API
        if chunk_dict.get("delta") and not self._first_content_recorded:
            self._first_content_recorded = True
            first_token_time = time.time()
            record_span_timing(self._span, TIME_TO_FIRST_TOKEN, first_token_time)
            record_span_timing(self._span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)
        if chunk_dict.get("response"):
            response = chunk_dict.get("response", {})
            if response.get("status") == "completed":
                response_output = response.get("output") or []
                for output in response_output:
                    if output.get("type") == "function_call":
                        self._complete_response.setdefault("output", []).append(output)
                    else:
                        assistant_text = ""
                        for content_chunk in output.get("content") or []:
                            assistant_text += content_chunk.get("text", "")
                        self._complete_response["choices"] = [
                            {"message": {"role": "assistant", "content": assistant_text}}
                        ]

                usage = response.get("usage", {})
                self._complete_response["usage"] = usage

        self._span.add_event("llm.content.completion.chunk")

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete"""
        for choice in self._complete_response.get("choices", []):
            msg = choice.get("message", {})
            if isinstance(msg.get("tool_calls"), dict):
                msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"].keys())]
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        set_response_attributes(self._span, self._complete_response)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
