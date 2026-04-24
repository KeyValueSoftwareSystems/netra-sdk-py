import logging
import time
from collections.abc import Awaitable
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.agno.utils import (
    _ENTITY_SPAN_TYPE_MAP,
    ATTR_ENTITY,
    LLM_SYSTEM_AGNO,
    NETRA_SPAN_TYPE,
    extract_knowledge_attributes,
    extract_memory_attributes,
    extract_vectordb_attributes,
    get_tool_arguments,
    get_tool_name,
    is_run_content,
    serialize_value,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.instrumentation.utils import record_span_timing
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

# Span name constants
AGENT_RUN_SPAN = "agno.agent.run"
AGENT_CONTINUE_RUN_SPAN = "agno.agent.continue_run"
TEAM_RUN_SPAN = "agno.team.run"
WORKFLOW_RUN_SPAN = "agno.workflow.run"
TOOL_EXECUTE_SPAN = "agno.tool.execute"
VECTORDB_SEARCH_SPAN = "agno.vectordb.search"
VECTORDB_UPSERT_SPAN = "agno.vectordb.upsert"
MEMORY_ADD_SPAN = "agno.memory.add"
MEMORY_SEARCH_SPAN = "agno.memory.search"
KNOWLEDGE_SEARCH_SPAN = "agno.knowledge.search"
LLM_RESPONSE_DURATION = "llm.response.duration"


class SpanStreamingWrapper:
    """
    Wraps a synchronous streaming response to keep the OTel span open.

    The span is finalized — response attributes set, status written, context
    detached, span ended — when iteration completes or an unhandled exception
    occurs.
    """

    def __init__(
        self,
        span: Span,
        response: Iterator[Any],
        ctx_token: Any = None,
    ) -> None:
        self._span = span
        self._response = response
        self._ctx_token = ctx_token
        self._last_response: Any = None
        self._content_chunks: List[str] = []
        self._finalized = False

    def __enter__(self) -> "SpanStreamingWrapper":
        if hasattr(self._response, "__enter__"):
            try:
                self._response.__enter__()
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in stream __enter__: %s", e)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._response, "__exit__"):
            try:
                self._response.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in stream __exit__: %s", e)
        if exc_type is not None:
            self._finalize(error=exc_val)

    def __iter__(self) -> "SpanStreamingWrapper":
        return self

    def __next__(self) -> Any:
        try:
            event = next(self._response)
            self._last_response = event
            try:
                if is_run_content(event):
                    content = getattr(event, "content", None)
                    if content:
                        self._content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate stream content: %s", e)
            return event
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _finalize(self, error: Optional[Exception] = None) -> None:
        if self._finalized:
            return
        self._finalized = True

        if error is None:
            try:
                if self._last_response is not None:
                    set_response_attributes(self._span, self._last_response)
                if self._content_chunks:
                    self._span.set_attribute("output", "".join(self._content_chunks))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes on stream end: %s", e)

        try:
            record_span_timing(self._span, LLM_RESPONSE_DURATION)
            if error is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
                self._span.record_exception(error)
            else:
                self._span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to set span status after streaming: %s", e)

        try:
            if self._ctx_token is not None:
                context_api.detach(self._ctx_token)
        except Exception as e:
            logger.debug("netra.instrumentation.agno: failed to detach context after streaming: %s", e)

        try:
            self._span.end()
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to end span after streaming: %s", e)

    def __del__(self) -> None:
        try:
            if not self._finalized:
                self._finalize()
        except Exception:
            pass


class AsyncSpanStreamingWrapper:
    """
    Wraps an asynchronous streaming response to keep the OTel span open.

    The span is finalized — response attributes set, status written, context
    detached, span ended — when async iteration completes or an unhandled
    exception occurs.
    """

    def __init__(
        self,
        span: Span,
        response: AsyncIterator[Any],
        ctx_token: Any = None,
    ) -> None:
        self._span = span
        self._response = response
        self._ctx_token = ctx_token
        self._last_response: Any = None
        self._content_chunks: List[str] = []
        self._finalized = False

    async def __aenter__(self) -> "AsyncSpanStreamingWrapper":
        if hasattr(self._response, "__aenter__"):
            try:
                await self._response.__aenter__()
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in async stream __aenter__: %s", e)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._response, "__aexit__"):
            try:
                await self._response.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in async stream __aexit__: %s", e)
        if exc_type is not None:
            self._finalize(error=exc_val)

    def __aiter__(self) -> "AsyncSpanStreamingWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._response.__anext__()
            self._last_response = event
            try:
                if is_run_content(event):
                    content = getattr(event, "content", None)
                    if content:
                        self._content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate async stream content: %s", e)
            return event
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _finalize(self, error: Optional[Exception] = None) -> None:
        if self._finalized:
            return
        self._finalized = True

        if error is None:
            try:
                if self._last_response is not None:
                    set_response_attributes(self._span, self._last_response)
                if self._content_chunks:
                    self._span.set_attribute("output", "".join(self._content_chunks))
            except Exception as e:
                logger.warning(
                    "netra.instrumentation.agno: failed to set response attributes on async stream end: %s", e
                )

        try:
            record_span_timing(self._span, LLM_RESPONSE_DURATION)
            if error is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(error)))
                self._span.record_exception(error)
            else:
                self._span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to set span status after async streaming: %s", e)

        try:
            if self._ctx_token is not None:
                context_api.detach(self._ctx_token)
        except Exception as e:
            logger.debug("netra.instrumentation.agno: failed to detach context after async streaming: %s", e)

        try:
            self._span.end()
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to end span after async streaming: %s", e)

    def __del__(self) -> None:
        try:
            if not self._finalized:
                self._finalize()
        except Exception:
            pass


def _get_span_name(instance: Any, prefix: str, default: Optional[str] = None) -> str:
    name = getattr(instance, "name", None) or default or "unknown"
    return f"{prefix}.{name}" if name else prefix


def _set_common_span_attributes(span: Span, entity_type: str) -> None:
    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))


def _sync_non_stream(
    tracer: Tracer,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": request_type})
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
        return wrapped(*args, **kwargs)

    ctx_token = None
    try:
        ctx_token = context_api.attach(set_span_in_context(span))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", span_name, e)

    try:
        set_request_attributes(span, instance, args, kwargs, request_type)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

    try:
        response = wrapped(*args, **kwargs)
        end_time = time.time()
        try:
            set_response_attributes(span, response)
            record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e)
        return response
    except Exception as e:
        try:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        except Exception as span_err:
            logger.error("netra.instrumentation.agno: failed to record error for %s: %s", span_name, span_err)
        raise
    finally:
        try:
            if ctx_token is not None:
                context_api.detach(ctx_token)
            span.end()
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to end span for %s: %s", span_name, e)


def _sync_stream_start(
    tracer: Tracer,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": request_type})
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
        return wrapped(*args, **kwargs)

    try:
        ctx_token = context_api.attach(set_span_in_context(span))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach span context for %s: %s", span_name, e)
        try:
            span.end()
        except Exception:
            pass
        return wrapped(*args, **kwargs)

    try:
        set_request_attributes(span, instance, args, kwargs, request_type)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        try:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            context_api.detach(ctx_token)
            span.end()
        except Exception as span_err:
            logger.error("netra.instrumentation.agno: failed to finalize span on %s error: %s", span_name, span_err)
        raise

    return SpanStreamingWrapper(
        span=span,
        response=response,
        ctx_token=ctx_token,
    )


async def _async_non_stream(
    tracer: Tracer,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": request_type})
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
        return await wrapped(*args, **kwargs)

    ctx_token = None
    try:
        ctx_token = context_api.attach(set_span_in_context(span))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", span_name, e)

    try:
        set_request_attributes(span, instance, args, kwargs, request_type)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

    try:
        response = await wrapped(*args, **kwargs)
        end_time = time.time()
        try:
            set_response_attributes(span, response)
            record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e)
        return response
    except Exception as e:
        try:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
        except Exception as span_err:
            logger.error("netra.instrumentation.agno: failed to record error for %s: %s", span_name, span_err)
        raise
    finally:
        try:
            if ctx_token is not None:
                context_api.detach(ctx_token)
            span.end()
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to end span for %s: %s", span_name, e)


def _async_stream_start(
    tracer: Tracer,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": request_type})
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
        return wrapped(*args, **kwargs)

    try:
        ctx_token = context_api.attach(set_span_in_context(span))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach span context for %s: %s", span_name, e)
        try:
            span.end()
        except Exception as end_err:
            logger.error("netra.instrumentation.agno: failed to end span for %s: %s", span_name, end_err)
        return wrapped(*args, **kwargs)

    try:
        set_request_attributes(span, instance, args, kwargs, request_type)
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        try:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            context_api.detach(ctx_token)
            span.end()
        except Exception as span_err:
            logger.error("netra.instrumentation.agno: failed to finalize span on %s error: %s", span_name, span_err)
        raise

    return AsyncSpanStreamingWrapper(
        span=span,
        response=response,
        ctx_token=ctx_token,
    )


def agent_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_RUN_SPAN)

        if kwargs.get("stream"):
            return _sync_stream_start(tracer, wrapped, instance, args, kwargs, span_name, "agent")
        return _sync_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "agent")

    return wrapper


def agent_arun_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_RUN_SPAN)

        if kwargs.get("stream"):
            return _async_stream_start(tracer, wrapped, instance, args, kwargs, span_name, "agent")
        return _async_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "agent")

    return wrapper


def agent_continue_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_CONTINUE_RUN_SPAN)
        return _sync_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "agent")

    return wrapper


def agent_acontinue_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_CONTINUE_RUN_SPAN)
        return _async_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "agent")

    return wrapper


def team_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, TEAM_RUN_SPAN)
        return _sync_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "team")

    return wrapper


def team_arun_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, TEAM_RUN_SPAN)
        return _async_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "team")

    return wrapper


def workflow_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, WORKFLOW_RUN_SPAN)

        if kwargs.get("stream"):
            return _sync_stream_start(tracer, wrapped, instance, args, kwargs, span_name, "workflow")
        return _sync_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "workflow")

    return wrapper


def workflow_arun_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, WORKFLOW_RUN_SPAN)

        if kwargs.get("stream"):
            return _async_stream_start(tracer, wrapped, instance, args, kwargs, span_name, "workflow")
        return _async_non_stream(tracer, wrapped, instance, args, kwargs, span_name, "workflow")

    return wrapper


def tool_execute_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        tool_name = get_tool_name(instance)

        try:
            span = tracer.start_span(tool_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "tool"})
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", tool_name, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", tool_name, e)

        try:
            set_request_attributes(span, instance, args, kwargs, "tool")
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", tool_name, e)

        try:
            arguments = get_tool_arguments(instance, kwargs)
            if arguments:
                span.set_attribute("input", arguments)
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set tool arguments for %s: %s", tool_name, e)

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                if response is not None:
                    span.set_attribute("output", serialize_value(response, clean=True))
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", tool_name, e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error("netra.instrumentation.agno: failed to record error for %s: %s", tool_name, span_err)
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", tool_name, e)

    return wrapper


def tool_aexecute_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        tool_name = get_tool_name(instance)

        try:
            span = tracer.start_span(tool_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "tool"})
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", tool_name, e)
            return await wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", tool_name, e)

        try:
            set_request_attributes(span, instance, args, kwargs, "tool")
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", tool_name, e)

        try:
            arguments = get_tool_arguments(instance, kwargs)
            if arguments:
                span.set_attribute("input", arguments)
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set tool arguments for %s: %s", tool_name, e)

        try:
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                if response is not None:
                    span.set_attribute("output", serialize_value(response, clean=True))
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", tool_name, e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error("netra.instrumentation.agno: failed to record error for %s: %s", tool_name, span_err)
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", tool_name, e)

    return wrapper


def vectordb_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, VECTORDB_SEARCH_SPAN, type(instance).__name__)

        try:
            span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"})
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", span_name, e)

        try:
            _set_common_span_attributes(span, "vectordb")
            span.set_attributes(extract_vectordb_attributes(instance, "search"))
            if args:
                span.set_attribute("input", str(args[0]))
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error("netra.instrumentation.agno: failed to record error for %s: %s", span_name, span_err)
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", span_name, e)

    return wrapper


def vectordb_upsert_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, VECTORDB_UPSERT_SPAN, type(instance).__name__)

        try:
            span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"})
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", span_name, e)

        try:
            _set_common_span_attributes(span, "vectordb")
            span.set_attributes(extract_vectordb_attributes(instance, "upsert"))
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error("netra.instrumentation.agno: failed to record error for %s: %s", span_name, span_err)
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", span_name, e)

    return wrapper


def memory_add_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            span = tracer.start_span(MEMORY_ADD_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"})
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", MEMORY_ADD_SPAN, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", MEMORY_ADD_SPAN, e)

        try:
            _set_common_span_attributes(span, "memory")
            span.set_attributes(extract_memory_attributes(instance, args, "add_user_memory"))
        except Exception as e:
            logger.warning(
                "netra.instrumentation.agno: failed to set request attributes for %s: %s", MEMORY_ADD_SPAN, e
            )

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning(
                    "netra.instrumentation.agno: failed to set response attributes for %s: %s", MEMORY_ADD_SPAN, e
                )
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error("netra.instrumentation.agno: failed to record error for %s: %s", MEMORY_ADD_SPAN, span_err)
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", MEMORY_ADD_SPAN, e)

    return wrapper


def memory_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            span = tracer.start_span(
                MEMORY_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"}
            )
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", MEMORY_SEARCH_SPAN, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", MEMORY_SEARCH_SPAN, e)

        try:
            _set_common_span_attributes(span, "memory")
            span.set_attributes(extract_memory_attributes(instance, args, "search_user_memories"))
        except Exception as e:
            logger.warning(
                "netra.instrumentation.agno: failed to set request attributes for %s: %s", MEMORY_SEARCH_SPAN, e
            )

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning(
                    "netra.instrumentation.agno: failed to set response attributes for %s: %s", MEMORY_SEARCH_SPAN, e
                )
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error(
                    "netra.instrumentation.agno: failed to record error for %s: %s", MEMORY_SEARCH_SPAN, span_err
                )
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", MEMORY_SEARCH_SPAN, e)

    return wrapper


def knowledge_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        try:
            span = tracer.start_span(
                KNOWLEDGE_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "knowledge"}
            )
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to start span for %s: %s", KNOWLEDGE_SEARCH_SPAN, e)
            return wrapped(*args, **kwargs)

        ctx_token = None
        try:
            ctx_token = context_api.attach(set_span_in_context(span))
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", KNOWLEDGE_SEARCH_SPAN, e)

        try:
            _set_common_span_attributes(span, "knowledge")
            span.set_attributes(extract_knowledge_attributes(instance))
            if args:
                span.set_attribute("input", str(args[0]))
        except Exception as e:
            logger.warning(
                "netra.instrumentation.agno: failed to set request attributes for %s: %s", KNOWLEDGE_SEARCH_SPAN, e
            )

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning(
                    "netra.instrumentation.agno: failed to set response attributes for %s: %s", KNOWLEDGE_SEARCH_SPAN, e
                )
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception as span_err:
                logger.error(
                    "netra.instrumentation.agno: failed to record error for %s: %s", KNOWLEDGE_SEARCH_SPAN, span_err
                )
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end span for %s: %s", KNOWLEDGE_SEARCH_SPAN, e)

    return wrapper
