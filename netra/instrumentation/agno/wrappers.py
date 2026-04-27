import json
import logging
import time
from collections.abc import Awaitable
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    format_messages_as_input,
    format_response_as_output,
    get_tool_arguments,
    get_tool_name,
    is_assistant_response,
    is_run_content,
    serialize_value,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
    update_active_span_with_system_prompt,
)
from netra.instrumentation.utils import record_span_timing
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)

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
LLM_CALL_SPAN = "agno.llm.call"
LLM_RESPONSE_DURATION = "llm.response.duration"


def _start_span(
    tracer: Tracer,
    span_name: str,
    request_type: str,
) -> Tuple[Optional[Span], Any]:
    """Start a span and attach it to the current context.

    Returns (span, ctx_token) on success, or (None, None) if either step fails.
    """
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": request_type})
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start span for %s: %s", span_name, e)
        return None, None
    try:
        ctx_token = context_api.attach(set_span_in_context(span))
        return span, ctx_token
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach context for %s: %s", span_name, e)
        try:
            span.end()
        except Exception:
            pass
        return None, None


def _close_span(span: Span, ctx_token: Any, error: Optional[Exception] = None) -> None:
    """Record timing, set status, detach context, end span."""
    try:
        record_span_timing(span, LLM_RESPONSE_DURATION)
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to set span status: %s", e)
    try:
        if ctx_token is not None:
            context_api.detach(ctx_token)
    except Exception as e:
        logger.debug("netra.instrumentation.agno: failed to detach context: %s", e)
    try:
        span.end()
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to end span: %s", e)


def _get_span_name(instance: Any, prefix: str, default: Optional[str] = None) -> str:
    name = getattr(instance, "name", None) or default or "unknown"
    return f"{prefix}.{name}" if name else prefix


def _set_common_span_attributes(span: Span, entity_type: str) -> None:
    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))


class _BaseStreamWrapper:
    """Shared base for all span streaming wrappers."""

    def __init__(self, span: Span, response: Any, ctx_token: Any = None) -> None:
        self._span = span
        self._response = response
        self._ctx_token = ctx_token
        self._content_chunks: List[str] = []
        self._last_response: Any = None
        self._finalized = False

    def _set_output_on_success(self) -> None:
        """Override to write output attributes before the span closes."""

    def _finalize(self, error: Optional[Exception] = None) -> None:
        if self._finalized:
            return
        self._finalized = True
        if error is None:
            try:
                self._set_output_on_success()
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set output attrs on stream end: %s", e)
        _close_span(self._span, self._ctx_token, error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def __del__(self) -> None:
        try:
            if not self._finalized:
                self._finalize()
        except Exception:
            pass


class _AgentStreamOutputMixin:
    """Output strategy for agent/team/workflow streams."""

    _last_response: Any
    _span: Span
    _content_chunks: List[str]

    def _set_output_on_success(self) -> None:
        if self._last_response is not None:
            set_response_attributes(self._span, self._last_response)
        if self._content_chunks:
            self._span.set_attribute("output", "".join(self._content_chunks))


class _LlmStreamOutputMixin:
    """Output strategy for LLM model streams."""

    _content_chunks: List[str]
    _span: Span

    def _set_output_on_success(self) -> None:
        if self._content_chunks:
            content = "".join(self._content_chunks)
            self._span.set_attribute("output", json.dumps([{"role": "assistant", "content": content}]))


class SpanStreamingWrapper(_AgentStreamOutputMixin, _BaseStreamWrapper):
    """Sync streaming wrapper for agent/team/workflow spans."""

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


class AsyncSpanStreamingWrapper(_AgentStreamOutputMixin, _BaseStreamWrapper):
    """Async streaming wrapper for agent/team/workflow spans."""

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


class LlmSpanStreamingWrapper(_LlmStreamOutputMixin, _BaseStreamWrapper):
    """Sync streaming wrapper for LLM model spans."""

    def __iter__(self) -> "LlmSpanStreamingWrapper":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._response)
            try:
                if is_assistant_response(chunk):
                    content = getattr(chunk, "content", None)
                    if content:
                        self._content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate llm stream content: %s", e)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise


class AsyncLlmSpanStreamingWrapper(_LlmStreamOutputMixin, _BaseStreamWrapper):
    """Async streaming wrapper for LLM model spans."""

    def __aiter__(self) -> "AsyncLlmSpanStreamingWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._response.__anext__()
            try:
                if is_assistant_response(chunk):
                    content = getattr(chunk, "content", None)
                    if content:
                        self._content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate async llm stream content: %s", e)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize(error=e)
            raise


def _sync_non_stream(
    tracer: Tracer,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    span, ctx_token = _start_span(tracer, span_name, request_type)
    if span is None:
        return wrapped(*args, **kwargs)

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
    span, ctx_token = _start_span(tracer, span_name, request_type)
    if span is None:
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

    return SpanStreamingWrapper(span=span, response=response, ctx_token=ctx_token)


async def _async_non_stream(
    tracer: Tracer,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    span_name: str,
    request_type: str,
) -> Any:
    span, ctx_token = _start_span(tracer, span_name, request_type)
    if span is None:
        return await wrapped(*args, **kwargs)

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
    span, ctx_token = _start_span(tracer, span_name, request_type)
    if span is None:
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

    return AsyncSpanStreamingWrapper(span=span, response=response, ctx_token=ctx_token)


def _make_run_wrapper(span_prefix: str, entity_type: str, is_async: bool = False) -> Callable[..., Any]:
    """Factory producing run/arun wrappers for agent, team, and workflow."""

    def outer(tracer: Tracer) -> Callable[..., Any]:
        stream_fn = _async_stream_start if is_async else _sync_stream_start
        nonstream_fn = _async_non_stream if is_async else _sync_non_stream

        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            if should_suppress_instrumentation():
                return wrapped(*args, **kwargs)
            span_name = _get_span_name(instance, span_prefix)
            if kwargs.get("stream"):
                return stream_fn(tracer, wrapped, instance, args, kwargs, span_name, entity_type)
            return nonstream_fn(tracer, wrapped, instance, args, kwargs, span_name, entity_type)

        return wrapper

    return outer


agent_run_wrapper = _make_run_wrapper(AGENT_RUN_SPAN, "agent")
agent_arun_wrapper = _make_run_wrapper(AGENT_RUN_SPAN, "agent", is_async=True)
team_run_wrapper = _make_run_wrapper(TEAM_RUN_SPAN, "team")
team_arun_wrapper = _make_run_wrapper(TEAM_RUN_SPAN, "team", is_async=True)
workflow_run_wrapper = _make_run_wrapper(WORKFLOW_RUN_SPAN, "workflow")
workflow_arun_wrapper = _make_run_wrapper(WORKFLOW_RUN_SPAN, "workflow", is_async=True)


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


def _set_tool_span_attrs(
    span: Span,
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    tool_name: str,
) -> None:
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
        span, ctx_token = _start_span(tracer, tool_name, "tool")
        if span is None:
            return wrapped(*args, **kwargs)

        _set_tool_span_attrs(span, instance, args, kwargs, tool_name)

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
        span, ctx_token = _start_span(tracer, tool_name, "tool")
        if span is None:
            return await wrapped(*args, **kwargs)

        _set_tool_span_attrs(span, instance, args, kwargs, tool_name)

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


def _make_simple_sync_wrapper(
    get_span_name_fn: Callable[[Any], str],
    request_type: str,
    set_attrs_fn: Callable[[Span, Any, Tuple[Any, ...], Dict[str, Any]], None],
) -> Callable[..., Any]:
    """Factory for non-streaming span wrappers with custom attribute setup."""

    def outer(tracer: Tracer) -> Callable[..., Any]:
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            if should_suppress_instrumentation():
                return wrapped(*args, **kwargs)

            span_name = get_span_name_fn(instance)
            span, ctx_token = _start_span(tracer, span_name, request_type)
            if span is None:
                return wrapped(*args, **kwargs)

            try:
                set_attrs_fn(span, instance, args, kwargs)
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

            try:
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                try:
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    logger.warning(
                        "netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e
                    )
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

    return outer


def _vectordb_search_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    _set_common_span_attributes(span, "vectordb")
    span.set_attributes(extract_vectordb_attributes(instance, "search"))
    if args:
        span.set_attribute("input", str(args[0]))


def _vectordb_upsert_attrs(span: Span, instance: Any, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]) -> None:
    _set_common_span_attributes(span, "vectordb")
    span.set_attributes(extract_vectordb_attributes(instance, "upsert"))


def _memory_add_attrs(span: Span, instance: Any, args: Tuple[Any, ...], _kwargs: Dict[str, Any]) -> None:
    _set_common_span_attributes(span, "memory")
    span.set_attributes(extract_memory_attributes(instance, args, "add_user_memory"))


def _memory_search_attrs(span: Span, instance: Any, args: Tuple[Any, ...], _kwargs: Dict[str, Any]) -> None:
    _set_common_span_attributes(span, "memory")
    span.set_attributes(extract_memory_attributes(instance, args, "search_user_memories"))


def _knowledge_search_attrs(span: Span, instance: Any, args: Tuple[Any, ...], _kwargs: Dict[str, Any]) -> None:
    _set_common_span_attributes(span, "knowledge")
    span.set_attributes(extract_knowledge_attributes(instance))
    if args:
        span.set_attribute("input", str(args[0]))


vectordb_search_wrapper = _make_simple_sync_wrapper(
    get_span_name_fn=lambda inst: _get_span_name(inst, VECTORDB_SEARCH_SPAN, type(inst).__name__),
    request_type="vectordb",
    set_attrs_fn=_vectordb_search_attrs,
)

vectordb_upsert_wrapper = _make_simple_sync_wrapper(
    get_span_name_fn=lambda inst: _get_span_name(inst, VECTORDB_UPSERT_SPAN, type(inst).__name__),
    request_type="vectordb",
    set_attrs_fn=_vectordb_upsert_attrs,
)

memory_add_wrapper = _make_simple_sync_wrapper(
    get_span_name_fn=lambda _: MEMORY_ADD_SPAN,
    request_type="memory",
    set_attrs_fn=_memory_add_attrs,
)

memory_search_wrapper = _make_simple_sync_wrapper(
    get_span_name_fn=lambda _: MEMORY_SEARCH_SPAN,
    request_type="memory",
    set_attrs_fn=_memory_search_attrs,
)

knowledge_search_wrapper = _make_simple_sync_wrapper(
    get_span_name_fn=lambda _: KNOWLEDGE_SEARCH_SPAN,
    request_type="knowledge",
    set_attrs_fn=_knowledge_search_attrs,
)


def _start_llm_span(tracer: Tracer, instance: Any) -> Optional[Span]:
    model_id = getattr(instance, "id", None) or getattr(instance, "name", None) or type(instance).__name__
    span_name = f"{LLM_CALL_SPAN}.{model_id}" if model_id else LLM_CALL_SPAN
    try:
        span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "llm"})
        span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
        span.set_attribute(ATTR_ENTITY, "llm")
        span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get("llm", SpanType.SPAN))
        return span
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to start LLM span: %s", e)
        return None


def _setup_llm_span_with_input(
    tracer: Tracer,
    instance: Any,
    messages: Any,
) -> Tuple[Optional[Span], Any]:
    """Start LLM span, update system prompt on parent span, attach context, set input."""
    if messages:
        update_active_span_with_system_prompt(messages)

    span = _start_llm_span(tracer, instance)
    if span is None:
        return None, None

    ctx_token = None
    try:
        ctx_token = context_api.attach(set_span_in_context(span))
    except Exception as e:
        logger.error("netra.instrumentation.agno: failed to attach LLM span context: %s", e)

    if messages:
        input_str = format_messages_as_input(messages)
        if input_str:
            span.set_attribute("input", input_str)

    return span, ctx_token


def model_response_capture_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Model.response — creates a child LLM span."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        messages = args[0] if args else kwargs.get("messages")
        span, ctx_token = _setup_llm_span_with_input(tracer, instance, messages)
        if span is None:
            return wrapped(*args, **kwargs)

        try:
            response = wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                output_str = format_response_as_output(response)
                if output_str:
                    span.set_attribute("output", output_str)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set LLM response attributes: %s", e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception:
                pass
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end LLM span: %s", e)

    return wrapper


def model_response_stream_capture_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Model.response_stream — LLM span closes when the stream ends."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        messages = args[0] if args else kwargs.get("messages")
        span, ctx_token = _setup_llm_span_with_input(tracer, instance, messages)
        if span is None:
            return wrapped(*args, **kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception:
                pass
            raise

        return LlmSpanStreamingWrapper(span=span, response=response, ctx_token=ctx_token)

    return wrapper


def model_aresponse_capture_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Model.aresponse — creates a child LLM span (async)."""

    async def _capture(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        messages = args[0] if args else kwargs.get("messages")
        span, ctx_token = _setup_llm_span_with_input(tracer, instance, messages)
        if span is None:
            return await wrapped(*args, **kwargs)

        try:
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
            try:
                output_str = format_response_as_output(response)
                if output_str:
                    span.set_attribute("output", output_str)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set async LLM response attributes: %s", e)
            return response
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            except Exception:
                pass
            raise
        finally:
            try:
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to end async LLM span: %s", e)

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        return _capture(wrapped, instance, args, kwargs)

    return wrapper


def model_aresponse_stream_capture_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Model.aresponse_stream — async LLM span closes when stream ends."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        messages = args[0] if args else kwargs.get("messages")
        span, ctx_token = _setup_llm_span_with_input(tracer, instance, messages)
        if span is None:
            return wrapped(*args, **kwargs)

        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            try:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()
            except Exception:
                pass
            raise

        return AsyncLlmSpanStreamingWrapper(span=span, response=response, ctx_token=ctx_token)

    return wrapper
