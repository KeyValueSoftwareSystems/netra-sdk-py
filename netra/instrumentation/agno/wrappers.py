import logging
import time
from collections.abc import Awaitable
from contextvars import ContextVar
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

# Prevent duplicate spans when an agent runs inside a team or workflow
_agno_team_active: ContextVar[bool] = ContextVar("_agno_team_active", default=False)
_agno_workflow_active: ContextVar[bool] = ContextVar("_agno_workflow_active", default=False)


def _get_span_name(instance: Any, prefix: str, default: Optional[str] = None) -> str:
    """Build a span name from an Agno instance name and a dot-separated prefix.

    Args:
        instance: An Agno object that may have a ``name`` attribute.
        prefix: The base span name (e.g. ``"agno.agent.run"``).
        default: Fallback used when ``instance.name`` is absent.

    Returns:
        ``"<prefix>.<name>"`` if a name is found, otherwise ``prefix`` alone.
    """
    name = getattr(instance, "name", None) or default or "unknown"
    return f"{prefix}.{name}" if name else prefix


def _set_common_span_attributes(span: Span, entity_type: str) -> None:
    """Set the three Agno span attributes shared by all entity types.

    Writes ``LLM_SYSTEM``, ``gen_ai.entity``, and ``netra.span.type``.

    Args:
        span: The active OpenTelemetry span.
        entity_type: Entity type string (e.g. ``"vectordb"``, ``"memory"``).
    """
    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))


def agent_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Agent.run`` (sync).

    Handles both streaming (``stream=True``) and non-streaming calls.
    Streaming responses are wrapped in :class:`AgnoStreamingWrapper` so the
    span stays open until the caller exhausts the iterator.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        if _agno_team_active.get() or _agno_workflow_active.get():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_RUN_SPAN)
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            try:
                span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"})
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
                set_request_attributes(span, instance, args, kwargs, "agent")
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
                    logger.error(
                        "netra.instrumentation.agno: failed to finalize span on %s error: %s",
                        span_name,
                        span_err,
                    )
                raise

            return AgnoStreamingWrapper(span=span, response=response, ctx_token=ctx_token)

        else:
            with tracer.start_as_current_span(
                span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"}
            ) as span:
                try:
                    set_request_attributes(span, instance, args, kwargs, "agent")
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

    return wrapper


def agent_arun_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Agent.arun`` (async).

    Uses a sync dispatcher so streaming (``stream=True``) returns an async
    generator directly rather than a coroutine, matching Agno's own behaviour.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        if _agno_team_active.get() or _agno_workflow_active.get():
            return wrapped(*args, **kwargs)

        if kwargs.get("stream", False):
            return _agent_arun_stream(tracer, wrapped, instance, args, kwargs)
        return _agent_arun_non_stream(tracer, wrapped, instance, args, kwargs)

    return wrapper


async def _agent_arun_non_stream(
    tracer: Tracer,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Execute async non-streaming ``Agent.arun`` within a single span.

    Args:
        tracer: The OpenTelemetry tracer.
        wrapped: The original ``Agent.arun`` coroutine callable.
        instance: The Agno Agent object.
        args: Positional arguments forwarded to the original call.
        kwargs: Keyword arguments forwarded to the original call.

    Returns:
        The RunResponse returned by ``Agent.arun``.
    """
    span_name = _get_span_name(instance, AGENT_RUN_SPAN)

    with tracer.start_as_current_span(
        span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"}
    ) as span:
        try:
            set_request_attributes(span, instance, args, kwargs, "agent")
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
            set_response_attributes(span, response)
            record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
            span.set_status(Status(StatusCode.OK))
            return response
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def _agent_arun_stream(
    tracer: Tracer,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> AsyncIterator[Any]:
    """Execute async streaming ``Agent.arun``, yielding events and finalizing the span.

    Accumulates streamed content chunks; the last event's metrics are used for
    token usage. The span is ended in the ``finally`` block regardless of error.

    Args:
        tracer: The OpenTelemetry tracer.
        wrapped: The original ``Agent.arun`` async-generator callable.
        instance: The Agno Agent object.
        args: Positional arguments forwarded to the original call.
        kwargs: Keyword arguments forwarded to the original call.

    Yields:
        Each streaming event from the underlying ``Agent.arun`` call.
    """
    span_name = _get_span_name(instance, AGENT_RUN_SPAN)
    span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"})
    token = context_api.attach(set_span_in_context(span))

    try:
        set_request_attributes(span, instance, args, kwargs, "agent")
    except Exception as e:
        logger.warning("netra.instrumentation.agno: failed to set request attributes for %s: %s", span_name, e)

    last_response: Any = None
    content_chunks: List[str] = []
    errored = False

    try:
        async for event in wrapped(*args, **kwargs):
            last_response = event
            try:
                content = getattr(event, "content", None)
                if content:
                    content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate stream content: %s", e)
            yield event
    except Exception as e:
        errored = True
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        if last_response is not None:
            try:
                set_response_attributes(span, last_response)
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set response attributes for %s: %s", span_name, e)
        # Streamed chunks give a more complete output than last_response.content alone
        if content_chunks:
            try:
                span.set_attribute("output", "".join(content_chunks))
            except Exception as e:
                logger.warning("netra.instrumentation.agno: failed to set accumulated output for %s: %s", span_name, e)
        record_span_timing(span, LLM_RESPONSE_DURATION)
        if not errored:
            span.set_status(Status(StatusCode.OK))
        context_api.detach(token)
        span.end()


def agent_continue_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Agent.continue_run`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_CONTINUE_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "agent")
                response = wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def agent_acontinue_run_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Return a wrapt-compatible wrapper for ``Agent.acontinue_run`` (async).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, AGENT_CONTINUE_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "agent")
                response = await wrapped(*args, **kwargs)
                end_time = time.time()
                set_response_attributes(span, response)
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def team_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Team.run`` (sync).

    Sets ``_agno_team_active`` while running so that nested ``Agent.run``
    calls within the team do not create duplicate spans.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, TEAM_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "team"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "team")
                cv_token = _agno_team_active.set(True)
                try:
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                finally:
                    _agno_team_active.reset(cv_token)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def team_arun_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Return a wrapt-compatible wrapper for ``Team.arun`` (async).

    Sets ``_agno_team_active`` while running so that nested ``Agent.arun``
    calls within the team do not create duplicate spans.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, TEAM_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "team"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "team")
                cv_token = _agno_team_active.set(True)
                try:
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                finally:
                    _agno_team_active.reset(cv_token)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def workflow_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Workflow.run_workflow`` (sync).

    Sets ``_agno_workflow_active`` so nested agent runs don't produce duplicate
    spans.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, WORKFLOW_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "workflow"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "workflow")
                cv_token = _agno_workflow_active.set(True)
                try:
                    response = wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                finally:
                    _agno_workflow_active.reset(cv_token)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def workflow_arun_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Return a wrapt-compatible wrapper for ``Workflow.arun_workflow`` (async).

    Sets ``_agno_workflow_active`` so nested agent runs don't produce duplicate
    spans.

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, WORKFLOW_RUN_SPAN)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "workflow"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "workflow")
                cv_token = _agno_workflow_active.set(True)
                try:
                    response = await wrapped(*args, **kwargs)
                    end_time = time.time()
                    set_response_attributes(span, response)
                    record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                    span.set_status(Status(StatusCode.OK))
                    return response
                finally:
                    _agno_workflow_active.reset(cv_token)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def tool_execute_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``FunctionCall.execute`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        tool_name = get_tool_name(instance)

        with tracer.start_as_current_span(
            tool_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "tool"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "tool")
                arguments = get_tool_arguments(instance, kwargs)
                if arguments:
                    span.set_attribute("input", arguments)

                response = wrapped(*args, **kwargs)
                end_time = time.time()

                if response is not None:
                    span.set_attribute("output", serialize_value(response, clean=True))

                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def tool_aexecute_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Return a wrapt-compatible wrapper for ``FunctionCall.aexecute`` (async).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        tool_name = get_tool_name(instance)

        with tracer.start_as_current_span(
            tool_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "tool"}
        ) as span:
            try:
                set_request_attributes(span, instance, args, kwargs, "tool")
                arguments = get_tool_arguments(instance, kwargs)
                if arguments:
                    span.set_attribute("input", arguments)

                response = await wrapped(*args, **kwargs)
                end_time = time.time()

                if response is not None:
                    span.set_attribute("output", serialize_value(response, clean=True))

                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def vectordb_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``VectorDb.search`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, VECTORDB_SEARCH_SPAN, type(instance).__name__)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"}
        ) as span:
            try:
                _set_common_span_attributes(span, "vectordb")
                span.set_attributes(extract_vectordb_attributes(instance, "search"))
                if args:
                    try:
                        span.set_attribute("input", str(args[0]))
                    except Exception as e:
                        logger.debug("netra.instrumentation.agno: failed to set input for vectordb search: %s", e)

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def vectordb_upsert_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``VectorDb.upsert`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = _get_span_name(instance, VECTORDB_UPSERT_SPAN, type(instance).__name__)

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"}
        ) as span:
            try:
                _set_common_span_attributes(span, "vectordb")
                span.set_attributes(extract_vectordb_attributes(instance, "upsert"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def memory_add_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Memory.add_user_memory`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            MEMORY_ADD_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"}
        ) as span:
            try:
                _set_common_span_attributes(span, "memory")
                span.set_attributes(extract_memory_attributes(instance, args, "add_user_memory"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def memory_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``Memory.search_user_memories`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            MEMORY_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"}
        ) as span:
            try:
                _set_common_span_attributes(span, "memory")
                span.set_attributes(extract_memory_attributes(instance, args, "search_user_memories"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def knowledge_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt-compatible wrapper for ``AgentKnowledge.search`` (sync).

    Args:
        tracer: The OpenTelemetry tracer to use for span creation.

    Returns:
        A wrapper function compatible with ``wrapt.wrap_function_wrapper``.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            KNOWLEDGE_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "knowledge"}
        ) as span:
            try:
                _set_common_span_attributes(span, "knowledge")
                span.set_attributes(extract_knowledge_attributes(instance))
                if args:
                    try:
                        span.set_attribute("input", str(args[0]))
                    except Exception as e:
                        logger.debug("netra.instrumentation.agno: failed to set input for knowledge search: %s", e)

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


class AgnoStreamingWrapper:
    """Wraps a synchronous streaming Agno response to keep the OTel span open.

    The span is finalized (attributes set, status written, context detached,
    span ended) when iteration completes via ``StopIteration`` or an unhandled
    exception occurs during ``__next__``.
    """

    def __init__(self, span: Span, response: Iterator[Any], ctx_token: Any = None) -> None:
        """
        Args:
            span: The open OTel span created before streaming started.
            response: The raw streaming iterator from ``Agent.run``.
            ctx_token: OTel context token from ``context_api.attach``; detached on finalization.
        """
        self._span = span
        self._response = response
        self._last_response: Any = None
        self._ctx_token = ctx_token
        self._content_chunks: List[str] = []
        self._finalized = False

    def __enter__(self) -> "AgnoStreamingWrapper":
        """Delegate context-manager entry to the underlying response if supported."""
        if hasattr(self._response, "__enter__"):
            try:
                self._response.__enter__()
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in stream __enter__: %s", e)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Delegate context-manager exit; finalize span with error status on exception."""
        if hasattr(self._response, "__exit__"):
            try:
                self._response.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.debug("netra.instrumentation.agno: error in stream __exit__: %s", e)
        if exc_type is not None and not self._finalized:
            try:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self._span.record_exception(exc_val)
                if self._ctx_token is not None:
                    context_api.detach(self._ctx_token)
                self._span.end()
            except Exception as e:
                logger.error("netra.instrumentation.agno: failed to finalize span on stream exit error: %s", e)
            self._finalized = True

    def __iter__(self) -> "AgnoStreamingWrapper":
        return self

    def __next__(self) -> Any:
        """Yield the next event; finalize the span on ``StopIteration`` or error."""
        try:
            event = self._response.__next__()
            self._last_response = event
            try:
                content = getattr(event, "content", None)
                if content:
                    self._content_chunks.append(str(content))
            except Exception as e:
                logger.debug("netra.instrumentation.agno: failed to accumulate stream content: %s", e)
            return event
        except StopIteration:
            self._finalize_span()
            raise
        except Exception as e:
            if not self._finalized:
                try:
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._span.record_exception(e)
                    if self._ctx_token is not None:
                        context_api.detach(self._ctx_token)
                    self._span.end()
                except Exception as span_err:
                    logger.error(
                        "netra.instrumentation.agno: failed to finalize span on stream iteration error: %s",
                        span_err,
                    )
                self._finalized = True
            raise

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying response object."""
        return getattr(self._response, name)

    def _finalize_span(self) -> None:
        """Set response attributes, record timing, and end the span exactly once."""
        if self._finalized:
            return
        self._finalized = True
        try:
            if self._last_response is not None:
                set_response_attributes(self._span, self._last_response)
            # Streamed chunks give a more complete output than last_response.content alone
            if self._content_chunks:
                self._span.set_attribute("output", "".join(self._content_chunks))
        except Exception as e:
            logger.warning("netra.instrumentation.agno: failed to set response attributes on stream end: %s", e)
        try:
            record_span_timing(self._span, LLM_RESPONSE_DURATION)
            self._span.set_status(Status(StatusCode.OK))
            if self._ctx_token is not None:
                context_api.detach(self._ctx_token)
            self._span.end()
        except Exception as e:
            logger.error("netra.instrumentation.agno: failed to end span after streaming: %s", e)
