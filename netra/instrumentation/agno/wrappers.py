import json
import logging
import time
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Dict, Iterator, Tuple

from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.agno.utils import (
    ATTR_ENTITY,
    LLM_SYSTEM_AGNO,
    NETRA_SPAN_TYPE,
    _ENTITY_SPAN_TYPE_MAP,
    extract_knowledge_attributes,
    extract_memory_attributes,
    extract_vectordb_attributes,
    get_tool_arguments,
    get_tool_name,
    set_request_attributes,
    set_response_attributes,
    should_suppress_instrumentation,
)
from netra.span_wrapper import SpanType
from netra.instrumentation.utils import record_span_timing

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

# Timing constants
LLM_RESPONSE_DURATION = "llm.response.duration"

_agno_team_active: ContextVar[bool] = ContextVar("_agno_team_active", default=False)
_agno_workflow_active: ContextVar[bool] = ContextVar("_agno_workflow_active", default=False)


def _set_common_attributes(span: Span, entity_type: str) -> None:
    """Set common Agno span attributes shared across all entity types."""
    span.set_attribute(SpanAttributes.LLM_SYSTEM, LLM_SYSTEM_AGNO)
    span.set_attribute(ATTR_ENTITY, entity_type)
    span.set_attribute(NETRA_SPAN_TYPE, _ENTITY_SPAN_TYPE_MAP.get(entity_type, SpanType.SPAN))


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------


def agent_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.run (sync), handles both streaming and non-streaming."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        if _agno_team_active.get() or _agno_workflow_active.get():
            return wrapped(*args, **kwargs)

        agent_name = getattr(instance, "name", "unknown")
        span_name = f"{AGENT_RUN_SPAN}.{agent_name}"
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"})
            try:
                context = context_api.attach(set_span_in_context(span))
                set_request_attributes(span, instance, args, kwargs, "agent")
                response = wrapped(*args, **kwargs)
                return AgnoStreamingWrapper(span=span, response=response)
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.end()
                raise
            finally:
                context_api.detach(context)
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
                    logger.error("netra.instrumentation.agno: %s", e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

    return wrapper


def agent_arun_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.arun (async), handles both streaming and non-streaming.

    Uses a sync dispatcher because Agno's Agent.arun(stream=True) returns an
    async generator directly (not a coroutine), so the wrapper must return an
    async iterable rather than a coroutine for the streaming path.
    """

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        if _agno_team_active.get() or _agno_workflow_active.get():
            return wrapped(*args, **kwargs)

        is_streaming = kwargs.get("stream", False)
        if is_streaming:
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
    """Execute async non-streaming Agent.arun within a span."""
    agent_name = getattr(instance, "name", "unknown")
    span_name = f"{AGENT_RUN_SPAN}.{agent_name}"

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
            logger.error("netra.instrumentation.agno: %s", e)
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
    """Wrap an async streaming Agent.arun, ending the span when iteration completes."""
    agent_name = getattr(instance, "name", "unknown")
    span_name = f"{AGENT_RUN_SPAN}.{agent_name}"

    span = tracer.start_span(span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "agent"})
    token = context_api.attach(set_span_in_context(span))
    try:
        set_request_attributes(span, instance, args, kwargs, "agent")
    except Exception as e:
        logger.error("netra.instrumentation.agno: %s", e)

    last_response: Any = None
    errored = False
    try:
        async for event in wrapped(*args, **kwargs):
            last_response = event
            yield event
    except Exception as e:
        errored = True
        logger.error("netra.instrumentation.agno: %s", e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        if last_response is not None:
            try:
                set_response_attributes(span, last_response)
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
        record_span_timing(span, LLM_RESPONSE_DURATION)
        if not errored:
            span.set_status(Status(StatusCode.OK))
        context_api.detach(token)
        span.end()


def agent_continue_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Agent.continue_run (sync)."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        agent_name = getattr(instance, "name", "unknown")
        span_name = f"{AGENT_CONTINUE_RUN_SPAN}.{agent_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def agent_acontinue_run_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for Agent.acontinue_run."""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        agent_name = getattr(instance, "name", "unknown")
        span_name = f"{AGENT_CONTINUE_RUN_SPAN}.{agent_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Team wrappers
# ---------------------------------------------------------------------------


def team_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Team.run (sync).

    Sets the _agno_team_active ContextVar so nested agent runs don't create
    duplicate spans.
    """

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        team_name = getattr(instance, "name", "unknown")
        span_name = f"{TEAM_RUN_SPAN}.{team_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def team_arun_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for Team.arun."""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        team_name = getattr(instance, "name", "unknown")
        span_name = f"{TEAM_RUN_SPAN}.{team_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Workflow wrappers
# ---------------------------------------------------------------------------


def workflow_run_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Workflow.run_workflow (sync)."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "unknown")
        span_name = f"{WORKFLOW_RUN_SPAN}.{workflow_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def workflow_arun_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for Workflow.arun_workflow."""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "unknown")
        span_name = f"{WORKFLOW_RUN_SPAN}.{workflow_name}"

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
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


def tool_execute_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for FunctionCall.execute (sync)."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
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
                    try:
                        span.set_attribute(
                            "output", json.dumps(response) if isinstance(response, dict) else str(response)
                        )
                    except Exception:
                        span.set_attribute("output", str(response))

                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def tool_aexecute_wrapper(tracer: Tracer) -> Callable[..., Awaitable[Any]]:
    """Async wrapper for FunctionCall.aexecute."""

    async def wrapper(
        wrapped: Callable[..., Awaitable[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
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
                    try:
                        span.set_attribute(
                            "output", json.dumps(response) if isinstance(response, dict) else str(response)
                        )
                    except Exception:
                        span.set_attribute("output", str(response))

                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# VectorDB wrappers
# ---------------------------------------------------------------------------


def vectordb_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for VectorDb.search (sync)."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        db_name = getattr(instance, "name", None) or type(instance).__name__
        span_name = f"{VECTORDB_SEARCH_SPAN}.{db_name}"

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"}
        ) as span:
            try:
                _set_common_attributes(span, "vectordb")
                span.set_attributes(extract_vectordb_attributes(instance, "search"))
                if args:
                    try:
                        span.set_attribute("input", str(args[0]))
                    except Exception:
                        pass

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def vectordb_upsert_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for VectorDb.upsert (sync)."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        db_name = getattr(instance, "name", None) or type(instance).__name__
        span_name = f"{VECTORDB_UPSERT_SPAN}.{db_name}"

        with tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes={"llm.request.type": "vectordb"}
        ) as span:
            try:
                _set_common_attributes(span, "vectordb")
                span.set_attributes(extract_vectordb_attributes(instance, "upsert"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Memory wrappers
# ---------------------------------------------------------------------------


def memory_add_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Memory.add_user_memory or MemoryManager equivalent."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            MEMORY_ADD_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"}
        ) as span:
            try:
                _set_common_attributes(span, "memory")
                span.set_attributes(extract_memory_attributes(instance, args, "add_user_memory"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def memory_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for Memory.search_user_memories or MemoryManager equivalent."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            MEMORY_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "memory"}
        ) as span:
            try:
                _set_common_attributes(span, "memory")
                span.set_attributes(extract_memory_attributes(instance, args, "search_user_memories"))

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Knowledge wrappers
# ---------------------------------------------------------------------------


def knowledge_search_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Wrapper for AgentKnowledge.search / Knowledge.search."""

    def wrapper(
        wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        with tracer.start_as_current_span(
            KNOWLEDGE_SEARCH_SPAN, kind=SpanKind.CLIENT, attributes={"llm.request.type": "knowledge"}
        ) as span:
            try:
                _set_common_attributes(span, "knowledge")
                span.set_attributes(extract_knowledge_attributes(instance))
                if args:
                    try:
                        span.set_attribute("input", str(args[0]))
                    except Exception:
                        pass

                response = wrapped(*args, **kwargs)
                end_time = time.time()
                record_span_timing(span, LLM_RESPONSE_DURATION, end_time)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Streaming wrapper classes
# ---------------------------------------------------------------------------


class AgnoStreamingWrapper:
    """Wrapper for synchronous streaming Agno responses.

    Follows the same pattern as OpenAI's StreamingWrapper: wraps the response
    iterator, processes events, and finalizes the span when iteration completes.
    """

    def __init__(self, span: Span, response: Iterator[Any]) -> None:
        self._span = span
        self._response = response
        self._last_response: Any = None

    def __enter__(self) -> "AgnoStreamingWrapper":
        if hasattr(self._response, "__enter__"):
            self._response.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._response, "__exit__"):
            self._response.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self._span.record_exception(exc_val)
            self._span.end()

    def __iter__(self) -> "AgnoStreamingWrapper":
        return self

    def __next__(self) -> Any:
        try:
            event = self._response.__next__()
            self._last_response = event
            return event
        except StopIteration:
            self._finalize_span()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete."""
        if self._last_response is not None:
            try:
                set_response_attributes(self._span, self._last_response)
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class AgnoAsyncStreamingWrapper:
    """Wrapper for asynchronous streaming Agno responses.

    Follows the same pattern as OpenAI's AsyncStreamingWrapper: wraps the async
    response iterator, processes events, and finalizes the span when iteration
    completes.
    """

    def __init__(self, span: Span, response: AsyncIterator[Any]) -> None:
        self._span = span
        self._response = response
        self._last_response: Any = None

    async def __aenter__(self) -> "AgnoAsyncStreamingWrapper":
        if hasattr(self._response, "__aenter__"):
            await self._response.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._response, "__aexit__"):
            await self._response.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self._span.record_exception(exc_val)
            self._span.end()

    def __aiter__(self) -> "AgnoAsyncStreamingWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._response.__anext__()
            self._last_response = event
            return event
        except StopAsyncIteration:
            self._finalize_span()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def _finalize_span(self) -> None:
        """Finalize span when streaming is complete."""
        if self._last_response is not None:
            try:
                set_response_attributes(self._span, self._last_response)
            except Exception as e:
                logger.error("netra.instrumentation.agno: %s", e)
        record_span_timing(self._span, LLM_RESPONSE_DURATION)
        self._span.set_status(Status(StatusCode.OK))
        self._span.end()
