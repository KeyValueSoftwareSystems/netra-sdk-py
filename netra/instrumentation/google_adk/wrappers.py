import json
import logging
import time
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable, Dict, Generator, List, Tuple, cast

from opentelemetry import context as opentelemetry_context
from opentelemetry import trace as opentelemetry_api_trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer

from netra.config import Config
from netra.instrumentation.google_adk.utils import (
    NETRA_SPAN_TYPE,
    build_llm_request_for_trace,
    extract_agent_attributes,
    extract_llm_request_attributes,
    extract_llm_response_attributes,
)
from netra.instrumentation.utils import record_span_timing
from netra.span_wrapper import SpanType

TIME_TO_FIRST_TOKEN = "gen_ai.performance.time_to_first_token"
RELATIVE_TIME_TO_FIRST_TOKEN = "gen_ai.performance.relative_time_to_first_token"

logger = logging.getLogger(__name__)


class NoOpSpan:
    """Span implementation that silently discards all operations, used to suppress ADK's built-in tracing."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Initialize the NoOpSpan, discarding all arguments."""

    def __enter__(self) -> "NoOpSpan":
        """Enter the context manager.

        Returns: The NoOpSpan instance.
        """
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit the context manager, discarding all arguments."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        """Set a span attribute (no-op), discarding all arguments."""

    def set_attributes(self, *_args: Any, **_kwargs: Any) -> None:
        """Set multiple span attributes (no-op), discarding all arguments."""

    def add_event(self, *_args: Any, **_kwargs: Any) -> None:
        """Add a span event (no-op), discarding all arguments."""

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        """Set the span status (no-op), discarding all arguments."""

    def update_name(self, *_args: Any, **_kwargs: Any) -> None:
        """Update the span name (no-op), discarding all arguments."""

    def is_recording(self) -> bool:
        """Check whether the span is recording.

        Returns: Always False for a NoOpSpan.
        """
        return False

    def end(self, *_args: Any, **_kwargs: Any) -> None:
        """End the span (no-op), discarding all arguments."""

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        """Record an exception on the span (no-op), discarding all arguments."""


class NoOpTracer:
    """Tracer that suppresses ADK's own span emission to avoid duplicates with Netra spans."""

    @contextmanager
    def start_as_current_span(self, *_args: Any, **_kwargs: Any) -> Generator[Span, None, None]:
        """Yield the current real span without creating a new one or modifying context.

        ADK captures the yielded span and later calls ``trace.use_span(span)`` to rebind
        context for after-model callbacks.  Returning a NoOpSpan here would corrupt that
        context (the NoOpSpan carries no trace info), so we yield the live span instead,
        which lets ``trace.use_span`` restore the correct parent context.
        """
        yield opentelemetry_api_trace.get_current_span()

    def start_span(self, *_args: Any, **_kwargs: Any) -> NoOpSpan:
        """Start a new span (no-op).

        Returns: A NoOpSpan instance.
        """
        return NoOpSpan()

    def use_span(self, *_args: Any, **_kwargs: Any) -> NoOpSpan:
        """Use an existing span as context (no-op).

        Returns: A NoOpSpan instance.
        """
        return NoOpSpan()


class _SpanScope:
    """Manages span lifecycle for async generators (start, attach context, record errors, detach, end)."""

    def __init__(self, tracer: Tracer, name: str, kind: SpanKind = SpanKind.CLIENT) -> None:
        """Start a span and attach it as the current context.

        Args:
            tracer: The OpenTelemetry tracer used to start the span.
            name: The span name.
            kind: The span kind; defaults to SpanKind.CLIENT.
        """
        self.span = tracer.start_span(name, kind=kind)
        ctx = opentelemetry_api_trace.set_span_in_context(self.span)
        self._token = opentelemetry_context.attach(ctx)

    def record_error(self, exc: Exception) -> None:
        """Record an exception on the span and set its status to ERROR.

        Args:
            exc: The exception to record.
        """
        try:
            self.span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(exc))
            self.span.record_exception(exc)
            self.span.set_status(StatusCode.ERROR, str(exc))
        except Exception as e:
            logger.warning("Failed to record error on span: %s", e)

    def end(self) -> None:
        """Detach the span from the active context and end it."""
        try:
            opentelemetry_context.detach(self._token)
        except Exception as e:
            logger.warning("Failed to detach span context: %s", e)
        try:
            self.span.end()
        except Exception as e:
            logger.warning("Failed to end span: %s", e)


def base_agent_run_async_wrapper(tracer: Tracer) -> Callable[..., AsyncIterator[Any]]:
    """Return a wrapt wrapper that creates an agent span around BaseAgent.run_async.

    Args:
        tracer: The OpenTelemetry tracer used to create agent spans.

    Returns:
        A wrapt-compatible wrapper function for BaseAgent.run_async.
    """

    def wrapper(
        wrapped: Callable[..., AsyncIterator[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Wrap a single BaseAgent.run_async call, creating and closing an agent span.

        Args:
            wrapped: The original BaseAgent.run_async coroutine.
            instance: The BaseAgent instance being called.
            args: Positional arguments passed to run_async.
            kwargs: Keyword arguments passed to run_async.

        Returns:
            An async generator that yields ADK events with agent span instrumentation.
        """

        async def new_function() -> AsyncIterator[Any]:
            agent_name = getattr(instance, "name", "unknown")
            try:
                scope = _SpanScope(tracer, f"ADK.Agent.{agent_name}")
            except Exception as e:
                logger.warning("Failed to start agent span: %s", e)
                async for event in wrapped(*args, **kwargs):
                    yield event
                return

            try:
                span = scope.span
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "adk")
                span.set_attribute("gen_ai.entity", "agent")
                span.set_attribute(NETRA_SPAN_TYPE, SpanType.AGENT)
                span.set_attributes(extract_agent_attributes(instance))

                invocation_context = args[0] if args else kwargs.get("invocation_context")
                if invocation_context:
                    if hasattr(invocation_context, "invocation_id"):
                        span.set_attribute("adk.invocation_id", invocation_context.invocation_id)
                    user_content = getattr(invocation_context, "user_content", None)
                    if user_content:
                        parts = getattr(user_content, "parts", []) or []
                        user_texts = [
                            str(getattr(p, "text", "")) for p in parts if getattr(p, "text", None) is not None
                        ]
                        if user_texts:
                            span.set_attribute("input", "\n".join(user_texts))
            except Exception as e:
                logger.warning("Failed to set agent span attributes: %s", e)

            last_text_output: List[str] = []
            try:
                async for event in wrapped(*args, **kwargs):
                    try:
                        parts = event.content.parts if event.content and event.content.parts else []
                        texts = [str(getattr(p, "text", "")) for p in parts if getattr(p, "text", None) is not None]
                        if texts:
                            last_text_output = texts
                    except Exception as e:
                        logger.warning("Failed to extract agent event text: %s", e)
                    yield event
            except Exception as e:
                scope.record_error(e)
                raise
            finally:
                if last_text_output:
                    try:
                        span.set_attribute("output", "\n".join(last_text_output))
                    except Exception as e:
                        logger.warning("Failed to set agent output attribute: %s", e)
                scope.end()

        return new_function()

    return cast(Callable[..., Any], wrapper)


def run_and_handle_error_wrapper(tracer: Tracer) -> Callable[..., AsyncIterator[Any]]:
    """Return a wrapt wrapper that creates an LLM generation span around BaseLlmFlow._run_and_handle_error.

    Args:
        tracer: The OpenTelemetry tracer used to create LLM generation spans.

    Returns:
        A wrapt-compatible wrapper function for _run_and_handle_error.
    """

    def wrapper(
        wrapped: Callable[..., AsyncIterator[Any]], _instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """Wrap a single _run_and_handle_error call, creating and closing an LLM span.

        Args:
            wrapped: The original _run_and_handle_error coroutine.
            _instance: The BaseLlmFlow instance.
            args: Positional arguments passed to _run_and_handle_error.
            kwargs: Keyword arguments passed to _run_and_handle_error.

        Returns:
            An async generator that yields ADK events with LLM span instrumentation.
        """

        async def new_function() -> AsyncIterator[Any]:
            invocation_context = args[1] if len(args) > 1 else kwargs.get("invocation_context")
            llm_request = args[2] if len(args) > 2 else kwargs.get("llm_request")
            model_name = getattr(llm_request, "model", "unknown") if llm_request else "unknown"

            try:
                scope = _SpanScope(tracer, model_name)
            except Exception as e:
                logger.warning("Failed to start LLM span: %s", e)
                async for item in wrapped(*args, **kwargs):
                    yield item
                return

            try:
                span = scope.span
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "request")
                span.set_attribute(NETRA_SPAN_TYPE, SpanType.GENERATION)

                if invocation_context:
                    if hasattr(invocation_context, "invocation_id"):
                        span.set_attribute("adk.invocation_id", invocation_context.invocation_id)
                    agent = getattr(invocation_context, "agent", None)
                    if agent and hasattr(agent, "name"):
                        span.set_attribute("gen_ai.agent.name", agent.name)

                if llm_request:
                    llm_request_dict = build_llm_request_for_trace(llm_request)
                    span.set_attributes(extract_llm_request_attributes(llm_request_dict))

            except Exception as e:
                logger.warning("Failed to set LLM span attributes: %s", e)

            accumulated_text: List[str] = []
            last_response = None

            # Peek-ahead buffer: hold back one item so the inner generator is
            # fully exhausted — and the span closed — before the last item
            # reaches the caller.  Without this, the caller processes the last
            # item (potentially launching sub-agent / tool spans) while the LLM
            # span is still open, inflating its duration.
            prev_item = None
            span_ended = False
            first_token_recorded = False

            try:
                # Capture timestamp right before the LLM call to measure TIME_TO_FIRST_TOKEN accurately.
                # ADK uses iterator start time as the closest approximation (span start introduces larger variance).
                # GenAI uses span start time since delay to actual call is negligible.
                # Keeps latency metrics reasonably aligned across both.
                llm_call_start = time.time()
                async for item in wrapped(*args, **kwargs):
                    if not first_token_recorded:
                        first_token_time = time.time()
                        record_span_timing(span, TIME_TO_FIRST_TOKEN, first_token_time, reference_time=llm_call_start)
                        record_span_timing(span, RELATIVE_TIME_TO_FIRST_TOKEN, first_token_time, use_root_span=True)
                        first_token_recorded = True
                    last_response = item
                    try:
                        parts = item.content.parts if item.content and item.content.parts else []
                        for part in parts:
                            if (text := getattr(part, "text", None)) is not None:
                                accumulated_text.append(str(text))
                    except Exception as e:
                        logger.warning("Failed to extract LLM response text: %s", e)
                    if prev_item is not None:
                        yield prev_item
                    prev_item = item

                # Inner generator is exhausted.  Close the span now, before
                # handing the last item to the caller.
                if last_response is not None:
                    try:
                        response_attrs = extract_llm_response_attributes(last_response, accumulated_text)
                        span.set_attributes(response_attrs)
                    except Exception as e:
                        logger.warning("Failed to set LLM response attributes: %s", e)

                scope.end()
                span_ended = True

                # Yield the last item after ending the span
                if prev_item is not None:
                    yield prev_item

            except Exception as e:
                if not span_ended:
                    scope.record_error(e)
                    scope.end()
                raise

        return new_function()

    return cast(Callable[..., Any], wrapper)


def call_tool_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt wrapper that creates a tool span around __call_tool_async.

    Args:
        tracer: The OpenTelemetry tracer used to create tool spans.

    Returns:
        A wrapt-compatible wrapper function for __call_tool_async.
    """

    def wrapper(wrapped: Callable[..., Any], _instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Wrap a single __call_tool_async call, creating and closing a tool span.

        Args:
            wrapped: The original __call_tool_async coroutine.
            _instance: The BaseLlmFlow instance.
            args: Positional arguments passed to __call_tool_async.
            kwargs: Keyword arguments passed to __call_tool_async.

        Returns:
            A coroutine that awaits the tool call with tool span instrumentation.
        """

        async def new_function() -> Any:
            tool = args[0] if args else kwargs.get("tool")
            tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})
            tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")

            tool_name = getattr(tool, "name", "unknown_tool")

            try:
                scope = _SpanScope(tracer, tool_name)
            except Exception as e:
                logger.warning("Failed to start tool span: %s", e)
                return await wrapped(*args, **kwargs)

            try:
                span = scope.span
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "tool")
                span.set_attribute(NETRA_SPAN_TYPE, SpanType.TOOL)
                span.set_attribute("gen_ai.tool.name", tool_name)

                if tool is not None and hasattr(tool, "description"):
                    span.set_attribute("gen_ai.tool.description", tool.description)
                if tool is not None and hasattr(tool, "is_long_running"):
                    span.set_attribute("gen_ai.tool.is_long_running", tool.is_long_running)

                if tool_args is not None:
                    try:
                        span.set_attribute(
                            "input", json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                        )
                    except Exception:
                        span.set_attribute("input", str(tool_args))

                if tool_context:
                    if hasattr(tool_context, "function_call_id"):
                        span.set_attribute("tool.call_id", tool_context.function_call_id)
                    if hasattr(tool_context, "invocation_context"):
                        span.set_attribute("adk.invocation_id", tool_context.invocation_context.invocation_id)
            except Exception as e:
                logger.warning("Failed to set tool span attributes: %s", e)

            try:
                result = await wrapped(*args, **kwargs)

                if result is not None:
                    try:
                        span.set_attribute("output", json.dumps(result) if isinstance(result, dict) else str(result))
                    except Exception:
                        span.set_attribute("output", str(result))

                return result
            except Exception as e:
                scope.record_error(e)
                raise
            finally:
                scope.end()

        return new_function()

    return cast(Callable[..., Any], wrapper)
