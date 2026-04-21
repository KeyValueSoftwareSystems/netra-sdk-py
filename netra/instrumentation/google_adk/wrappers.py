import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Tuple, cast

from opentelemetry import context as opentelemetry_context
from opentelemetry import trace as opentelemetry_api_trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode, Tracer

from netra.config import Config
from netra.instrumentation.google_adk.utils import (
    NETRA_SPAN_TYPE,
    build_llm_request_for_trace,
    extract_agent_attributes,
    extract_llm_request_attributes,
    extract_llm_response_attributes,
    extract_scalar_event_attributes,
)
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)


class NoOpSpan:
    """Span implementation that silently discards all operations, used to suppress ADK's built-in tracing."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *_args: Any) -> None:
        pass

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_attributes(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def add_event(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_status(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def update_name(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def end(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class NoOpTracer:
    """Tracer that returns NoOpSpans, injected into ADK modules to prevent duplicate span emission."""

    def start_as_current_span(self, *_args: Any, **_kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, *_args: Any, **_kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def use_span(self, *_args: Any, **_kwargs: Any) -> NoOpSpan:
        return NoOpSpan()


class _SpanScope:
    """Manages span lifecycle for async generators (start, attach context, record errors, detach, end)."""

    def __init__(self, tracer: Tracer, name: str, kind: SpanKind = SpanKind.CLIENT) -> None:
        self.span = tracer.start_span(name, kind=kind)
        ctx = opentelemetry_api_trace.set_span_in_context(self.span)
        self._token = opentelemetry_context.attach(ctx)
        self._token_valid = True

    def record_error(self, exc: Exception) -> None:
        """Record the exception in the span"""
        try:
            self.span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(exc))
            self.span.record_exception(exc)
            self.span.set_status(StatusCode.ERROR, str(exc))
        except Exception as e:
            logger.warning("Failed to record error on span: %s", e)

    def detach(self) -> None:
        """Detach the span from the active context without ending the span"""
        if not self._token_valid:
            return
        try:
            opentelemetry_context.detach(self._token)
            self._token_valid = False
        except Exception as e:
            logger.warning("Failed to detach span context: %s", e)

    def end(self) -> None:
        """Detach the span from active context (if attached) and end the span"""
        try:
            self.detach()
            self.span.end()
        except Exception as e:
            logger.warning("Failed to end span: %s", e)


def base_agent_run_async_wrapper(tracer: Tracer) -> Callable[..., AsyncIterator[Any]]:
    """Return a wrapt wrapper that creates an agent span around BaseAgent.run_async."""

    def wrapper(
        wrapped: Callable[..., AsyncIterator[Any]], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
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

                invocation_context = args[0] if args else None
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

            try:
                async for event in wrapped(*args, **kwargs):
                    yield event
            except Exception as e:
                scope.record_error(e)
                raise
            finally:
                scope.end()

        return new_function()

    return cast(Callable[..., Any], wrapper)


def base_llm_flow_call_llm_async_wrapper(tracer: Tracer) -> Callable[..., AsyncIterator[Any]]:
    """Return a wrapt wrapper that creates an LLM generation span around BaseLlmFlow._call_llm_async."""

    def wrapper(
        wrapped: Callable[..., AsyncIterator[Any]], _instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        async def new_function() -> AsyncIterator[Any]:
            llm_request = args[1] if len(args) > 1 else None
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

                if llm_request:
                    llm_request_dict = build_llm_request_for_trace(llm_request)
                    span.set_attributes(extract_llm_request_attributes(llm_request_dict))

            except Exception as e:
                logger.warning("Failed to set LLM span attributes: %s", e)

            # Detach the LLM span from the active context so subsequent spans
            # (sub-agent calls, tool executions) remain direct children of the
            # parent agent span rather than nested under this LLM span.
            scope.detach()

            accumulated_text: List[str] = []
            last_response = None
            last_usage_response = None

            try:
                async for item in wrapped(*args, **kwargs):
                    last_response = item
                    if getattr(item, "usage_metadata", None) is not None:
                        last_usage_response = item
                    try:
                        parts = item.content.parts if item.content and item.content.parts else []
                        for part in parts:
                            if (text := getattr(part, "text", None)) is not None:
                                accumulated_text.append(str(text))
                    except Exception:
                        pass
                    yield item

                if last_response is not None:
                    try:
                        response_attrs = extract_llm_response_attributes(last_response, accumulated_text)
                        if last_usage_response is not None and last_usage_response is not last_response:
                            _, usage_attrs = extract_scalar_event_attributes(last_usage_response)
                            usage_keys = {
                                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                                "gen_ai.usage.cached_tokens",
                                "gen_ai.usage.thoughts_tokens",
                            }
                            for key in usage_keys:
                                if key in usage_attrs:
                                    response_attrs[key] = usage_attrs[key]
                        span.set_attributes(response_attrs)
                    except Exception as e:
                        logger.warning("Failed to set LLM response attributes: %s", e)

            except Exception as e:
                scope.record_error(e)
                raise
            finally:
                scope.end()

        return new_function()

    return cast(Callable[..., Any], wrapper)


def call_tool_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """Return a wrapt wrapper that creates a tool span around __call_tool_async."""

    def wrapper(wrapped: Callable[..., Any], _instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
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
