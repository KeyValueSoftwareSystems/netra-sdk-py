import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Tuple, cast

import wrapt
from opentelemetry import context as opentelemetry_context
from opentelemetry import trace as opentelemetry_api_trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, StatusCode, Tracer

from netra.config import Config
from netra.instrumentation.google_adk.utils import (
    _build_llm_request_for_trace,
    extract_agent_attributes,
    extract_llm_attributes,
    extract_llm_request_attributes,
    extract_llm_response_attributes,
)
from netra.span_wrapper import SpanType

logger = logging.getLogger(__name__)


class NoOpSpan:
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

    def record_error(self, exc: Exception) -> None:
        try:
            self.span.set_attribute(f"{Config.LIBRARY_NAME}.entity.error", str(exc))
            self.span.record_exception(exc)
            self.span.set_status(StatusCode.ERROR, str(exc))
        except Exception as e:
            logger.warning("Failed to record error on span: %s", e)

    def end(self) -> None:
        try:
            opentelemetry_context.detach(self._token)
            self.span.end()
        except Exception as e:
            logger.warning("Failed to end span: %s", e)


def base_agent_run_async_wrapper(tracer: Tracer) -> Callable[..., AsyncIterator[Any]]:
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
                span.set_attribute("netra.span.type", SpanType.AGENT)
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
                span.set_attribute("netra.span.type", SpanType.GENERATION)

                if llm_request:
                    llm_request_dict = _build_llm_request_for_trace(llm_request)
                    span.set_attributes(extract_llm_request_attributes(llm_request_dict))

            except Exception as e:
                logger.warning("Failed to set LLM span attributes: %s", e)

            accumulated_text: List[str] = []
            last_response = None

            try:
                async for item in wrapped(*args, **kwargs):
                    last_response = item
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
                        span.set_attributes(extract_llm_response_attributes(last_response, accumulated_text))
                    except Exception as e:
                        logger.warning("Failed to set LLM response attributes: %s", e)

            except Exception as e:
                scope.record_error(e)
                raise
            finally:
                scope.end()

        return new_function()

    return cast(Callable[..., Any], wrapper)


def finalize_model_response_event_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = wrapped(*args, **kwargs)

        llm_request = args[0] if len(args) > 0 else kwargs.get("llm_request")
        llm_response = args[1] if len(args) > 1 else kwargs.get("llm_response")

        current_span = opentelemetry_api_trace.get_current_span()
        if current_span.is_recording() and llm_request and llm_response:
            span_name = getattr(current_span, "name", "")
            if "ADK.LLM" in span_name:
                llm_request_dict = _build_llm_request_for_trace(llm_request)
                llm_response_json = llm_response.model_dump_json(exclude_none=True)
                llm_attrs = extract_llm_attributes(llm_request_dict, llm_response)

                for key, value in llm_attrs.items():
                    if "usage" in key or "completion" in key or "response" in key:
                        current_span.set_attribute(key, value)

        return result

    return cast(Callable[..., Any], wrapper)


def adk_trace_tool_call_wrapper(tracer: Tracer) -> Callable[..., Any]:
    @wrapt.decorator  # type: ignore[misc]
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = wrapped(*args, **kwargs)

        tool_args = args[0] if args else kwargs.get("args")
        current_span = opentelemetry_api_trace.get_current_span()
        if current_span.is_recording() and tool_args is not None:
            current_span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
            current_span.set_attribute("gcp.vertex.agent.tool_call_args", str(tool_args))
        return result

    return cast(Callable[..., Any], wrapper)


def adk_trace_tool_response_wrapper(tracer: Tracer) -> Callable[..., Any]:
    @wrapt.decorator  # type: ignore[misc]
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = wrapped(*args, **kwargs)

        invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
        event_id = args[1] if len(args) > 1 else kwargs.get("event_id")
        function_response_event = args[2] if len(args) > 2 else kwargs.get("function_response_event")

        current_span = opentelemetry_api_trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
            if invocation_context:
                current_span.set_attribute("gcp.vertex.agent.invocation_id", invocation_context.invocation_id)
            if event_id:
                current_span.set_attribute("gcp.vertex.agent.event_id", event_id)
            if function_response_event:
                current_span.set_attribute(
                    "gcp.vertex.agent.tool_response", function_response_event.model_dump_json(exclude_none=True)
                )
            current_span.set_attribute("gcp.vertex.agent.llm_request", "{}")
            current_span.set_attribute("gcp.vertex.agent.llm_response", "{}")
        return result

    return cast(Callable[..., Any], wrapper)


def adk_trace_call_llm_wrapper(tracer: Tracer) -> Callable[..., Any]:
    @wrapt.decorator  # type: ignore[misc]
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = wrapped(*args, **kwargs)

        invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
        event_id = args[1] if len(args) > 1 else kwargs.get("event_id")
        llm_request = args[2] if len(args) > 2 else kwargs.get("llm_request")
        llm_response = args[3] if len(args) > 3 else kwargs.get("llm_response")

        current_span = opentelemetry_api_trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
            if llm_request:
                current_span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, llm_request.model)
            if invocation_context:
                current_span.set_attribute("gcp.vertex.agent.invocation_id", invocation_context.invocation_id)
                current_span.set_attribute("gcp.vertex.agent.session_id", invocation_context.session.id)
            if event_id:
                current_span.set_attribute("gcp.vertex.agent.event_id", event_id)

            if llm_request:
                llm_request_dict = _build_llm_request_for_trace(llm_request)
                current_span.set_attribute("gcp.vertex.agent.llm_request", json.dumps(llm_request_dict))

                llm_response_json = None
                if llm_response:
                    llm_response_json = llm_response.model_dump_json(exclude_none=True)
                    current_span.set_attribute("gcp.vertex.agent.llm_response", llm_response_json)

                llm_attrs = extract_llm_attributes(llm_request_dict, llm_response)
                for key, value in llm_attrs.items():
                    current_span.set_attribute(key, value)

        return result

    return cast(Callable[..., Any], wrapper)


def adk_trace_send_data_wrapper(tracer: Tracer) -> Callable[..., Any]:
    @wrapt.decorator  # type: ignore[misc]
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        result = wrapped(*args, **kwargs)

        invocation_context = args[0] if len(args) > 0 else kwargs.get("invocation_context")
        event_id = args[1] if len(args) > 1 else kwargs.get("event_id")
        data = args[2] if len(args) > 2 else kwargs.get("data")

        current_span = opentelemetry_api_trace.get_current_span()
        if current_span.is_recording():
            if invocation_context:
                current_span.set_attribute("gcp.vertex.agent.invocation_id", invocation_context.invocation_id)
            if event_id:
                current_span.set_attribute("gcp.vertex.agent.event_id", event_id)
            if data:
                from google.genai import types

                current_span.set_attribute(
                    "gcp.vertex.agent.data",
                    json.dumps(
                        [
                            types.Content(role=content.role, parts=content.parts).model_dump(exclude_none=True)
                            for content in data
                        ]
                    ),
                )
        return result

    return cast(Callable[..., Any], wrapper)


def call_tool_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        async def new_function() -> Any:
            tool = args[0] if args else kwargs.get("tool")
            tool_args = args[1] if len(args) > 1 else kwargs.get("args", {})
            tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")

            tool_name = getattr(tool, "name", "unknown_tool")
            # span_name = f"ADK.Tool.{tool_name}"
            span_name = tool_name

            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "tool")
                span.set_attribute("netra.span.type", "tool")

                span.set_attribute("gen_ai.tool.name", tool_name)
                if tool is not None and hasattr(tool, "description"):
                    span.set_attribute("gen_ai.tool.description", getattr(tool, "description"))
                if tool is not None and hasattr(tool, "is_long_running"):
                    span.set_attribute("gen_ai.tool.is_long_running", getattr(tool, "is_long_running"))
                span.set_attribute("gen_ai.tool.parameters", str(tool_args))

                if tool_context and hasattr(tool_context, "function_call_id"):
                    span.set_attribute("tool.call_id", tool_context.function_call_id)
                if tool_context and hasattr(tool_context, "invocation_context"):
                    span.set_attribute("adk.invocation_id", tool_context.invocation_context.invocation_id)

                result = await wrapped(*args, **kwargs)

                if result:
                    if isinstance(result, dict):
                        span.set_attribute("gen_ai.tool.result", json.dumps(result))
                    else:
                        span.set_attribute("gen_ai.tool.result", str(result))

                return result

        return new_function()

    return wrapper
