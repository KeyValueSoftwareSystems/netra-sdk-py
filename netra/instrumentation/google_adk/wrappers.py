import json
import logging
import sys
from typing import Any, AsyncIterator, Callable, Dict, Tuple, cast

import wrapt
from opentelemetry import trace as opentelemetry_api_trace
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Tracer

from netra.instrumentation.google_adk.utils import (
    _build_llm_request_for_trace,
    _extract_llm_attributes,
    extract_agent_attributes,
)

logger = logging.getLogger(__name__)


class NoOpSpan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_attributes(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update_name(self, *args: Any, **kwargs: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class NoOpTracer:
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    def use_span(self, *args: Any, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()


def base_agent_run_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        async def new_function() -> AsyncIterator[Any]:
            agent_name = instance.name if hasattr(instance, "name") else "unknown"
            span_name = f"adk.agent.{agent_name}"

            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "agent")
                span.set_attribute("netra.span.type", "span")

                span.set_attributes(extract_agent_attributes(instance))
                if len(args) > 0 and hasattr(args[0], "invocation_id"):
                    span.set_attribute("adk.invocation_id", args[0].invocation_id)

                async_gen = wrapped(*args, **kwargs)
                async for item in async_gen:
                    yield item

        return new_function()

    return cast(Callable[..., Any], wrapper)


# --- Patching helpers and public patch/unpatch API ---
_wrapped_methods: list[tuple[Any, str]] = []


def _patch(
    module_name: str,
    object_name: str,
    method_name: str,
    wrapper_function: Callable[[Tracer], Callable[..., Any]],
    tracer: Tracer,
) -> None:
    try:
        module = __import__(module_name, fromlist=[object_name])
        obj = getattr(module, object_name)
        wrapt.wrap_function_wrapper(obj, method_name, wrapper_function(tracer))
        _wrapped_methods.append((obj, method_name))
        logger.debug(f"Wrapped {module_name}.{object_name}.{method_name}")
    except Exception as e:
        logger.warning(f"Failed to wrap {module_name}.{object_name}.{method_name}: {e}")


def _patch_module_function(
    module_name: str,
    function_name: str,
    wrapper_function: Callable[[Tracer], Callable[..., Any]],
    tracer: Tracer,
) -> None:
    try:
        module = __import__(module_name, fromlist=[function_name])
        wrapt.wrap_function_wrapper(module, function_name, wrapper_function(tracer))
        _wrapped_methods.append((module, function_name))
        logger.debug(f"Wrapped {module_name}.{function_name}")
    except Exception as e:
        logger.warning(f"Failed to wrap {module_name}.{function_name}: {e}")


def patch_adk(tracer: Tracer) -> None:
    logger.debug("Applying Google ADK patches")

    noop_tracer = NoOpTracer()
    try:
        import google.adk.telemetry as adk_telemetry

        adk_telemetry.tracer = noop_tracer
        logger.debug("Replaced ADK tracer with NoOpTracer")
    except Exception as e:
        logger.warning(f"Failed replacing ADK tracer: {e}")

    for module_name in [
        "google.adk.runners",
        "google.adk.agents.base_agent",
        "google.adk.flows.llm_flows.base_llm_flow",
        "google.adk.flows.llm_flows.functions",
    ]:
        if module_name in sys.modules:
            try:
                module = sys.modules[module_name]
                if hasattr(module, "tracer"):
                    setattr(module, "tracer", noop_tracer)
                    logger.debug(f"Replaced tracer in {module_name}")
            except Exception as e:
                logger.warning(f"Failed replacing tracer in {module_name}: {e}")

    _patch("google.adk.agents.base_agent", "BaseAgent", "run_async", base_agent_run_async_wrapper, tracer)

    _patch_module_function("google.adk.telemetry", "trace_tool_call", adk_trace_tool_call_wrapper, tracer)
    _patch_module_function("google.adk.telemetry", "trace_tool_response", adk_trace_tool_response_wrapper, tracer)
    _patch_module_function("google.adk.telemetry", "trace_call_llm", adk_trace_call_llm_wrapper, tracer)
    _patch_module_function("google.adk.telemetry", "trace_send_data", adk_trace_send_data_wrapper, tracer)

    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_call_llm_async",
        base_llm_flow_call_llm_async_wrapper,
        tracer,
    )

    _patch(
        "google.adk.flows.llm_flows.base_llm_flow",
        "BaseLlmFlow",
        "_finalize_model_response_event",
        finalize_model_response_event_wrapper,
        tracer,
    )

    _patch_module_function("google.adk.flows.llm_flows.functions", "__call_tool_async", call_tool_async_wrapper, tracer)

    logger.info("Google ADK patching complete")


def unpatch_adk() -> None:
    logger.debug("Removing Google ADK patches")

    try:
        import google.adk.telemetry as adk_telemetry
        from opentelemetry import trace

        adk_telemetry.tracer = trace.get_tracer("gcp.vertex.agent")
        logger.debug("Restored ADK tracer")
    except Exception as e:
        logger.warning(f"Failed restoring ADK tracer: {e}")

    for obj, method_name in _wrapped_methods:
        try:
            if hasattr(getattr(obj, method_name), "__wrapped__"):
                original = getattr(obj, method_name).__wrapped__
                setattr(obj, method_name, original)
                logger.debug(f"Unwrapped {obj}.{method_name}")
        except Exception as e:
            logger.warning(f"Failed unwrapping {obj}.{method_name}: {e}")

    _wrapped_methods.clear()
    logger.info("Google ADK unpatching complete")


def base_llm_flow_call_llm_async_wrapper(tracer: Tracer) -> Callable[..., Any]:
    def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        async def new_function() -> AsyncIterator[Any]:
            model_name = "unknown"
            llm_request = None

            if len(args) > 1:
                llm_request = args[1]
                if hasattr(llm_request, "model"):
                    model_name = llm_request.model

            span_name = f"adk.llm.{model_name}"

            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "request")
                span.set_attribute("netra.span.type", "generation")

                if llm_request:
                    llm_request_dict = _build_llm_request_for_trace(llm_request)
                    llm_attrs = _extract_llm_attributes(llm_request_dict, None)
                    for key, value in llm_attrs.items():
                        span.set_attribute(key, value)

                async_gen = wrapped(*args, **kwargs)
                async for item in async_gen:
                    yield item

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
            if "adk.llm" in span_name:
                llm_request_dict = _build_llm_request_for_trace(llm_request)
                llm_response_json = llm_response.model_dump_json(exclude_none=True)
                llm_attrs = _extract_llm_attributes(llm_request_dict, llm_response_json)

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
            current_span.set_attribute("gcp.vertex.agent.tool_call_args", json.dumps(tool_args))
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

                llm_attrs = _extract_llm_attributes(llm_request_dict, llm_response_json)
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
            span_name = f"adk.tool.{tool_name}"

            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute(SpanAttributes.LLM_SYSTEM, "gcp.vertex.agent")
                span.set_attribute("gen_ai.entity", "tool")
                span.set_attribute("netra.span.type", "tool")

                span.set_attribute("gen_ai.tool.name", tool_name)
                if tool is not None and hasattr(tool, "description"):
                    span.set_attribute("gen_ai.tool.description", getattr(tool, "description"))
                if tool is not None and hasattr(tool, "is_long_running"):
                    span.set_attribute("gen_ai.tool.is_long_running", getattr(tool, "is_long_running"))
                span.set_attribute("gen_ai.tool.parameters", json.dumps(tool_args))

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
