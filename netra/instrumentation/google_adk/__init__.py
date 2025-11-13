import logging
import sys
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.google_adk.version import __version__
from netra.instrumentation.google_adk.wrappers import (
    NoOpTracer,
    adk_trace_call_llm_wrapper,
    adk_trace_send_data_wrapper,
    adk_trace_tool_call_wrapper,
    adk_trace_tool_response_wrapper,
    base_agent_run_async_wrapper,
    base_llm_flow_call_llm_async_wrapper,
    call_tool_async_wrapper,
    finalize_model_response_event_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("google-adk >= 0.1.0",)


class NetraGoogleADKInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom Google ADK instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        # Set ADK tracer to NoOpTracer to prevent ADK from creating its own spans
        try:
            import google.adk.telemetry as adk_telemetry

            adk_telemetry.tracer = NoOpTracer()
        except Exception as e:
            logger.debug(f"Unable to replace ADK tracer: {e}")

        for module_name in (
            "google.adk.runners",
            "google.adk.agents.base_agent",
            "google.adk.flows.llm_flows.base_llm_flow",
            "google.adk.flows.llm_flows.functions",
        ):
            try:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if hasattr(module, "tracer"):
                        setattr(module, "tracer", NoOpTracer())
            except Exception as e:
                logger.debug(f"Unable to replace tracer in {module_name}: {e}")

        try:
            wrap_function_wrapper(
                "google.adk.agents.base_agent",
                "BaseAgent.run_async",
                base_agent_run_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument BaseAgent.run_async: {e}")

        try:
            wrap_function_wrapper("google.adk.telemetry", "trace_tool_call", adk_trace_tool_call_wrapper(tracer))
            wrap_function_wrapper(
                "google.adk.telemetry", "trace_tool_response", adk_trace_tool_response_wrapper(tracer)
            )
            wrap_function_wrapper("google.adk.telemetry", "trace_call_llm", adk_trace_call_llm_wrapper(tracer))
            wrap_function_wrapper("google.adk.telemetry", "trace_send_data", adk_trace_send_data_wrapper(tracer))
        except Exception as e:
            logger.error(f"Failed to instrument ADK telemetry functions: {e}")

        try:
            wrap_function_wrapper(
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._call_llm_async",
                base_llm_flow_call_llm_async_wrapper(tracer),
            )
            wrap_function_wrapper(
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._finalize_model_response_event",
                finalize_model_response_event_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument BaseLlmFlow methods: {e}")

        try:
            wrap_function_wrapper(
                "google.adk.flows.llm_flows.functions",
                "__call_tool_async",
                call_tool_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument __call_tool_async: {e}")

    def _uninstrument(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # Unwrap in reverse order
        try:
            unwrap("google.adk.flows.llm_flows.functions", "__call_tool_async")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument __call_tool_async")

        try:
            unwrap(
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._finalize_model_response_event",
            )
            unwrap(
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._call_llm_async",
            )
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument BaseLlmFlow methods")

        try:
            unwrap("google.adk.telemetry", "trace_send_data")
            unwrap("google.adk.telemetry", "trace_call_llm")
            unwrap("google.adk.telemetry", "trace_tool_response")
            unwrap("google.adk.telemetry", "trace_tool_call")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument ADK telemetry functions")

        try:
            unwrap("google.adk.agents.base_agent", "BaseAgent.run_async")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument BaseAgent.run_async")

        try:
            import google.adk.telemetry as adk_telemetry
            from opentelemetry import trace as otel_trace

            adk_telemetry.tracer = otel_trace.get_tracer("gcp.vertex.agent")
        except Exception:
            pass


__all__ = ["NetraGoogleADKInstrumentor"]
