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
    base_agent_run_async_wrapper,
    base_llm_flow_call_llm_async_wrapper,
    call_tool_async_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("google-adk >= 0.1.0",)

_ADK_TRACER_MODULES = (
    "google.adk.agents.base_agent",
    "google.adk.flows.llm_flows.base_llm_flow",
    "google.adk.flows.llm_flows.functions",
    "google.adk.models.gemini_context_cache_manager",
    "google.adk.models.google_llm",
    "google.adk.plugins.bigquery_agent_analytics_plugin",
    "google.adk.runners",
    "google.adk.telemetry",
    "google.adk.telemetry.tracing",
    "google.cloud.bigquery.opentelemetry_tracing",
    "google.cloud.pubsub_v1.open_telemetry.publish_message_wrapper",
    "google.cloud.pubsub_v1.open_telemetry.subscribe_opentelemetry",
    "google.cloud.pubsub_v1.publisher._batch.thread",
    "google.cloud.spanner_v1._opentelemetry_tracing",
    "google.cloud.sqlalchemy_spanner._opentelemetry_tracing",
    "google.cloud.storage._opentelemetry_tracing",
)


class NetraGoogleADKInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Custom Google ADK instrumentor for Netra SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the package requirements for this instrumentor."""
        return _instruments

    def _instrument(self, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """Patch ADK with Netra spans and replace ADK's own tracers with NoOps to avoid duplicates."""
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        # Replace ADK's own tracers with NoOp to prevent duplicate spans
        for module_name in _ADK_TRACER_MODULES:
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
            wrap_function_wrapper(
                "google.adk.flows.llm_flows.base_llm_flow",
                "BaseLlmFlow._call_llm_async",
                base_llm_flow_call_llm_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument BaseLlmFlow._call_llm_async: {e}")

        try:
            wrap_function_wrapper(
                "google.adk.flows.llm_flows.functions",
                "__call_tool_async",
                call_tool_async_wrapper(tracer),
            )
        except Exception as e:
            logger.error(f"Failed to instrument __call_tool_async: {e}")

    def _uninstrument(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Remove Netra wrappers. Note: replaced ADK tracers are intentionally left as NoOps."""
        try:
            unwrap("google.adk.flows.llm_flows.functions", "__call_tool_async")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument __call_tool_async")

        try:
            unwrap("google.adk.flows.llm_flows.base_llm_flow", "BaseLlmFlow._call_llm_async")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument BaseLlmFlow._call_llm_async")

        try:
            unwrap("google.adk.agents.base_agent", "BaseAgent.run_async")
        except (AttributeError, ModuleNotFoundError):
            logger.debug("Failed to uninstrument BaseAgent.run_async")


__all__ = ["NetraGoogleADKInstrumentor"]
