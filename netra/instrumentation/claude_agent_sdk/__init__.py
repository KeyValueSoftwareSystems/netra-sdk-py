import wrapt
import logging
from opentelemetry.trace import Tracer, get_tracer
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from netra.instrumentation.claude_agent_sdk.version import __version__
from netra.instrumentation.claude_agent_sdk.wrappers import (
    client_query_wrapper, 
    client_response_wrapper, 
    query_wrapper
)

logger = logging.getLogger(__name__)

_instruments = ("claude_agent_sdk >= 0.1.0", )

class NetraClaudeAgentSDKInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self):
        return _instruments
    
    def _instrument(self, **kwargs):
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return
        self._instrument_query(tracer)
        self._instrument_client_query(tracer)
        self._instrument_client_response(tracer)

    def _uninstrument(self, **kwargs):
        self._uninstrument_query()
        self._uninstrument_client_query()
        self._uninstrument_client_response()

    def _instrument_query(self, tracer: Tracer):
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk._internal.client", 
                "InternalClient.process_query", 
                query_wrapper(tracer) 
            )
        except Exception as e:
            logger.error(f"Failed to instrument claude-agent-sdk query: {e}")

    def _instrument_client_query(self, tracer: Tracer):
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk.client", 
                "ClaudeSDKClient.query", 
                client_query_wrapper() 
            )
        except Exception as e:
            logger.error(f"Failed to instrument claude-sdk-client query: {e}")

    def _instrument_client_response(self, tracer: Tracer):
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk.client", 
                "ClaudeSDKClient.receive_messages", 
                client_response_wrapper(tracer) 
            )
        except Exception as e:
            logger.error(f"Failed to instrument claude-sdk-client response: {e}")

    def _uninstrument_query(self):
        try:
            unwrap("claude_agent_sdk._internal.client", "InternalClient.process_query")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-agent-sdk query")

    def _uninstrument_client_query(self):
        try:
            unwrap("claude_agent_sdk.client", "ClaudeSDKClient.query")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-sdk-client query")

    def _uninstrument_client_response(self):
        try:
            unwrap("claude_agent_sdk.client", "ClaudeSDKClient.receive_messages")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-sdk-client response")