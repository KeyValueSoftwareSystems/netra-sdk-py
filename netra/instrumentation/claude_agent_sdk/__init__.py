import logging
from typing import Any

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Tracer, get_tracer

from netra.instrumentation.claude_agent_sdk.version import __version__
from netra.instrumentation.claude_agent_sdk.wrappers import client_query_wrapper, client_response_wrapper, query_wrapper

logger = logging.getLogger(__name__)

_instruments = ("claude_agent_sdk >= 0.1.0",)


class NetraClaudeAgentSDKInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    def instrumentation_dependencies(self) -> tuple[str, ...]:
        """
        Return the list of packages required for this instrumentation to function.

        Args:
            None

        Returns:
            tuple: A tuple of pip requirement strings for the instrumented library.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Set up OpenTelemetry instrumentation for the Claude Agent SDK.

        Wraps InternalClient.process_query, ClaudeSDKClient.query, and
        ClaudeSDKClient.receive_messages with tracing wrappers.

        Args:
            **kwargs: Accepts an optional 'tracer_provider' (TracerProvider) to use
                      instead of the global provider.

        Returns:
            None
        """
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return
        self._instrument_query(tracer)
        self._instrument_client_query(tracer)
        self._instrument_client_response(tracer)

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Remove all custom instrumentation wrappers from the Claude Agent SDK.

        Args:
            **kwargs: Not used; accepted for compatibility with BaseInstrumentor interface.

        Returns:
            None
        """
        self._uninstrument_query()
        self._uninstrument_client_query()
        self._uninstrument_client_response()

    def _instrument_query(self, tracer: Tracer) -> None:
        """
        Wrap InternalClient.process_query with a tracing wrapper.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the query wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk._internal.client", "InternalClient.process_query", query_wrapper(tracer)
            )
        except Exception as e:
            logger.error(f"Failed to instrument claude-agent-sdk query: {e}")

    def _instrument_client_query(self, tracer: Tracer) -> None:
        """
        Wrap ClaudeSDKClient.query to capture the prompt for downstream tracing.

        Args:
            tracer (Tracer): The OpenTelemetry tracer (accepted for interface consistency).

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper("claude_agent_sdk.client", "ClaudeSDKClient.query", client_query_wrapper())
        except Exception as e:
            logger.error(f"Failed to instrument claude-sdk-client query: {e}")

    def _instrument_client_response(self, tracer: Tracer) -> None:
        """
        Wrap ClaudeSDKClient.receive_messages with a tracing wrapper.

        Args:
            tracer (Tracer): The OpenTelemetry tracer to pass to the response wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper(
                "claude_agent_sdk.client", "ClaudeSDKClient.receive_messages", client_response_wrapper(tracer)
            )
        except Exception as e:
            logger.error(f"Failed to instrument claude-sdk-client response: {e}")

    def _uninstrument_query(self) -> None:
        """
        Remove the tracing wrapper from InternalClient.process_query.

        Args:
            None

        Returns:
            None
        """
        try:
            unwrap("claude_agent_sdk._internal.client", "InternalClient.process_query")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-agent-sdk query")

    def _uninstrument_client_query(self) -> None:
        """
        Remove the tracing wrapper from ClaudeSDKClient.query.

        Args:
            None

        Returns:
            None
        """
        try:
            unwrap("claude_agent_sdk.client", "ClaudeSDKClient.query")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-sdk-client query")

    def _uninstrument_client_response(self) -> None:
        """
        Remove the tracing wrapper from ClaudeSDKClient.receive_messages.

        Args:
            None

        Returns:
            None
        """
        try:
            unwrap("claude_agent_sdk.client", "ClaudeSDKClient.receive_messages")
        except (AttributeError, ModuleNotFoundError):
            logger.error(f"Failed to uninstrument claude-sdk-client response")
