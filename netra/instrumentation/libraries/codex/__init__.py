import logging
from typing import Any, Callable

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer

from netra.instrumentation.libraries.codex.version import __version__
from netra.instrumentation.libraries.codex.wrappers import (
    async_stream_wrapper,
    close_wrapper,
    start_turn_wrapper,
    stream_wrapper,
    thread_response_wrapper,
    write_message_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("openai-codex >= 0.162.0",)

# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.codex"

_CLIENT_MODULE = "openai_codex.client"
_API_MODULE = "openai_codex.api"


class NetraCodexInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """
    Instrumentor for the OpenAI Codex Python SDK (``openai-codex``).

    The SDK drives a ``codex app-server`` subprocess over JSON-RPC, so model calls never
    pass through Python HTTP clients. Spans are built from the turn notification stream:
    one ``codex.turn`` span per turn, with TOOL child spans for shell commands, file
    changes, MCP/dynamic tool calls and web searches, and per-model USAGE spans.
    """

    def instrumentation_dependencies(self) -> tuple[str, ...]:
        """
        Return the list of packages required for this instrumentation to function.

        Returns:
            tuple: A tuple of pip requirement strings for the instrumented library.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """
        Wrap the Codex client and turn handle methods with tracing wrappers.

        Args:
            **kwargs: Accepts an optional 'tracer_provider' (TracerProvider) to use
                      instead of the global provider.

        Returns:
            None
        """
        try:
            tracer = get_tracer(_TRACER_NAME, __version__, kwargs.get("tracer_provider"))
        except Exception as e:
            logger.error(f"Failed to initialize tracer: {e}")
            return

        thread_wrapper = thread_response_wrapper()
        self._wrap(_CLIENT_MODULE, "CodexClient.thread_start", thread_wrapper)
        self._wrap(_CLIENT_MODULE, "CodexClient.thread_resume", thread_wrapper)
        self._wrap(_CLIENT_MODULE, "CodexClient.thread_fork", thread_wrapper)
        self._wrap(_CLIENT_MODULE, "CodexClient._start_turn", start_turn_wrapper(tracer))
        self._wrap(_CLIENT_MODULE, "CodexClient._write_message", write_message_wrapper())
        self._wrap(_CLIENT_MODULE, "CodexClient.close", close_wrapper())
        self._wrap(_API_MODULE, "TurnHandle.stream", stream_wrapper(tracer))
        self._wrap(_API_MODULE, "AsyncTurnHandle.stream", async_stream_wrapper(tracer))

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Remove all tracing wrappers from the Codex SDK.

        Args:
            **kwargs: Not used; accepted for compatibility with BaseInstrumentor interface.

        Returns:
            None
        """
        for method in (
            "CodexClient.thread_start",
            "CodexClient.thread_resume",
            "CodexClient.thread_fork",
            "CodexClient._start_turn",
            "CodexClient._write_message",
            "CodexClient.close",
        ):
            self._unwrap(_CLIENT_MODULE, method)
        self._unwrap(_API_MODULE, "TurnHandle.stream")
        self._unwrap(_API_MODULE, "AsyncTurnHandle.stream")

    @staticmethod
    def _wrap(module: str, name: str, wrapper: Callable[..., Any]) -> None:
        """
        Wrap one SDK method, logging (not raising) if it is missing.

        Args:
            module (str): The module containing the class.
            name (str): ``Class.method`` to wrap.
            wrapper (Callable): The wrapt wrapper.

        Returns:
            None
        """
        try:
            wrapt.wrap_function_wrapper(module, name, wrapper)
        except Exception as e:
            logger.error(f"Failed to instrument codex {name}: {e}")

    @staticmethod
    def _unwrap(module: str, name: str) -> None:
        """
        Remove the wrapper from one SDK method.

        Args:
            module (str): The module containing the class.
            name (str): ``Class.method`` to unwrap.

        Returns:
            None
        """
        try:
            # unwrap() takes the dotted path of the class that owns the method, not of its module.
            class_name, method = name.split(".", 1)
            unwrap(f"{module}.{class_name}", method)
        except (AttributeError, ImportError):
            logger.error(f"Failed to uninstrument codex {name}")
