import logging
from typing import Any, AsyncIterator, Callable, Tuple

from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, UserMessage
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.claude_agent_sdk.utils import (
    set_assistant_message_attributes,
    set_request_attributes,
    set_result_message_attributes,
    set_system_message_attributes,
    set_user_message_attributes,
)

logger = logging.getLogger(__name__)

QUERY_SPAN_NAME = "claude-agent.query"
AGENT_CONVERSATION_SPAN_NAME = "claude-agent.conversation"


async def _dispatch_messages(
    tracer: Tracer,
    root_span: Span,
    root_ctx: Context,
    aiterator: AsyncIterator[Any],
    prompt_index: int,
) -> AsyncIterator[Any]:
    """
    Dispatch each incoming SDK message to its span attribute handler and yield it.

    Args:
        tracer (Tracer): The OpenTelemetry tracer for creating child spans.
        root_span (Span): The root span to attach message attributes to.
        root_ctx (Context): The root span context for child span parenting.
        aiterator (AsyncIterator): The async iterator of SDK messages to process.
        prompt_index (int): The current prompt index passed to the result message attribute setter.

    Returns:
        AsyncIterator: Yields each SDK message after processing its span attributes.
    """
    while True:
        try:
            message = await anext(aiterator)
        except StopAsyncIteration:
            return

        try:
            if isinstance(message, SystemMessage):
                set_system_message_attributes(root_span, message)
            elif isinstance(message, ResultMessage):
                set_result_message_attributes(root_span, message, prompt_index)
            elif isinstance(message, AssistantMessage):
                set_assistant_message_attributes(tracer, root_ctx, message)
            elif isinstance(message, UserMessage):
                set_user_message_attributes(tracer, root_ctx, message)
        except Exception as e:
            logger.error("Failed to record message attributes: %s", e)

        yield message


def query_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces a single InternalClient.process_query call with child spans per message.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: An async generator wrapper function for InternalClient.process_query.
    """

    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Wrap InternalClient.process_query to create a root span and dispatch per-message child spans.

        Args:
            wrapped (Callable): The original process_query function.
            instance (Any): The InternalClient instance.
            args (Tuple): Positional arguments
            kwargs (dict): Keyword arguments; may include 'prompt' and 'options'.

        Returns:
            AsyncIterator: Yields SDK messages with span instrumentation applied.
        """
        prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        options = args[1] if len(args) > 1 else kwargs.get("options")

        root_span = tracer.start_span(QUERY_SPAN_NAME, kind=SpanKind.CLIENT)
        root_ctx = trace.set_span_in_context(root_span)
        prompt_index = 0

        try:
            try:
                with trace.use_span(root_span, end_on_exit=False):
                    prompt_index = set_request_attributes(root_span, prompt, options)
            except Exception as e:
                logger.error("Instrumentation setup failed: %s", e)

            aiterator = aiter(wrapped(*args, **kwargs))
            async for message in _dispatch_messages(tracer, root_span, root_ctx, aiterator, prompt_index):
                yield message

        except GeneratorExit:
            raise
        except Exception as e:
            logger.error("netra.instrumentation.claude-agent-sdk: %s", e)
            try:
                root_span.record_exception(e)
                root_span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception as span_err:
                logger.error("Failed to record exception on span: %s", span_err)
            raise
        finally:
            try:
                root_span.end()
            except Exception as e:
                logger.error("Failed to end span: %s", e)

    return wrapper


def client_query_wrapper() -> Callable[..., Any]:
    """
    Return a wrapper that captures the prompt from ClaudeSDKClient.query for later tracing.

    Args:
        None

    Returns:
        Callable: An async wrapper function for ClaudeSDKClient.query.
    """

    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Intercept ClaudeSDKClient.query to store the prompt on the instance for later tracing.

        Args:
            wrapped (Callable): The original query function.
            instance (Any): The ClaudeSDKClient instance; prompt is stored as _instrumentation_prompt.
            args (Tuple): Positional arguments
            kwargs (dict): Keyword arguments; may include 'prompt'.

        Returns:
            Any: The result of the original query function.
        """
        instance._instrumentation_prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        return await wrapped(*args, **kwargs)

    return wrapper


def client_response_wrapper(tracer: Tracer) -> Callable[..., Any]:
    """
    Return a wrapper that traces a full ClaudeSDKClient.receive_messages call covering all messages.

    Args:
        tracer (Tracer): The OpenTelemetry tracer to use for creating spans.

    Returns:
        Callable: An async generator wrapper function for ClaudeSDKClient.receive_messages.
    """

    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Wrap ClaudeSDKClient.receive_messages to create a root span and dispatch per-message child spans.

        Args:
            wrapped (Callable): The original receive_messages function.
            instance (Any): The ClaudeSDKClient instance; prompt and options are read from it.
            args (Tuple): Positional arguments forwarded to the original function.
            kwargs (dict): Keyword arguments forwarded to the original function.

        Returns:
            AsyncIterator: Yields SDK messages with span instrumentation applied.
        """
        prompt = getattr(instance, "_instrumentation_prompt", None)
        options = getattr(instance, "options", None)

        root_span = tracer.start_span(AGENT_CONVERSATION_SPAN_NAME, kind=SpanKind.CLIENT)
        root_ctx = trace.set_span_in_context(root_span)
        prompt_index = 0

        try:
            try:
                with trace.use_span(root_span, end_on_exit=False):
                    prompt_index = set_request_attributes(root_span, prompt, options)
            except Exception as e:
                logger.error("Instrumentation setup failed: %s", e)

            aiterator = aiter(wrapped(*args, **kwargs))
            async for message in _dispatch_messages(tracer, root_span, root_ctx, aiterator, prompt_index):
                yield message

        except GeneratorExit:
            raise
        except Exception as e:
            logger.error("netra.instrumentation.claude-agent-sdk: %s", e)
            try:
                root_span.record_exception(e)
                root_span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception as span_err:
                logger.error("Failed to record exception on span: %s", span_err)
            raise
        finally:
            try:
                root_span.end()
            except Exception as e:
                logger.error("Failed to end span: %s", e)

    return wrapper
