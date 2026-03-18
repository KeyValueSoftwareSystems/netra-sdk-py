import logging
from typing import Any, AsyncIterator, Callable, Tuple
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from claude_agent_sdk import SystemMessage, AssistantMessage, UserMessage, ResultMessage

from netra.instrumentation.claude_agent_sdk.utils import (
    set_request_attributes,
    set_system_message_attributes,
    set_assistant_message_attributes,
    set_user_message_attributes,
    set_result_message_attributes,
)

logger = logging.getLogger(__name__)

QUERY_SPAN_NAME = "claude-agent.query"
AGENT_CONVERSATION_SPAN_NAME = "claude-agent.conversation"


async def _dispatch_messages(
    tracer: Tracer,
    root_span: Span,
    root_ctx: Context,
    aiterator: AsyncIterator,
    prompt_index: int,
) -> AsyncIterator:
    """Dispatch each incoming SDK message to its span attribute handler and yield it."""
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


def query_wrapper(tracer: Tracer):
    """Traces a single query through InternalClient, creating child spans per message."""
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
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
            except Exception:
                pass
            raise
        finally:
            try:
                root_span.end()
            except Exception as e:
                logger.error("Failed to end span: %s", e)

    return wrapper


def client_query_wrapper():
    """Intercepts ClaudeSDKClient.query to capture the prompt for later tracing."""
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        instance._instrumentation_prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        return await wrapped(*args, **kwargs)

    return wrapper


def client_response_wrapper(tracer: Tracer):
    """Traces a full ClaudeSDKClient conversation, covering all messages from prompt to result."""
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
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
            except Exception:
                pass
            raise
        finally:
            try:
                root_span.end()
            except Exception as e:
                logger.error("Failed to end span: %s", e)

    return wrapper
