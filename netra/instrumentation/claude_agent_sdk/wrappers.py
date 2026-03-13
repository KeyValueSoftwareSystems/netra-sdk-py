import json
import logging
from typing import Any, Callable, Dict, Tuple
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from netra.instrumentation.claude_agent_sdk.utils import (
    set_request_attributes, 
    set_response_message_attributes
)

logger = logging.getLogger(__name__)

QUERY_SPAN_NAME = "claude-agent-sdk.query"
SDK_CLIENT_SPAN_NAME = "claude-agent-sdk.sdk-client"

def query_wrapper(tracer: Tracer):
    async def wrapper(
        wrapped: Callable[..., Any], 
        instance: Any, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> Any:
        prompt_index = 0
        span = tracer.start_span(
            QUERY_SPAN_NAME,
            kind=SpanKind.CLIENT
        )
        try:
            with trace.use_span(span, end_on_exit=False):
                prompt_index = set_request_attributes(span, kwargs, prompt_index)
                aiterator = aiter(wrapped(*args, **kwargs))
                
            while True:
                with trace.use_span(span, end_on_exit=False):
                    try:
                        message = await anext(aiterator)
                    except StopAsyncIteration:
                        break
                
                prompt_index = set_response_message_attributes(span, message, prompt_index)
                yield message
                
        except GeneratorExit:
            raise
        except Exception as e:
            logger.error("netra.instrumentation.claude-agent-sdk: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()

    return wrapper

def client_query_wrapper():
    async def wrapper(
        wrapped: Callable[..., Any], 
        instance: Any, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> Any:
        # Add prompt to the client object to map to the corresponding response 
        # in recieve function
        prompt = args[0] if len(args) > 0 else kwargs.get("prompt")
        instance._prompt_data = { "prompt": prompt }
        return await wrapped(*args, **kwargs)
    return wrapper


def client_response_wrapper(tracer: Tracer):
    async def wrapper(
        wrapped: Callable[..., Any], 
        instance: Any, 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> Any:
        span = tracer.start_span(
            SDK_CLIENT_SPAN_NAME,
            kind=SpanKind.CLIENT
        )
        prompt_index = 0
        try:
            with trace.use_span(span, end_on_exit=False):
                if hasattr(instance, "_prompt_data"):
                   prompt_index = set_request_attributes(span, instance._prompt_data, prompt_index)
                   
            with trace.use_span(span, end_on_exit=False):
                aiterator = aiter(wrapped(*args, **kwargs))

            while True:
                with trace.use_span(span, end_on_exit=False):
                    try:
                        message = await anext(aiterator)
                        prompt_index = set_response_message_attributes(span, message, prompt_index)
                    except StopAsyncIteration:
                        break
                yield message

        except Exception as e:
            logger.error("netra.instrumentation.claude-agent-sdk: %s", e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()

    return wrapper