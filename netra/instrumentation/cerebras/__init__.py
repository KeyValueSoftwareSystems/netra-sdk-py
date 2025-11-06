import logging
import os
from typing import Any, Collection, Dict, Generator, Optional

from opentelemetry import context as context_api
from netra.instrumentation.cerebras.config import Config
from netra.instrumentation.cerebras.utils import dont_throw, should_send_prompts
from netra.instrumentation.cerebras.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_ID
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer, get_tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper
from functools import partial
import inspect

logger = logging.getLogger(__name__)

_instruments = ("cerebras-cloud-sdk >= 1.0.0",)

# CORRECTED: The actual module structure in Cerebras SDK
WRAPPED_METHODS = [
    {
        "package": "cerebras.cloud.sdk.resources.chat.completions",
        "object": "CompletionsResource",
        "method": "create",
        "span_name": "cerebras.chat.completions",
    },
    {
        "package": "cerebras.cloud.sdk.resources.completions",
        "object": "CompletionsResource",
        "method": "create",
        "span_name": "cerebras.completions",
    },
    {
        "package": "cerebras.cloud.sdk.resources.chat.completions",
        "object": "AsyncCompletionsResource",
        "method": "create",
        "span_name": "cerebras.chat.completions.async",
    },
    {
        "package": "cerebras.cloud.sdk.resources.completions",
        "object": "AsyncCompletionsResource",
        "method": "create",
        "span_name": "cerebras.completions.async",
    },
]


def _set_span_attribute(span: Span, name: str, value: Any) -> None:
    if value is not None and value != "":
        # Convert non-primitive to string to avoid attribute type errors
        try:
            span.set_attribute(name, value)
        except Exception:
            try:
                span.set_attribute(name, str(value))
            except Exception:
                # best-effort only
                pass


@dont_throw
def _set_input_attributes(span: Span, llm_request_type: LLMRequestTypeValues, kwargs: Dict[str, Any]) -> None:
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
    _set_span_attribute(span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty"))

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("prompt"))
        elif llm_request_type == LLMRequestTypeValues.CHAT:
            messages = kwargs.get("messages")
            if messages:
                for index, message in enumerate(messages):
                    if isinstance(message, dict):
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                            message.get("role", "user")
                        )
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                            message.get("content")
                        )
                    elif hasattr(message, "role") and hasattr(message, "content"):
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                            message.role
                        )
                        _set_span_attribute(
                            span,
                            f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                            message.content
                        )


@dont_throw
def _set_span_chat_completion_response(span: Span, response: Any) -> None:
    if hasattr(response, "id"):
        _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)
    if hasattr(response, "choices") and response.choices:
        for index, choice in enumerate(response.choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            if hasattr(choice, "message"):
                _set_span_attribute(span, f"{prefix}.role", getattr(choice.message, "role", "assistant"))
                _set_span_attribute(span, f"{prefix}.content", getattr(choice.message, "content", ""))
            if hasattr(choice, "finish_reason"):
                _set_span_attribute(span, f"{prefix}.finish_reason", choice.finish_reason)
    if hasattr(response, "usage"):
        usage = response.usage
        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
            if hasattr(usage, "completion_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)
            if hasattr(usage, "total_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)
        elif isinstance(usage, dict):
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens"))
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))


@dont_throw
def _set_span_completion_response(span: Span, response: Any) -> None:
    if hasattr(response, "id"):
        _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)
    if hasattr(response, "choices") and response.choices:
        for index, choice in enumerate(response.choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            if hasattr(choice, "text"):
                _set_span_attribute(span, f"{prefix}.content", choice.text)
            if hasattr(choice, "finish_reason"):
                _set_span_attribute(span, f"{prefix}.finish_reason", choice.finish_reason)
    if hasattr(response, "usage"):
        usage = response.usage
        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
            if hasattr(usage, "completion_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)
            if hasattr(usage, "total_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)
        elif isinstance(usage, dict):
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.get("completion_tokens"))
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))


@dont_throw
def _set_response_attributes(span: Span, llm_request_type: LLMRequestTypeValues, response: Any) -> None:
    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_completion_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_completion_response(span, response)


def _llm_request_type_by_method(package_name: Optional[str]) -> LLMRequestTypeValues:
    if package_name and "chat" in package_name:
        return LLMRequestTypeValues.CHAT
    elif package_name and "completions" in package_name:
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def _handle_streaming_response(
    span: Span,
    response: Any,
    llm_request_type: LLMRequestTypeValues,
    ctx_token: Optional[Any] = None,
) -> Generator[Any, None, None]:
    """Handle streaming response for synchronous calls. Ensures context is detached."""
    response_id = None
    content_parts = []
    usage_info = None
    finish_reason = None

    try:
        for chunk in response:
            if hasattr(chunk, "id") and not response_id:
                response_id = chunk.id

            if hasattr(chunk, "choices") and chunk.choices:
                for choice in chunk.choices:
                    if llm_request_type == LLMRequestTypeValues.CHAT:
                        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                            content = choice.delta.content
                            if content:
                                content_parts.append(content)
                    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
                        if hasattr(choice, "text"):
                            text = choice.text
                            if text:
                                content_parts.append(text)

                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

            yield chunk

        # Set span attributes after streaming completes
        if response_id:
            _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

        if should_send_prompts() and content_parts:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
            full_content = "".join(content_parts)
            _set_span_attribute(span, f"{prefix}.content", full_content)
            _set_span_attribute(span, f"{prefix}.role", "assistant")

            if finish_reason:
                _set_span_attribute(span, f"{prefix}.finish_reason", finish_reason)

        if usage_info:
            if hasattr(usage_info, "prompt_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage_info.prompt_tokens)
            if hasattr(usage_info, "completion_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage_info.completion_tokens)
            if hasattr(usage_info, "total_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage_info.total_tokens)

        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(e)
        raise
    finally:
        try:
            span.end()
        except Exception:
            pass
        if ctx_token is not None:
            try:
                context_api.detach(ctx_token)
            except Exception:
                pass


@dont_throw
async def _handle_async_streaming_response(
    span: Span,
    response: Any,
    llm_request_type: LLMRequestTypeValues,
    ctx_token: Optional[Any] = None,
):
    """Handle streaming response for async calls. Ensures context is detached."""
    response_id = None
    content_parts = []
    usage_info = None
    finish_reason = None

    try:
        async for chunk in response:
            if hasattr(chunk, "id") and not response_id:
                response_id = chunk.id

            if hasattr(chunk, "choices") and chunk.choices:
                for choice in chunk.choices:
                    if llm_request_type == LLMRequestTypeValues.CHAT:
                        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                            content = choice.delta.content
                            if content:
                                content_parts.append(content)
                    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
                        if hasattr(choice, "text"):
                            text = choice.text
                            if text:
                                content_parts.append(text)

                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        finish_reason = choice.finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

            yield chunk

        # Set span attributes after streaming completes
        if response_id:
            _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

        if should_send_prompts() and content_parts:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
            full_content = "".join(content_parts)
            _set_span_attribute(span, f"{prefix}.content", full_content)
            _set_span_attribute(span, f"{prefix}.role", "assistant")

            if finish_reason:
                _set_span_attribute(span, f"{prefix}.finish_reason", finish_reason)

        if usage_info:
            if hasattr(usage_info, "prompt_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage_info.prompt_tokens)
            if hasattr(usage_info, "completion_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage_info.completion_tokens)
            if hasattr(usage_info, "total_tokens"):
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage_info.total_tokens)

        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(e)
        raise
    finally:
        try:
            span.end()
        except Exception:
            pass
        if ctx_token is not None:
            try:
                context_api.detach(ctx_token)
            except Exception:
                pass


def _wrap(tracer: Tracer, to_wrap: Dict[str, str], wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """Wrapper function for tracing Cerebras SDK calls."""
    # Determine if the wrapped callable is an async function
    is_async = inspect.iscoroutinefunction(wrapped)

    # Detect streaming robustly: kwargs or last positional boolean
    is_streaming = kwargs.get("stream", False)
    try:
        if not is_streaming and args:
            # if the SDK passes stream as a positional boolean (legacy), detect it
            last_pos = args[-1]
            if isinstance(last_pos, bool):
                is_streaming = is_streaming or last_pos is True
    except Exception:
        # best-effort only
        pass

    name = to_wrap.get("span_name")
    package_name = to_wrap.get("package")
    llm_request_type = _llm_request_type_by_method(package_name)

    # Check for suppression
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        # Return original semantics (async -> coroutine, sync -> value)
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Cerebras",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    ctx = set_span_in_context(span)
    ctx_token = context_api.attach(ctx)

    detached = False  # track if we've detached in finalize path

    def _finalize_span(status=Status(StatusCode.OK), error=None):
        nonlocal detached
        if error:
            try:
                span.set_status(Status(StatusCode.ERROR))
            except Exception:
                pass
            try:
                span.record_exception(error)
            except Exception:
                pass
        else:
            try:
                span.set_status(status)
            except Exception:
                pass

        if not span.is_recording():
            # detach context if not already detached
            if not detached and ctx_token is not None:
                try:
                    context_api.detach(ctx_token)
                except Exception:
                    pass
                detached = True
            return

        try:
            span.end()
        except Exception:
            pass

        if not detached and ctx_token is not None:
            try:
                context_api.detach(ctx_token)
            except Exception:
                pass
            detached = True

    try:
        # Set request input attributes
        if span.is_recording():
            _set_input_attributes(span, llm_request_type, kwargs)

        if is_async:
            async def async_wrapper():
                try:
                    response = await wrapped(*args, **kwargs)

                    if is_streaming:
                        # Span ends inside async streaming handler; pass ctx_token for proper detach
                        return _handle_async_streaming_response(span, response, llm_request_type, ctx_token)
                    else:
                        if span.is_recording():
                            _set_response_attributes(span, llm_request_type, response)
                        _finalize_span()
                        return response

                except Exception as e:
                    _finalize_span(Status(StatusCode.ERROR), e)
                    raise

            return async_wrapper()

        else:
            response = wrapped(*args, **kwargs)

            if is_streaming:
                # Span ends inside streaming handler; pass ctx_token for proper detach
                return _handle_streaming_response(span, response, llm_request_type, ctx_token)
            else:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)
                _finalize_span()
                return response

    except Exception as e:
        _finalize_span(Status(StatusCode.ERROR), e)
        raise


class CerebrasInstrumentor(BaseInstrumentor):
    """An instrumentor for Cerebras's client library."""

    def __init__(self, exception_logger: Optional[Any] = None) -> None:
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                module_path = package  # fully qualified already

                # Import the module
                try:
                    module_obj = __import__(module_path, fromlist=[wrap_object])
                except ImportError:
                    # Try alternative import path
                    module_obj = __import__(package, fromlist=[wrap_object])

                if hasattr(module_obj, wrap_object):
                    # Create wrapper with tracer and method info
                    wrapper = partial(_wrap, tracer, wrapped_method)

                    # Wrap the method
                    wrap_function_wrapper(
                        module_path,
                        f"{wrap_object}.{wrap_method}",
                        wrapper,
                    )
                    logger.info(f"Successfully instrumented {wrap_object}.{wrap_method} in {module_path}")
                else:
                    logger.warning(f"Skipped missing object {wrap_object} in {module_path}")
            except Exception as e:
                logger.error(
                    f"Error wrapping {wrap_object}.{wrap_method}: {e}", exc_info=True
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        for wrapped_method in WRAPPED_METHODS:
            package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                module_path = package

                unwrap(
                    __import__(module_path, fromlist=[wrap_object]),
                    f"{wrap_object}.{wrap_method}"
                )
            except Exception as e:
                logger.error(f"Error unwrapping {wrap_object}.{wrap_method}: {e}")
