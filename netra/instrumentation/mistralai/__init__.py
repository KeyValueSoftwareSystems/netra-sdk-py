"""OpenTelemetry Mistral AI instrumentation"""

import logging
import os
import json
from typing import Collection

from mistralai import UsageInfo, ChatCompletionChoice, AssistantMessage
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_ID
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)

from mistralai.models import ChatCompletionResponse

from netra.instrumentation.mistralai.config import Config
from netra.instrumentation.mistralai.utils import dont_throw
from netra.instrumentation.mistralai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("mistralai >= 1.0.0",)

WRAPPED_METHODS = [
    {
        "module": "mistralai.chat",
        "object": "Chat",
        "method": "complete",
        "span_name": "mistralai.chat.complete",
        "streaming": False,
        "is_async": False,
    },
    {
        "module": "mistralai.chat",
        "object": "Chat",
        "method": "complete_async",
        "span_name": "mistralai.chat.complete_async",
        "streaming": False,
        "is_async": True,
    },
    {
        "module": "mistralai.chat",
        "object": "Chat",
        "method": "stream",
        "span_name": "mistralai.chat.stream",
        "streaming": True,
        "is_async": False,
    },
    {
        "module": "mistralai.chat",
        "object": "Chat",
        "method": "stream_async",
        "span_name": "mistralai.chat.stream_async",
        "streaming": True,
        "is_async": True,
    },
    {
        "module": "mistralai.embeddings",
        "object": "Embeddings",
        "method": "create",
        "span_name": "mistralai.embeddings",
        "streaming": False,
        "is_async": False,
    },
]


def should_send_prompts():
    return (
            os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def _set_input_attributes(span, llm_request_type, to_wrap, kwargs):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(
        span,
        SpanAttributes.LLM_IS_STREAMING,
        to_wrap.get("streaming"),
    )

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            messages = kwargs.get("messages", [])
            for index, message in enumerate(messages):
                # Handle both dict and object message formats
                if hasattr(message, 'content'):
                    content = message.content
                    role = message.role
                else:
                    content = message.get('content', '')
                    role = message.get('role', 'user')

                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                    content,
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                    role,
                )
        else:
            input_data = kwargs.get("input") or kwargs.get("inputs")

            if isinstance(input_data, str):
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user"
                )
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.0.content", input_data
                )
            elif isinstance(input_data, list):
                for index, prompt in enumerate(input_data):
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{index}.role",
                        "user",
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{index}.content",
                        str(prompt),
                    )


@dont_throw
def _set_response_attributes(span, llm_request_type, response):
    # Handle both object and dict response formats
    response_id = getattr(response, 'id', None) or response.get('id') if hasattr(response, 'get') else None
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

    if llm_request_type == LLMRequestTypeValues.EMBEDDING:
        return

    if should_send_prompts():
        choices = getattr(response, 'choices', None) or response.get('choices', []) if hasattr(response, 'get') else []
        for index, choice in enumerate(choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

            # Handle both object and dict choice formats
            if hasattr(choice, 'finish_reason'):
                finish_reason = choice.finish_reason
                message = choice.message
            else:
                finish_reason = choice.get('finish_reason')
                message = choice.get('message', {})

            _set_span_attribute(
                span,
                f"{prefix}.finish_reason",
                finish_reason,
            )

            # Handle message content
            if hasattr(message, 'content'):
                content = message.content
                role = message.role
            else:
                content = message.get('content', '')
                role = message.get('role', 'assistant')

            _set_span_attribute(
                span,
                f"{prefix}.content",
                (
                    content
                    if isinstance(content, str)
                    else json.dumps(content)
                ),
            )
            _set_span_attribute(
                span,
                f"{prefix}.role",
                role,
            )

    # Handle model attribute
    if hasattr(response, 'model'):
        model = response.model
        _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)

    # Handle usage information
    if not hasattr(response, 'usage'):
        return

    usage = response.usage

    if hasattr(usage, 'prompt_tokens'):
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens
    else:
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
        total_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
        output_tokens,
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
        input_tokens,
    )


def _accumulate_streaming_response(span, llm_request_type, response):
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    for res in response:
        yield res

        data = None
        if hasattr(res, 'data'):
            data = res.data

        if hasattr(data, 'model') and data.model:
            accumulated_response.model = data.model
        if hasattr(data, 'usage') and data.usage:
            accumulated_response.usage = data.usage
        # ID is the same for all chunks, so it's safe to overwrite it every time
        if hasattr(data, 'id') and data.id:
            accumulated_response.id = data.id

        choices = getattr(data, 'choices', [])
        for idx, choice in enumerate(choices):
            if len(accumulated_response.choices) <= idx:
                accumulated_response.choices.append(
                    ChatCompletionChoice(
                        index=idx,
                        message=AssistantMessage(role="assistant", content=""),
                        finish_reason=choice.finish_reason,
                    )
                )

            if hasattr(choice, 'finish_reason'):
                accumulated_response.choices[idx].finish_reason = choice.finish_reason

            # Handle delta content
            delta = getattr(choice, 'delta', None)
            if delta:
                if hasattr(delta, 'content') and delta.content:
                    accumulated_response.choices[idx].message.content += delta.content
                if hasattr(delta, 'role') and delta.role:
                    accumulated_response.choices[idx].message.role = delta.role

    _set_response_attributes(span, llm_request_type, accumulated_response)
    span.end()


async def _aaccumulate_streaming_response(span, llm_request_type, response):
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    async for res in response:
        yield res

        data = None
        if hasattr(res, 'data'):
            data = res.data

        if hasattr(data, 'model') and data.model:
            accumulated_response.model = data.model
        if hasattr(data, 'usage') and data.usage:
            accumulated_response.usage = data.usage
        # Id is the same for all chunks, so it's safe to overwrite it every time
        if hasattr(data, 'id') and data.id:
            accumulated_response.id = data.id

        choices = getattr(data, 'choices', [])
        for idx, choice in enumerate(choices):
            if len(accumulated_response.choices) <= idx:
                accumulated_response.choices.append(
                    ChatCompletionChoice(
                        index=idx,
                        message=AssistantMessage(role="assistant", content=""),
                        finish_reason=choice.finish_reason,
                    )
                )

            if hasattr(choice, 'finish_reason'):
                accumulated_response.choices[idx].finish_reason = choice.finish_reason

            # Handle delta content
            delta = getattr(choice, 'delta', None)
            if delta:
                if hasattr(delta, 'content') and delta.content:
                    accumulated_response.choices[idx].message.content += delta.content
                if hasattr(delta, 'role') and delta.role:
                    accumulated_response.choices[idx].message.role = delta.role

    _set_response_attributes(span, llm_request_type, accumulated_response)
    span.end()


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name):
    if method_name in ["complete", "complete_async", "stream", "stream_async"]:
        return LLMRequestTypeValues.CHAT
    elif method_name == "create" and "embeddings" in method_name:
        return LLMRequestTypeValues.EMBEDDING
    else:
        return LLMRequestTypeValues.UNKNOWN


@_with_tracer_wrapper
def _wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "MistralAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )
    if span.is_recording():
        _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

    response = wrapped(*args, **kwargs)

    if response:
        if span.is_recording():
            if to_wrap.get("streaming"):
                return _accumulate_streaming_response(span, llm_request_type, response)

            _set_response_attributes(span, llm_request_type, response)
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


@_with_tracer_wrapper
async def _awrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
            SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "MistralAI",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    )

    if span.is_recording():
        _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

    if to_wrap.get("streaming"):
        response = await wrapped(*args, **kwargs)
    else:
        response = await wrapped(*args, **kwargs)

    if response:
        if span.is_recording():
            if to_wrap.get("streaming"):
                return _aaccumulate_streaming_response(span, llm_request_type, response)

            _set_response_attributes(span, llm_request_type, response)
            span.set_status(Status(StatusCode.OK))

    span.end()
    return response


class MistralAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Mistral AI's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            module_name = wrapped_method.get("module")
            object_name = wrapped_method.get("object")
            method_name = wrapped_method.get("method")
            is_async = wrapped_method.get("is_async")

            wrapper_func = _awrap if is_async else _wrap

            wrap_function_wrapper(
                module_name,
                f"{object_name}.{method_name}",
                wrapper_func(tracer, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            module_name = wrapped_method.get("module")
            object_name = wrapped_method.get("object")
            method_name = wrapped_method.get("method")

            unwrap(
                f"{module_name}.{object_name}",
                method_name,
            )
