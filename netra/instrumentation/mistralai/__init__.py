"""OpenTelemetry Mistral AI instrumentation"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Callable, Collection, Dict, Generator, Optional, Tuple, Union

from mistralai import AssistantMessage, ChatCompletionChoice, UsageInfo
from mistralai.models import ChatCompletionResponse
from opentelemetry import context as context_api
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
from opentelemetry.trace import SpanKind, get_tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

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


def should_send_prompts() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or context_api.get_value(
        "override_enable_content_tracing"
    )


def _set_span_attribute(span: Any, name: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw
def _set_input_attributes(
    span: Any, llm_request_type: LLMRequestTypeValues, to_wrap: Dict[str, Any], kwargs: dict[str, Any]
) -> None:
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
                if hasattr(message, "content"):
                    content = message.content
                    role = message.role
                else:
                    content = message.get("content", "")
                    role = message.get("role", "user")

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
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", input_data)
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
def _set_response_attributes(span: Any, llm_request_type: LLMRequestTypeValues, response: Any) -> None:
    # Handle both object and dict response formats
    response_id = getattr(response, "id", None) or response.get("id") if hasattr(response, "get") else None
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

    if llm_request_type == LLMRequestTypeValues.EMBEDDING:
        return

    if should_send_prompts():
        choices = getattr(response, "choices", None) or response.get("choices", []) if hasattr(response, "get") else []
        for index, choice in enumerate(choices):
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

            # Handle both object and dict choice formats
            if hasattr(choice, "finish_reason"):
                finish_reason = choice.finish_reason
                message = choice.message
            else:
                finish_reason = choice.get("finish_reason")
                message = choice.get("message", {})

            _set_span_attribute(
                span,
                f"{prefix}.finish_reason",
                finish_reason,
            )

            # Handle message content
            if hasattr(message, "content"):
                content = message.content
                role = message.role
            else:
                content = message.get("content", "")
                role = message.get("role", "assistant")

            _set_span_attribute(
                span,
                f"{prefix}.content",
                (content if isinstance(content, str) else json.dumps(content)),
            )
            _set_span_attribute(
                span,
                f"{prefix}.role",
                role,
            )

    # Handle model attribute
    if hasattr(response, "model"):
        model = response.model
        _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, model)

    # Handle usage information
    if not hasattr(response, "usage"):
        return

    usage = response.usage

    if hasattr(usage, "prompt_tokens"):
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens
    else:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

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


def _accumulate_streaming_response(
    span: Any, llm_request_type: LLMRequestTypeValues, response: Any, token: Any = None
) -> Generator[Any, None, None]:
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    try:
        for res in response:
            yield res

            data = None
            if hasattr(res, "data"):
                data = res.data

            if data is not None and hasattr(data, "model") and data.model:
                accumulated_response.model = data.model
            if data is not None and hasattr(data, "usage") and data.usage:
                accumulated_response.usage = data.usage
            # ID is the same for all chunks, so it's safe to overwrite it every time
            if data is not None and hasattr(data, "id") and data.id:
                accumulated_response.id = data.id

            choices = getattr(data, "choices", [])
            for idx, choice in enumerate(choices):
                if len(accumulated_response.choices) <= idx:
                    accumulated_response.choices.append(
                        ChatCompletionChoice(
                            index=idx,
                            message=AssistantMessage(role="assistant", content=""),
                            finish_reason=choice.finish_reason,
                        )
                    )

                if hasattr(choice, "finish_reason"):
                    accumulated_response.choices[idx].finish_reason = choice.finish_reason

                # Handle delta content
                delta = getattr(choice, "delta", None)
                if delta:
                    if hasattr(delta, "content") and delta.content:
                        accumulated_response.choices[idx].message.content += delta.content
                    if hasattr(delta, "role") and delta.role:
                        accumulated_response.choices[idx].message.role = delta.role

        _set_response_attributes(span, llm_request_type, accumulated_response)
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end()
        if token is not None:
            context_api.detach(token)


async def _aaccumulate_streaming_response(
    span: Any, llm_request_type: LLMRequestTypeValues, response: Any, token: Any = None
) -> AsyncGenerator[Any, None]:
    accumulated_response = ChatCompletionResponse(
        id="",
        object="",
        created=0,
        model="",
        choices=[],
        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )

    try:
        async for res in response:
            yield res

            data = None
            if hasattr(res, "data"):
                data = res.data

            if data is not None and hasattr(data, "model") and data.model:
                accumulated_response.model = data.model
            if data is not None and hasattr(data, "usage") and data.usage:
                accumulated_response.usage = data.usage
            # Id is the same for all chunks, so it's safe to overwrite it every time
            if data is not None and hasattr(data, "id") and data.id:
                accumulated_response.id = data.id

            choices = getattr(data, "choices", [])
            for idx, choice in enumerate(choices):
                if len(accumulated_response.choices) <= idx:
                    accumulated_response.choices.append(
                        ChatCompletionChoice(
                            index=idx,
                            message=AssistantMessage(role="assistant", content=""),
                            finish_reason=choice.finish_reason,
                        )
                    )

                if hasattr(choice, "finish_reason"):
                    accumulated_response.choices[idx].finish_reason = choice.finish_reason

                # Handle delta content
                delta = getattr(choice, "delta", None)
                if delta:
                    if hasattr(delta, "content") and delta.content:
                        accumulated_response.choices[idx].message.content += delta.content
                    if hasattr(delta, "role") and delta.role:
                        accumulated_response.choices[idx].message.role = delta.role

        _set_response_attributes(span, llm_request_type, accumulated_response)
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end()
        if token is not None:
            context_api.detach(token)


def _with_tracer_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer: Any, to_wrap: Dict[str, Any]) -> Callable[..., Any]:
        def wrapper(wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name: Optional[str]) -> LLMRequestTypeValues:
    if method_name in ["complete", "complete_async", "stream", "stream_async"]:
        return LLMRequestTypeValues.CHAT
    elif method_name == "create" and "embeddings" in method_name:
        return LLMRequestTypeValues.EMBEDDING
    else:
        return LLMRequestTypeValues.UNKNOWN


@_with_tracer_wrapper
def _wrap(
    tracer: Any,
    to_wrap: Dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))

    if to_wrap.get("streaming"):
        span = tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "MistralAI",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        )

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

            response = wrapped(*args, **kwargs)

            if response:
                return _accumulate_streaming_response(span, llm_request_type, response, token)
            else:
                span.set_status(Status(StatusCode.ERROR))
                span.end()
                context_api.detach(token)

            return response
        except Exception:
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            context_api.detach(token)
            raise
    else:
        with tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "MistralAI",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        ) as span:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

            response = wrapped(*args, **kwargs)

            if response:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)
                    span.set_status(Status(StatusCode.OK))

            return response


@_with_tracer_wrapper
async def _awrap(
    tracer: Any,
    to_wrap: Dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    llm_request_type = _llm_request_type_by_method(to_wrap.get("method"))

    if to_wrap.get("streaming"):
        span = tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "MistralAI",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        )

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

            response = await wrapped(*args, **kwargs)

            if response:
                return _aaccumulate_streaming_response(span, llm_request_type, response, token)
            else:
                span.set_status(Status(StatusCode.ERROR))
                span.end()
                context_api.detach(token)

            return response
        except Exception:
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            context_api.detach(token)
            raise
    else:
        with tracer.start_as_current_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "MistralAI",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        ) as span:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, to_wrap, kwargs)

            response = await wrapped(*args, **kwargs)

            if response:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)
                    span.set_status(Status(StatusCode.OK))

            return response


class MistralAiInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for Mistral AI's client library."""

    def __init__(self, exception_logger: Optional[Callable[[Exception], None]] = None) -> None:
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
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

    def _uninstrument(self, **kwargs: Any) -> None:
        for wrapped_method in WRAPPED_METHODS:
            module_name = wrapped_method.get("module")
            object_name = wrapped_method.get("object")
            method_name = wrapped_method.get("method")

            unwrap(
                f"{module_name}.{object_name}",
                method_name,
            )
