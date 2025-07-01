import logging
import os
from typing import Any, Collection, Dict, Generator, Iterator, List, Optional, Union

from opentelemetry import context as context_api
from opentelemetry.instrumentation.cohere.config import Config
from opentelemetry.instrumentation.cohere.utils import dont_throw
from opentelemetry.instrumentation.cohere.version import __version__
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

logger = logging.getLogger(__name__)

_instruments = ("cohere >=4.2.7, <6",)

WRAPPED_METHODS = [
    {
        "object": "ClientV2",
        "method": "chat",
        "span_name": "cohere.chat",
    },
    {
        "object": "ClientV2",
        "method": "chat_stream",
        "span_name": "cohere.chat_stream",
    },
    {
        "object": "ClientV2",
        "method": "rerank",
        "span_name": "cohere.rerank",
    },
    {
        "object": "AsyncClientV2",
        "method": "chat",
        "span_name": "cohere.async.chat",
    },
    {
        "object": "AsyncClientV2",
        "method": "rerank",
        "span_name": "cohere.async.rerank",
    },
]


def should_send_prompts() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or context_api.get_value(
        "override_enable_content_tracing"
    )


def _set_span_attribute(span: Span, name: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


@dont_throw  # type: ignore[misc]
def _set_input_attributes(span: Span, llm_request_type: LLMRequestTypeValues, kwargs: Dict[str, Any]) -> None:
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample"))
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
                    if hasattr(message, "content"):
                        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "user")
                        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", message.content)
                    elif isinstance(message, dict):
                        _set_span_attribute(
                            span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", message.get("role", "user")
                        )
                        _set_span_attribute(
                            span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", message.get("content")
                        )
            else:
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content", kwargs.get("message"))
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            documents = kwargs.get("documents", [])
            for index, document in enumerate(documents):
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.role", "system")
                _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{index}.content", document)

            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{len(documents)}.role",
                "user",
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.{len(documents)}.content",
                kwargs.get("query"),
            )

    return


def _set_span_chat_response(span: Span, response: Any) -> None:
    index = 0
    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"

    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)

    if hasattr(response, "message") and hasattr(response.message, "content"):
        text_content = []
        for content_item in response.message.content:
            if hasattr(content_item, "text"):
                text_content.append(content_item.text)
        if text_content:
            _set_span_attribute(span, f"{prefix}.content", "\n".join(text_content))
            _set_span_attribute(span, f"{prefix}.role", "assistant")

    if not hasattr(response, "usage") or response.usage is None:
        logger.debug("No usage information found in response")
        return

    logger.debug(f"Response usage object: {response.usage}")

    input_tokens = None
    output_tokens = None

    if hasattr(response.usage, "billed_units") and response.usage.billed_units is not None:
        logger.debug(f"Found billed_units: {response.usage.billed_units}")
        if (
            hasattr(response.usage.billed_units, "input_tokens")
            and response.usage.billed_units.input_tokens is not None
        ):
            input_tokens = int(float(response.usage.billed_units.input_tokens))
            logger.debug(f"Extracted input_tokens from billed_units: {input_tokens}")
        if (
            hasattr(response.usage.billed_units, "output_tokens")
            and response.usage.billed_units.output_tokens is not None
        ):
            output_tokens = int(float(response.usage.billed_units.output_tokens))
            logger.debug(f"Extracted output_tokens from billed_units: {output_tokens}")

    if input_tokens is not None:
        logger.debug(f"Setting {SpanAttributes.LLM_USAGE_PROMPT_TOKENS} to {input_tokens}")
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)

    if output_tokens is not None:
        logger.debug(f"Setting {SpanAttributes.LLM_USAGE_COMPLETION_TOKENS} to {output_tokens}")
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)

    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
        logger.debug(f"Setting {SpanAttributes.LLM_USAGE_TOTAL_TOKENS} to {total_tokens}")
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
        logger.info(
            f"Successfully set token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}"
        )
    else:
        logger.warning(f"Could not extract complete token usage - Input: {input_tokens}, Output: {output_tokens}")


def _set_span_generations_response(span: Span, response: Any) -> None:
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)
    if hasattr(response, "generations"):
        generations = response.generations  # Cohere v5
    else:
        generations = response  # Cohere v4

    for index, generation in enumerate(generations):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.content", generation.text)
        _set_span_attribute(span, f"gen_ai.response.{index}.id", generation.id)


def _set_span_rerank_response(span: Span, response: Any) -> None:
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response.id)
    for idx, doc in enumerate(response.results):
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{idx}"
        _set_span_attribute(span, f"{prefix}.role", "assistant")
        content = f"Doc {doc.index}, Score: {doc.relevance_score}"
        if doc.document:
            if hasattr(doc.document, "text"):
                content += f"\n{doc.document.text}"
            else:
                content += f"\n{doc.document.get('text')}"
        _set_span_attribute(
            span,
            f"{prefix}.content",
            content,
        )


@dont_throw  # type: ignore[misc]
def _set_response_attributes(span: Span, llm_request_type: LLMRequestTypeValues, response: Any) -> None:

    if should_send_prompts():
        if llm_request_type == LLMRequestTypeValues.CHAT:
            _set_span_chat_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.COMPLETION:
            _set_span_generations_response(span, response)
        elif llm_request_type == LLMRequestTypeValues.RERANK:
            _set_span_rerank_response(span, response)


def _with_tracer_wrapper(func: Any) -> Any:
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer: Tracer, to_wrap: Dict[str, str]) -> Any:
        def wrapper(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _llm_request_type_by_method(method_name: Optional[str]) -> LLMRequestTypeValues:
    if method_name in ["chat", "chat_stream"]:
        return LLMRequestTypeValues.CHAT
    elif method_name == "generate":
        return LLMRequestTypeValues.COMPLETION
    elif method_name == "rerank":
        return LLMRequestTypeValues.RERANK
    else:
        return LLMRequestTypeValues.UNKNOWN


def _build_from_streaming_response(
    span: Span, response: Iterator[Any], llm_request_type: LLMRequestTypeValues, context_token: Any
) -> Generator[Any, None, None]:
    """Build response from streaming events and set span attributes."""
    response_id = None
    content_parts = []
    usage_info = None

    try:
        for event in response:
            if hasattr(event, "type"):
                if event.type == "message-start" and hasattr(event, "id"):
                    response_id = event.id

                elif event.type == "content-delta":
                    if (
                        hasattr(event, "delta")
                        and hasattr(event.delta, "message")
                        and hasattr(event.delta.message, "content")
                        and hasattr(event.delta.message.content, "text")
                    ):
                        content_parts.append(event.delta.message.content.text)

                elif event.type == "message-end":
                    if hasattr(event, "delta") and hasattr(event.delta, "usage"):
                        usage_info = event.delta.usage

            yield event

        if response_id:
            _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

        if should_send_prompts() and content_parts:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
            full_content = "".join(content_parts)
            _set_span_attribute(span, f"{prefix}.content", full_content)
            _set_span_attribute(span, f"{prefix}.role", "assistant")

        if usage_info and hasattr(usage_info, "billed_units") and usage_info.billed_units:
            input_tokens = None
            output_tokens = None

            if hasattr(usage_info.billed_units, "input_tokens") and usage_info.billed_units.input_tokens is not None:
                input_tokens = int(float(usage_info.billed_units.input_tokens))

            if hasattr(usage_info.billed_units, "output_tokens") and usage_info.billed_units.output_tokens is not None:
                output_tokens = int(float(usage_info.billed_units.output_tokens))

            if input_tokens is not None:
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
            if output_tokens is not None:
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens)
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
                _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

        span.set_status(Status(StatusCode.OK))

    except Exception:
        span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        span.end()
        context_api.detach(context_token)


@_with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap: Dict[str, str], wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    method_name = to_wrap.get("method")
    llm_request_type = _llm_request_type_by_method(method_name)

    if method_name == "chat_stream":
        span = tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Cohere",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        )

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        try:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

            response = wrapped(*args, **kwargs)

            if response:
                return _build_from_streaming_response(span, response, llm_request_type, token)
            else:
                span.set_status(Status(StatusCode.ERROR))
                span.end()
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
                SpanAttributes.LLM_SYSTEM: "Cohere",
                SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
            },
        ) as span:
            if span.is_recording():
                _set_input_attributes(span, llm_request_type, kwargs)

            response = wrapped(*args, **kwargs)

            if response:
                if span.is_recording():
                    _set_response_attributes(span, llm_request_type, response)
                    span.set_status(Status(StatusCode.OK))

            return response


@_with_tracer_wrapper
async def _async_wrap(
    tracer: Tracer, to_wrap: Dict[str, str], wrapped: Any, instance: Any, args: Any, kwargs: Any
) -> Any:
    """Instruments and calls every async function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    method_name = to_wrap.get("method")
    llm_request_type = _llm_request_type_by_method(method_name)

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Cohere",
            SpanAttributes.LLM_REQUEST_TYPE: llm_request_type.value,
        },
    ) as span:
        if span.is_recording():
            _set_input_attributes(span, llm_request_type, kwargs)

        response = await wrapped(*args, **kwargs)

        if response:
            if span.is_recording():
                _set_response_attributes(span, llm_request_type, response)
                span.set_status(Status(StatusCode.OK))

        return response


class CohereInstrumentor(BaseInstrumentor):  # type: ignore
    """An instrumentor for Cohere's client library."""

    def __init__(self, exception_logger: Optional[Any] = None) -> None:
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            # Use async wrapper for AsyncClientV2
            if wrap_object == "AsyncClientV2":
                wrap_function_wrapper(
                    "cohere",
                    f"{wrap_object}.{wrap_method}",
                    _async_wrap(tracer, wrapped_method),
                )
            else:
                wrap_function_wrapper(
                    "cohere",
                    f"{wrap_object}.{wrap_method}",
                    _wrap(tracer, wrapped_method),
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        for wrapped_method in WRAPPED_METHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"cohere.{wrap_object}",
                wrapped_method.get("method"),
            )
