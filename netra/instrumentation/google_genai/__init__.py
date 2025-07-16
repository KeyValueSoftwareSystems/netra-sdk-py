"""OpenTelemetry Google GenAI API instrumentation"""

import logging
import os
import types
from typing import Any, Callable, Collection, Dict, Optional, Tuple, Union

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    SpanAttributes,
)
from opentelemetry.trace import SpanKind, get_tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

from netra.instrumentation.google_genai.config import Config
from netra.instrumentation.google_genai.utils import dont_throw
from netra.instrumentation.google_genai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("google-genai >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content",
        "span_name": "genai.generate_content",
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content_stream",
        "span_name": "genai.generate_content_stream",
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_images",
        "span_name": "genai.generate_images",
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_videos",
        "span_name": "genai.generate_videos",
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content",
        "span_name": "genai.generate_content_async",
        "is_async": True,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content_stream",
        "span_name": "genai.generate_content_stream_async",
        "is_async": True,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_images",
        "span_name": "genai.generate_images_async",
        "is_async": True,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_videos",
        "span_name": "genai.generate_videos_async",
        "is_async": True,
    },
]


def should_send_prompts() -> bool:
    return (os.getenv("TRACELOOP_TRACE_CONTENT") or "true").lower() == "true" or context_api.get_value(
        "override_enable_content_tracing"
    )


def is_streaming_response(response: Any) -> bool:
    return isinstance(response, types.GeneratorType)


def is_async_streaming_response(response: Any) -> bool:
    return isinstance(response, types.AsyncGeneratorType)


def _set_span_attribute(span: Any, name: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_input_attributes(span: Any, args: tuple[Any, ...], kwargs: dict[str, Any], llm_model: str) -> None:
    if not should_send_prompts():
        return

    # Handle contents parameter
    if "contents" in kwargs:
        contents = kwargs["contents"]
        if isinstance(contents, str):
            # Simple string content
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                contents,
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )
        elif isinstance(contents, list):
            # List of content objects
            for i, content in enumerate(contents):
                if hasattr(content, "parts"):
                    for part in content.parts:
                        if hasattr(part, "text"):
                            _set_span_attribute(
                                span,
                                f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                                part.text,
                            )
                            _set_span_attribute(
                                span,
                                f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                                getattr(content, "role", "user"),
                            )
                elif isinstance(content, str):
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                        content,
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{i}.role",
                        "user",
                    )
    elif args and len(args) > 0:
        # Handle positional arguments
        prompt = ""
        for arg in args:
            if isinstance(arg, str):
                prompt = f"{prompt}{arg}\n"
            elif isinstance(arg, list):
                for subarg in arg:
                    prompt = f"{prompt}{subarg}\n"
        if prompt:
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.content",
                prompt,
            )
            _set_span_attribute(
                span,
                f"{SpanAttributes.LLM_PROMPTS}.0.role",
                "user",
            )

    # Extract model from kwargs or args
    model_name = kwargs.get("model", "unknown")
    if model_name != "unknown":
        llm_model = model_name

    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, llm_model)

    # Handle config parameter which might contain generation settings
    if "config" in kwargs and kwargs["config"]:
        config = kwargs["config"]
        if hasattr(config, "temperature"):
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TEMPERATURE, config.temperature)
        if hasattr(config, "max_output_tokens"):
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, config.max_output_tokens)
        if hasattr(config, "top_p"):
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, config.top_p)
        if hasattr(config, "top_k"):
            _set_span_attribute(span, SpanAttributes.LLM_TOP_K, config.top_k)

    return


@dont_throw
def _set_response_attributes(span: Any, response: Any, llm_model: str) -> None:
    _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, llm_model)

    # Handle response attributes for google.genai package
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        if hasattr(usage, "total_token_count"):
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
                usage.total_token_count,
            )
        if hasattr(usage, "candidates_token_count"):
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                usage.candidates_token_count,
            )
        if hasattr(usage, "prompt_token_count"):
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                usage.prompt_token_count,
            )

    # Handle response text
    if hasattr(response, "text") and response.text:
        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response.text)
        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant")
    elif hasattr(response, "candidates") and response.candidates:
        for index, candidate in enumerate(response.candidates):
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text"):
                        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.content", part.text)
                        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{index}.role", "assistant")

    return


def _handle_request(span: Any, args: tuple[Any, ...], kwargs: dict[str, Any], llm_model: str) -> None:
    if span.is_recording():
        _set_input_attributes(span, args, kwargs, llm_model)


@dont_throw
def _handle_response(span: Any, response: Any, llm_model: str) -> None:
    if span.is_recording():
        _set_response_attributes(span, response, llm_model)
        span.set_status(Status(StatusCode.OK))


def _with_tracer_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer: Any, to_wrap: dict[str, Any]) -> Callable[..., Any]:
        def wrapper(wrapped: Callable[..., Any], instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _build_from_streaming_response(span: Any, response: Any, llm_model: str, context_token: Any) -> Any:
    complete_response = ""
    try:
        for item in response:
            item_to_yield = item
            if hasattr(item, "text"):
                complete_response += str(item.text)
            yield item_to_yield

        _set_response_attributes(span, complete_response, llm_model)
        span.set_status(Status(StatusCode.OK))
    except Exception:
        span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        span.end()
        context_api.detach(context_token)


async def _abuild_from_streaming_response(span: Any, response: Any, llm_model: str, context_token: Any) -> Any:
    complete_response = ""
    try:
        async for item in response:
            item_to_yield = item
            if hasattr(item, "text"):
                complete_response += str(item.text)
            yield item_to_yield

        _set_response_attributes(span, complete_response, llm_model)
        span.set_status(Status(StatusCode.OK))
    except Exception:
        span.set_status(Status(StatusCode.ERROR))
        raise
    finally:
        span.end()
        context_api.detach(context_token)


@_with_tracer_wrapper
async def _awrap(
    tracer: Any,
    to_wrap: dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    llm_model = kwargs.get("model", "unknown")
    if llm_model != "unknown":
        llm_model = llm_model.replace("models/", "")

    name = to_wrap.get("span_name")
    method_name = to_wrap.get("method")

    if method_name == "generate_content_stream":
        span = tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Gemini",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
            },
        )

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        try:
            _handle_request(span, args, kwargs, llm_model)

            response = await wrapped(*args, **kwargs)

            if response:
                if is_streaming_response(response):
                    return _build_from_streaming_response(span, response, llm_model, token)
                elif is_async_streaming_response(response):
                    return _abuild_from_streaming_response(span, response, llm_model, token)
                else:
                    _handle_response(span, response, llm_model)
                    span.end()
                    context_api.detach(token)
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
                SpanAttributes.LLM_SYSTEM: "Gemini",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
            },
        ) as span:

            _handle_request(span, args, kwargs, llm_model)

            response = await wrapped(*args, **kwargs)
            ctx = set_span_in_context(span)
            token = context_api.attach(ctx)

            if response:
                if is_streaming_response(response):
                    return _build_from_streaming_response(span, response, llm_model, token)
                elif is_async_streaming_response(response):
                    return _abuild_from_streaming_response(span, response, llm_model, token)
                else:
                    _handle_response(span, response, llm_model)

            return response


@_with_tracer_wrapper
def _wrap(
    tracer: Any,
    to_wrap: dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    llm_model = kwargs.get("model", "unknown")
    if llm_model != "unknown":
        llm_model = llm_model.replace("models/", "")

    name = to_wrap.get("span_name")
    method_name = to_wrap.get("method")

    if method_name == "generate_content_stream":
        span = tracer.start_span(
            name,
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.LLM_SYSTEM: "Gemini",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
            },
        )

        ctx = set_span_in_context(span)
        token = context_api.attach(ctx)

        try:
            _handle_request(span, args, kwargs, llm_model)

            response = wrapped(*args, **kwargs)

            if response:
                if is_streaming_response(response):
                    return _build_from_streaming_response(span, response, llm_model, token)
                elif is_async_streaming_response(response):
                    return _abuild_from_streaming_response(span, response, llm_model, token)
                else:
                    _handle_response(span, response, llm_model)
                    span.end()
                    context_api.detach(token)
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
                SpanAttributes.LLM_SYSTEM: "Gemini",
                SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
            },
        ) as span:

            _handle_request(span, args, kwargs, llm_model)
            response = wrapped(*args, **kwargs)
            ctx = set_span_in_context(span)
            token = context_api.attach(ctx)
            if response:
                if is_streaming_response(response):
                    return _build_from_streaming_response(span, response, llm_model, token)
                elif is_async_streaming_response(response):
                    return _abuild_from_streaming_response(span, response, llm_model, token)
                else:
                    _handle_response(span, response, llm_model)
            return response


class GoogleGenAiInstrumentor(BaseInstrumentor):  # type: ignore
    """An instrumentor for Google GenAI's client library."""

    def __init__(self, exception_logger: Optional[Callable[[Exception], None]]) -> None:
        # Initialize the parent class
        super().__init__()
        # Set the exception logger in Config
        if exception_logger is not None:
            Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                (_awrap(tracer, wrapped_method) if wrapped_method.get("is_async") else _wrap(tracer, wrapped_method)),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method", ""),
            )
