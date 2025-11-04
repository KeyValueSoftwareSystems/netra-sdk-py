"""OpenTelemetry Groq instrumentation"""

import json
import logging
import os
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Union,
)

from groq._streaming import AsyncStream, Stream
from opentelemetry import context as context_api
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    error_metrics_attributes,
    model_as_dict,
    set_span_attribute,
    shared_metrics_attributes,
    should_send_prompts,
)
from opentelemetry.instrumentation.groq.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Counter, Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    LLMRequestTypeValues,
    Meters,
    SpanAttributes,
)
from opentelemetry.trace import Span, SpanKind, Tracer, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("groq >= 0.9.0",)

CONTENT_FILTER_KEY = "content_filter_results"

WRAPPED_METHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "span_name": "groq.chat",
    },
]
WRAPPED_AMETHODS = [
    {
        "package": "groq.resources.chat.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "span_name": "groq.chat",
    },
]


def is_streaming_response(response: object) -> bool:
    return isinstance(response, Stream) or isinstance(response, AsyncStream)


def _dump_content(content: Union[str, Iterable[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    json_serializable = []
    for item in content:
        if item.get("type") == "text":
            json_serializable.append({"type": "text", "text": item.get("text")})
        elif item.get("type") == "image":
            json_serializable.append(
                {
                    "type": "image",
                    "source": {
                        "type": item.get("source").get("type"),  # type:ignore[union-attr]
                        "media_type": item.get("source").get("media_type"),  # type:ignore[union-attr]
                        "data": str(item.get("source").get("data")),  # type:ignore[union-attr]
                    },
                }
            )
    return json.dumps(json_serializable)


@dont_throw  # type:ignore[misc]
def _set_input_attributes(span: Span, kwargs: Dict[str, Any]) -> None:
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample"))
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, SpanAttributes.LLM_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
    set_span_attribute(span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty"))
    set_span_attribute(span, SpanAttributes.LLM_IS_STREAMING, kwargs.get("stream") or False)

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", kwargs.get("prompt"))

        elif kwargs.get("messages") is not None:
            for i, message in enumerate(kwargs.get("messages")):  # type:ignore[arg-type]
                set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    _dump_content(message.get("content")),
                )
                set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", message.get("role"))


def _set_completions(span: Span, choices: Optional[Iterable[Dict[str, Any]]]) -> None:
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))

        if choice.get("content_filter_results"):
            set_span_attribute(
                span,
                f"{prefix}.{CONTENT_FILTER_KEY}",
                json.dumps(choice.get("content_filter_results")),
            )

        if choice.get("finish_reason") == "content_filter":
            set_span_attribute(span, f"{prefix}.role", "assistant")
            set_span_attribute(span, f"{prefix}.content", "FILTERED")

            return

        message = choice.get("message")
        if not message:
            return

        set_span_attribute(span, f"{prefix}.role", message.get("role"))
        set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if function_call:
            set_span_attribute(span, f"{prefix}.tool_calls.0.name", function_call.get("name"))
            set_span_attribute(
                span,
                f"{prefix}.tool_calls.0.arguments",
                function_call.get("arguments"),
            )

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                function = tool_call.get("function")
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.id",
                    tool_call.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.name",
                    function.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{prefix}.tool_calls.{i}.arguments",
                    function.get("arguments"),
                )


@dont_throw  # type:ignore[misc]
def _set_response_attributes(span: Span, response: Any, token_histogram: Optional[Histogram]) -> None:
    response = model_as_dict(response)
    set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cached_tokens: Optional[int] = None
    additional_tokens = usage.get("prompt_tokens_details")
    if additional_tokens:
        cached_tokens = additional_tokens.get("cached_tokens")

    if usage:
        set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        set_span_attribute(span, "gen_ai.usage.cached_tokens", cached_tokens)

    if isinstance(prompt_tokens, int) and prompt_tokens >= 0 and token_histogram is not None:
        token_histogram.record(
            prompt_tokens,
            attributes={
                SpanAttributes.LLM_TOKEN_TYPE: "input",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )

    if isinstance(completion_tokens, int) and completion_tokens >= 0 and token_histogram is not None:
        token_histogram.record(
            completion_tokens,
            attributes={
                SpanAttributes.LLM_TOKEN_TYPE: "output",
                SpanAttributes.LLM_RESPONSE_MODEL: response.get("model"),
            },
        )

    choices = response.get("choices")
    if should_send_prompts() and choices:
        _set_completions(span, choices)


def _with_tracer_wrapper(
    func: Callable[..., Any],
) -> Callable[[Tracer, Dict[str, Any]], Callable[[Callable[..., Any], Any, tuple[Any, ...], Dict[str, Any]], Any]]:
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer: Tracer, to_wrap: Dict[str, Any]) -> Any:
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _with_chat_telemetry_wrapper(func: Callable[..., Any]) -> Callable[
    [Tracer, Optional[Histogram], Optional[Counter], Optional[Histogram], Dict[str, Any]],
    Callable[[Callable[..., Any], Any, tuple[Any, ...], Dict[str, Any]], Any],
]:
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def _with_chat_telemetry(
        tracer: Tracer,
        token_histogram: Optional[Histogram],
        choice_counter: Optional[Counter],
        duration_histogram: Optional[Histogram],
        to_wrap: Dict[str, Any],
    ) -> Callable[[Callable[..., Any], Any, tuple[Any, ...], Dict[str, Any]], Any]:
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            return func(
                tracer,
                token_histogram,
                choice_counter,
                duration_histogram,
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return _with_chat_telemetry


def _create_metrics(meter: Meter) -> Tuple[Histogram, Counter, Histogram]:
    token_histogram = meter.create_histogram(
        name=Meters.LLM_TOKEN_USAGE,
        unit="token",
        description="Measures number of input and output tokens used",
    )

    choice_counter = meter.create_counter(
        name=Meters.LLM_GENERATION_CHOICES,
        unit="choice",
        description="Number of choices returned by chat completions call",
    )

    duration_histogram = meter.create_histogram(
        name=Meters.LLM_OPERATION_DURATION,
        unit="s",
        description="GenAI operation duration",
    )

    return token_histogram, choice_counter, duration_histogram


def _process_streaming_chunk(chunk: Any) -> Tuple[Optional[str], Optional[str], Any]:
    """Extract content, finish_reason and usage from a streaming chunk."""
    if not chunk.choices:
        return None, None, None

    delta = chunk.choices[0].delta
    content = delta.content if hasattr(delta, "content") else None
    finish_reason = chunk.choices[0].finish_reason

    # Extract usage from x_groq if present in the final chunk
    usage = None
    if hasattr(chunk, "x_groq") and chunk.x_groq and chunk.x_groq.usage:
        usage = chunk.x_groq.usage

    return content, finish_reason, usage


def _set_streaming_response_attributes(
    span: Span,
    accumulated_content: str,
    finish_reason: Optional[str] = None,
    usage: Any = None,
) -> None:
    """Set span attributes for accumulated streaming response."""
    if not span.is_recording():
        return

    prefix = f"{SpanAttributes.LLM_COMPLETIONS}.0"
    set_span_attribute(span, f"{prefix}.role", "assistant")
    set_span_attribute(span, f"{prefix}.content", accumulated_content)
    if finish_reason:
        set_span_attribute(span, f"{prefix}.finish_reason", finish_reason)
    usage_as_dict = model_as_dict(usage)
    cached_tokens: Optional[int] = None
    if usage_as_dict.get("prompt_tokens_details"):
        cached_tokens = usage_as_dict.get("prompt_tokens_details").get("cached_tokens")
    if usage:
        set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, usage.completion_tokens)
        set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.prompt_tokens)
        set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens)
        set_span_attribute(span, "gen_ai.usage.cached_tokens", cached_tokens)


def _create_stream_processor(response: Stream, span: Span) -> Iterator[Any]:
    """Create a generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    for chunk in response:
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
        yield chunk

    if span.is_recording():
        _set_streaming_response_attributes(span, accumulated_content, finish_reason, usage)
        span.set_status(Status(StatusCode.OK))
    span.end()


async def _create_async_stream_processor(response: AsyncStream, span: Span) -> AsyncIterator[Any]:
    """Create an async generator that processes a stream while collecting telemetry."""
    accumulated_content = ""
    finish_reason = None
    usage = None

    async for chunk in response:
        content, chunk_finish_reason, chunk_usage = _process_streaming_chunk(chunk)
        if content:
            accumulated_content += content
        if chunk_finish_reason:
            finish_reason = chunk_finish_reason
        if chunk_usage:
            usage = chunk_usage
        yield chunk

    if span.is_recording():
        _set_streaming_response_attributes(span, accumulated_content, finish_reason, usage)
        span.set_status(Status(StatusCode.OK))
    span.end()


@_with_chat_telemetry_wrapper
def _wrap(
    tracer: Tracer,
    token_histogram: Optional[Histogram],
    choice_counter: Optional[Counter],
    duration_histogram: Optional[Histogram],
    to_wrap: Dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    if span.is_recording():
        _set_input_attributes(span, kwargs)

    start_time = time.time()
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        raise e

    end_time = time.time()

    if is_streaming_response(response):
        try:
            return _create_stream_processor(response, span)
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        try:
            metric_attributes = shared_metrics_attributes(response)

            if duration_histogram:
                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes=metric_attributes,
                )

            if span.is_recording():
                _set_response_attributes(span, response, token_histogram)

        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set response attributes for groq span, error: %s",
                str(ex),
            )
        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


@_with_chat_telemetry_wrapper
async def _awrap(
    tracer: Tracer,
    token_histogram: Optional[Histogram],
    choice_counter: Optional[Counter],
    duration_histogram: Optional[Histogram],
    to_wrap: Dict[str, Any],
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    name = to_wrap.get("span_name")
    span = tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "Groq",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )
    try:
        if span.is_recording():
            _set_input_attributes(span, kwargs)

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set input attributes for groq span, error: %s", str(ex))

    start_time = time.time()
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        end_time = time.time()
        attributes = error_metrics_attributes(e)

        if duration_histogram:
            duration = end_time - start_time
            duration_histogram.record(duration, attributes=attributes)

        raise e

    end_time = time.time()

    if is_streaming_response(response):
        try:
            return await _create_async_stream_processor(response, span)  # type:ignore[misc]
        except Exception as ex:
            logger.warning(
                "Failed to process streaming response for groq span, error: %s",
                str(ex),
            )
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            raise
    elif response:
        metric_attributes = shared_metrics_attributes(response)

        if duration_histogram:
            duration = time.time() - start_time
            duration_histogram.record(
                duration,
                attributes=metric_attributes,
            )

        if span.is_recording():
            _set_response_attributes(span, response, token_histogram)

        if span.is_recording():
            span.set_status(Status(StatusCode.OK))
    span.end()
    return response


def is_metrics_enabled() -> bool:
    return (os.getenv("TRACELOOP_METRICS_ENABLED") or "true").lower() == "true"


class GroqInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for Groq's client library."""

    def __init__(
        self,
        enrich_token_usage: bool = False,
        exception_logger: Optional[Callable[[BaseException], None]] = None,
        get_common_metrics_attributes: Callable[[], Dict[str, Any]] = lambda: {},
    ) -> None:
        super().__init__()
        Config.exception_logger = exception_logger
        Config.enrich_token_usage = enrich_token_usage
        Config.get_common_metrics_attributes = get_common_metrics_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # meter and counters are inited here
        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            (
                token_histogram,
                choice_counter,
                duration_histogram,
            ) = _create_metrics(meter)
        else:
            (
                token_histogram,
                choice_counter,
                duration_histogram,
            ) = (None, None, None)

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _wrap(
                        tracer,
                        token_histogram,
                        choice_counter,
                        duration_histogram,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

        for wrapped_method in WRAPPED_AMETHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            try:
                wrap_function_wrapper(
                    wrap_package,
                    f"{wrap_object}.{wrap_method}",
                    _awrap(
                        tracer,
                        token_histogram,
                        choice_counter,
                        duration_histogram,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs: Any) -> None:
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"groq.resources.completions.{wrap_object}",
                wrapped_method.get("method"),
            )
