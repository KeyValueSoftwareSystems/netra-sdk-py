"""OpenTelemetry Mistral AI instrumentation"""

import json
import logging
import os
from importlib import import_module
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable, Collection, Dict, Generator, List, Optional, Tuple

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

from netra.instrumentation.libraries.mistralai.config import Config
from netra.instrumentation.libraries.mistralai.utils import dont_throw
from netra.instrumentation.libraries.mistralai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("mistralai >= 1.0.0",)


def _wrapped_methods(package: str) -> List[Dict[str, Any]]:
    """Return the methods to wrap when the SDK's resources live under *package*.

    Args:
        package: ``"mistralai"`` for SDK 1.x/2.x, ``"mistralai.client"`` for 3.x.

    Returns:
        One entry per wrapped method.
    """
    return [
        {
            "module": f"{package}.chat",
            "object": "Chat",
            "method": "complete",
            "span_name": "mistralai.chat.complete",
            "streaming": False,
            "is_async": False,
        },
        {
            "module": f"{package}.chat",
            "object": "Chat",
            "method": "complete_async",
            "span_name": "mistralai.chat.complete_async",
            "streaming": False,
            "is_async": True,
        },
        {
            "module": f"{package}.chat",
            "object": "Chat",
            "method": "stream",
            "span_name": "mistralai.chat.stream",
            "streaming": True,
            "is_async": False,
        },
        {
            "module": f"{package}.chat",
            "object": "Chat",
            "method": "stream_async",
            "span_name": "mistralai.chat.stream_async",
            "streaming": True,
            "is_async": True,
        },
        {
            "module": f"{package}.embeddings",
            "object": "Embeddings",
            "method": "create",
            "span_name": "mistralai.embeddings",
            "streaming": False,
            "is_async": False,
        },
    ]


# SDK 1.x/2.x.
WRAPPED_METHODS = _wrapped_methods("mistralai")
# SDK 3.x moved every resource under ``mistralai.client`` and made ``mistralai``
# itself a namespace package.
WRAPPED_METHODS_V3 = _wrapped_methods("mistralai.client")


def _select_wrapped_methods() -> List[Dict[str, Any]]:
    """Return the wrap table matching the installed SDK's layout.

    Probes the chat module rather than the version number, so a layout change
    is detected even in a release the table was not written for.

    Returns:
        The matching table, or an empty list when neither layout is present.
    """
    for table in (WRAPPED_METHODS_V3, WRAPPED_METHODS):
        try:
            module = import_module(table[0]["module"])
        except ImportError:
            continue
        if hasattr(module, table[0]["object"]):
            return table
    return []


# The exported OpenTelemetry scope name is a wire contract -- dashboards and
# backend queries key off it -- so it is pinned here rather than derived from
# ``__name__``, which would silently change the moment this file moves.
_TRACER_NAME = "netra.instrumentation.mistralai"


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
    response_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, response_id)

    if llm_request_type == LLMRequestTypeValues.EMBEDDING:
        return

    if should_send_prompts():
        choices = (response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)) or []
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


def _new_accumulated_response() -> SimpleNamespace:
    """Return an empty response to accumulate streamed chunks into.

    Plain objects rather than the SDK's own models, whose import path differs
    between SDK versions.
    """
    return SimpleNamespace(
        id="",
        model="",
        choices=[],
        usage=SimpleNamespace(prompt_tokens=0, total_tokens=0, completion_tokens=0),
    )


@dont_throw
def _accumulate_chunk(accumulated_response: SimpleNamespace, res: Any) -> None:
    """Fold one streamed chunk into *accumulated_response*.

    Runs inside the caller's own iteration, so it must never raise.

    Args:
        accumulated_response: The response being built up.
        res: One event from the SDK's stream.
    """
    data = getattr(res, "data", None)
    if data is None:
        return

    if getattr(data, "model", None):
        accumulated_response.model = data.model
    if getattr(data, "usage", None):
        accumulated_response.usage = data.usage
    # ID is the same for all chunks, so it's safe to overwrite it every time
    if getattr(data, "id", None):
        accumulated_response.id = data.id

    for idx, choice in enumerate(getattr(data, "choices", None) or []):
        while len(accumulated_response.choices) <= idx:
            accumulated_response.choices.append(
                SimpleNamespace(
                    index=len(accumulated_response.choices),
                    message=SimpleNamespace(role="assistant", content=""),
                    finish_reason=None,
                )
            )
        accumulated_choice = accumulated_response.choices[idx]

        if getattr(choice, "finish_reason", None):
            accumulated_choice.finish_reason = choice.finish_reason

        delta = getattr(choice, "delta", None)
        if delta:
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                accumulated_choice.message.content += content
            elif content:
                # Newer SDKs may stream a list of content chunks.
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        accumulated_choice.message.content += text
            if getattr(delta, "role", None):
                accumulated_choice.message.role = delta.role


def _accumulate_streaming_response(
    span: Any, llm_request_type: LLMRequestTypeValues, response: Any, token: Any = None
) -> Generator[Any, None, None]:
    accumulated_response = _new_accumulated_response()

    try:
        for res in response:
            yield res
            _accumulate_chunk(accumulated_response, res)

        _set_response_attributes(span, llm_request_type, accumulated_response)
        span.set_status(Status(StatusCode.OK))
    finally:
        span.end()
        if token is not None:
            context_api.detach(token)


async def _aaccumulate_streaming_response(
    span: Any, llm_request_type: LLMRequestTypeValues, response: Any, token: Any = None
) -> AsyncGenerator[Any, None]:
    accumulated_response = _new_accumulated_response()

    try:
        async for res in response:
            yield res
            _accumulate_chunk(accumulated_response, res)

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


def _installed_version() -> str:
    """Return the installed mistralai version, for the inactive warning."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mistralai")
    except PackageNotFoundError:
        return "(unknown version)"


class MistralAiInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for Mistral AI's client library."""

    def __init__(self, exception_logger: Optional[Callable[[Exception], None]] = None) -> None:
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(_TRACER_NAME, __version__, tracer_provider)
        self._wrapped_methods = []
        for wrapped_method in _select_wrapped_methods():
            module_name = wrapped_method.get("module")
            object_name = wrapped_method.get("object")
            method_name = wrapped_method.get("method")
            is_async = wrapped_method.get("is_async")

            wrapper_func = _awrap if is_async else _wrap

            try:
                wrap_function_wrapper(
                    module_name,
                    f"{object_name}.{method_name}",
                    wrapper_func(tracer, wrapped_method),
                )
            except (AttributeError, ImportError) as exc:
                logger.debug("Failed to wrap %s.%s.%s: %s", module_name, object_name, method_name, exc)
                continue
            self._wrapped_methods.append(wrapped_method)

        if not self._wrapped_methods:
            from netra.instrumentation.wiring.activation import warn_instrumentation_inactive

            warn_instrumentation_inactive("MISTRALAI", f"no known API found in mistralai {_installed_version()}")

    def _uninstrument(self, **kwargs: Any) -> None:
        # BaseInstrumentor is a singleton whose __init__ reruns on every
        # construction, so the table recorded at instrument time is re-derived
        # when it is not there.
        for wrapped_method in getattr(self, "_wrapped_methods", None) or _select_wrapped_methods():
            module_name = wrapped_method.get("module")
            object_name = wrapped_method.get("object")
            method_name = wrapped_method.get("method")

            unwrap(
                f"{module_name}.{object_name}",
                method_name,
            )
        self._wrapped_methods = []
