import logging
from contextvars import ContextVar
from copy import copy, deepcopy
from typing import Any, Callable, Mapping, Set, Tuple

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span, SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import BoundFunctionWrapper, FunctionWrapper

from netra.instrumentation.libraries.dspy.utils import (
    SPAN_KIND_CHAIN,
    SPAN_KIND_EMBEDDING,
    SPAN_KIND_LLM,
    SPAN_KIND_RETRIEVER,
    bind_arguments,
    convert_to_dict,
    extract_llm_input_messages,
    extract_llm_output_messages,
    extract_usage_info,
    flatten_attributes,
    get_input_value_from_method,
    get_llm_invocation_parameters,
    get_llm_model_name,
    get_predict_span_name,
    prediction_to_output_dict,
    redact_sensitive_arguments,
    safe_json_dumps,
    should_suppress_instrumentation,
)
from netra.instrumentation.span_utils import should_capture_content

logger = logging.getLogger(__name__)


class CopyableBoundFunctionWrapper(BoundFunctionWrapper):  # type: ignore
    """
    A bound function wrapper that can be copied and deep-copied.
    This allows DSPy classes to be copied when they use lm.copy().

    Reference: https://github.com/GrahamDumpleton/wrapt/issues/86#issuecomment-426161271
    """

    def __copy__(self) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(copy(self.__wrapped__), self._self_instance, self._self_wrapper)

    def __deepcopy__(self, memo: dict[int, Any] | None) -> "CopyableBoundFunctionWrapper":
        return CopyableBoundFunctionWrapper(
            deepcopy(self.__wrapped__, memo),
            self._self_instance,
            self._self_wrapper,
        )


class CopyableFunctionWrapper(FunctionWrapper):  # type: ignore
    """
    A function wrapper that can be copied and deep-copied.
    This is essential for DSPy's lm.copy() functionality.

    Reference: https://wrapt.readthedocs.io/en/master/wrappers.html#custom-function-wrappers
    """

    __bound_function_wrapper__ = CopyableBoundFunctionWrapper

    def __copy__(self) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(copy(self.__wrapped__), self._self_wrapper)

    def __deepcopy__(self, memo: dict[int, Any] | None) -> "CopyableFunctionWrapper":
        return CopyableFunctionWrapper(deepcopy(self.__wrapped__, memo), self._self_wrapper)


# Context variable to track which module instances are currently being instrumented
# This prevents duplicate spans from DSPy's internal decorator chains
_instrumenting_modules: ContextVar[Set[int]] = ContextVar("_instrumenting_modules", default=set())


def _record(span: Span, description: str, record: Callable[[], None]) -> None:
    """Run a telemetry step; failures are logged and never reach the user's call."""
    try:
        record()
    except Exception:
        logger.debug("dspy: failed to record %s on span %s", description, getattr(span, "name", ""), exc_info=True)


def _set_input(span: Span, value: Any, mime_type: str = "application/json") -> None:
    """Record ``gen_ai.request.input`` (content) when content capture is enabled."""
    if should_capture_content():
        span.set_attribute("gen_ai.request.input", value)
        span.set_attribute("gen_ai.request.input.mime_type", mime_type)


def _set_output(span: Span, value: Any, mark_ok: bool = False) -> None:
    """Record ``gen_ai.response.output`` (content) when content capture is enabled."""
    if should_capture_content():
        span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(value)))
        span.set_attribute("gen_ai.response.output.mime_type", "application/json")
    if mark_ok:
        span.set_status(Status(StatusCode.OK))


def _llm_span_attributes(instance: Any, operation: str) -> dict[str, Any]:
    span_attributes = dict(
        flatten_attributes(
            {
                SpanAttributes.LLM_SYSTEM: "dspy",
                "gen_ai.operation.name": operation,
                "dspy.span_kind": SPAN_KIND_LLM,
            }
        )
    )
    if model_name := get_llm_model_name(instance):
        span_attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name
    return span_attributes


def _set_lm_request_attributes(span: Span, instance: Any, kwargs: Mapping[str, Any], arguments: Any) -> None:
    span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")

    invocation_params = get_llm_invocation_parameters(instance, kwargs)
    if invocation_params:
        span.set_attribute(SpanAttributes.LLM_HEADERS, safe_json_dumps(invocation_params))
        if (temp := invocation_params.get("temperature")) is not None:
            span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, temp)
        if (max_tokens := invocation_params.get("max_tokens")) is not None:
            span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

    if should_capture_content():
        for key, value in extract_llm_input_messages(arguments):
            span.set_attribute(key, value)
        span.set_attribute(
            "gen_ai.request.input",
            safe_json_dumps(redact_sensitive_arguments(convert_to_dict(arguments))),
        )


def _set_lm_response_attributes(span: Span, response: Any) -> None:
    response_dict = convert_to_dict(response)
    if isinstance(response_dict, dict) and (resp_model := response_dict.get("model")):
        span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, resp_model)

    for key, value in extract_usage_info(response):
        span.set_attribute(key, value)

    if should_capture_content():
        for key, value in extract_llm_output_messages(response):
            span.set_attribute(key, value)
        span.set_attribute("gen_ai.response.output", safe_json_dumps(response_dict))
    span.set_status(Status(StatusCode.OK))


def _module_input_value(instance: Any, arguments: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    forward_method = getattr(instance.__class__, "forward", None)
    if forward_method:
        return get_input_value_from_method(forward_method, *args, **kwargs)
    return safe_json_dumps(arguments)


# Every wrapper below follows the same shape: only exceptions raised by the wrapped DSPy
# call reach the caller (``start_as_current_span`` records them and sets ERROR status);
# telemetry failures are logged at DEBUG and never change the call's result.


class LMCallWrapper:
    """
    Wrapper for LM.__call__ method - the primary sync interface for DSPy LM calls.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        arguments = bind_arguments(wrapped, *args, **kwargs)
        span_name = f"DSPy.{instance.__class__.__name__}.__call__"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes=_llm_span_attributes(instance, "call"),
        ) as span:
            _record(span, "request", lambda: _set_lm_request_attributes(span, instance, kwargs, arguments))
            response = wrapped(*args, **kwargs)
            _record(span, "response", lambda: _set_lm_response_attributes(span, response))
            return response


class LMAsyncCallWrapper:
    """
    Wrapper for LM.acall method - async interface for DSPy LM calls.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        arguments = bind_arguments(wrapped, *args, **kwargs)
        span_name = f"DSPy.{instance.__class__.__name__}.acall"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes=_llm_span_attributes(instance, "acall"),
        ) as span:
            _record(span, "request", lambda: _set_lm_request_attributes(span, instance, kwargs, arguments))
            response = await wrapped(*args, **kwargs)
            _record(span, "response", lambda: _set_lm_response_attributes(span, response))
            return response


class PredictForwardWrapper:
    """
    Wrapper for Predict.forward method - creates chain spans for predictions.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Handle Predict subclasses to avoid duplicate spans
        try:
            from dspy import Predict

            is_instance_of_predict_subclass = (
                isinstance(instance, Predict) and (cls := instance.__class__) is not Predict
            )
            has_overridden_forward_method = getattr(cls, "forward", None) is not getattr(Predict, "forward", None)
            wrapped_method_is_base_class_forward_method = wrapped.__qualname__ == Predict.forward.__qualname__
            if (
                is_instance_of_predict_subclass
                and has_overridden_forward_method
                and wrapped_method_is_base_class_forward_method
            ):
                return wrapped(*args, **kwargs)
        except ImportError:
            pass

        signature = kwargs.get("signature", getattr(instance, "signature", None))
        span_name = get_predict_span_name(instance)
        bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
            attributes={
                "dspy.span_kind": SPAN_KIND_CHAIN,
                "dspy.operation": "predict",
            },
        ) as span:
            _record(span, "input", lambda: _set_input(span, get_input_value_from_method(wrapped, *args, **kwargs)))

            prediction = wrapped(*args, **kwargs)

            def _record_output() -> None:
                if should_capture_content():
                    output_dict = (
                        prediction_to_output_dict(prediction, signature) if signature else convert_to_dict(prediction)
                    )
                    span.set_attribute("gen_ai.response.output", safe_json_dumps(output_dict))
                    span.set_attribute("gen_ai.response.output.mime_type", "application/json")
                span.set_status(Status(StatusCode.OK))

            _record(span, "output", _record_output)
            return prediction


class ModuleForwardWrapper:
    """
    Wrapper for Module.__call__ - instruments user-defined DSPy modules.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Re-entry detection: Skip if we're already instrumenting this instance
        # This prevents duplicate spans from DSPy's internal decorator chains
        instance_id = id(instance)
        instrumenting = _instrumenting_modules.get()

        if instance_id in instrumenting:
            # Already instrumenting this instance, skip to avoid duplicate span
            return wrapped(*args, **kwargs)

        span_name = f"DSPy.{instance.__class__.__name__}.forward"
        arguments = bind_arguments(wrapped, *args, **kwargs)

        # Mark this instance as being instrumented
        new_instrumenting = instrumenting.copy()
        new_instrumenting.add(instance_id)
        token = _instrumenting_modules.set(new_instrumenting)

        try:
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "dspy.span_kind": SPAN_KIND_CHAIN,
                    "dspy.operation": "module",
                },
            ) as span:
                _record(span, "input", lambda: _set_input(span, _module_input_value(instance, arguments, args, kwargs)))
                result = wrapped(*args, **kwargs)
                _record(span, "output", lambda: _set_output(span, result, mark_ok=True))
                return result
        finally:
            # Clean up: remove this instance from the tracking set
            _instrumenting_modules.reset(token)


class ModuleAsyncCallWrapper:
    """
    Wrapper for Module.acall - instruments async calls on user-defined DSPy modules.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        # Re-entry detection: Skip if we're already instrumenting this instance
        # This prevents duplicate spans from DSPy's internal decorator chains
        instance_id = id(instance)
        instrumenting = _instrumenting_modules.get()

        if instance_id in instrumenting:
            # Already instrumenting this instance, skip to avoid duplicate span
            return await wrapped(*args, **kwargs)

        span_name = f"DSPy.{instance.__class__.__name__}.acall"
        arguments = bind_arguments(wrapped, *args, **kwargs)

        # Mark this instance as being instrumented
        new_instrumenting = instrumenting.copy()
        new_instrumenting.add(instance_id)
        token = _instrumenting_modules.set(new_instrumenting)

        try:
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "dspy.span_kind": SPAN_KIND_CHAIN,
                    "dspy.operation": "module_async",
                },
            ) as span:
                _record(span, "input", lambda: _set_input(span, _module_input_value(instance, arguments, args, kwargs)))
                result = await wrapped(*args, **kwargs)
                _record(span, "output", lambda: _set_output(span, result, mark_ok=True))
                return result
        finally:
            # Clean up: remove this instance from the tracking set
            _instrumenting_modules.reset(token)


class RetrieverForwardWrapper:
    """
    Wrapper for Retrieve.forward - instruments retrieval operations.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        span_name = f"DSPy.{instance.__class__.__name__}.forward"
        bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                "dspy.span_kind": SPAN_KIND_RETRIEVER,
                "dspy.operation": "retrieve",
            },
        ) as span:
            _record(span, "input", lambda: _set_input(span, get_input_value_from_method(wrapped, *args, **kwargs)))

            prediction = wrapped(*args, **kwargs)

            def _record_output() -> None:
                if should_capture_content():
                    if isinstance(prediction, dict) and (passages := prediction.get("passages")):
                        for i, passage_text in enumerate(passages):
                            span.set_attribute(f"retrieval.documents.{i}.content", str(passage_text))
                    _set_output(span, prediction)
                span.set_status(Status(StatusCode.OK))

            _record(span, "output", _record_output)
            return prediction


class EmbedderCallWrapper:
    """
    Wrapper for Embedder.__call__ - instruments embedding operations.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        arguments = bind_arguments(wrapped, *args, **kwargs)
        input_texts = arguments.get("texts") or (args[0] if args else [])
        span_name = f"DSPy.{instance.__class__.__name__}.__call__"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                "dspy.span_kind": SPAN_KIND_EMBEDDING,
                "dspy.operation": "embedding",
            },
        ) as span:

            def _record_input() -> None:
                if model_name := getattr(instance, "name", None):
                    span.set_attribute("gen_ai.request.model", str(model_name))
                if should_capture_content():
                    if isinstance(input_texts, list):
                        for i, text in enumerate(input_texts):
                            span.set_attribute(f"gen_ai.embeddings.{i}.text", str(text))
                    _set_input(span, safe_json_dumps(input_texts), mime_type="text/plain")

            _record(span, "input", _record_input)

            response = wrapped(*args, **kwargs)

            def _record_output() -> None:
                # Embedding vectors are derived from (and can leak) the input text.
                if should_capture_content():
                    if isinstance(response, list):
                        for i, embedding in enumerate(response):
                            if hasattr(embedding, "tolist"):
                                span.set_attribute(f"gen_ai.embeddings.{i}.vector", str(embedding.tolist()))
                    _set_output(span, response)
                span.set_status(Status(StatusCode.OK))

            _record(span, "output", _record_output)
            return response


class ModuleForwardSyncWrapper:
    """Wrapper for dspy.Module.forward() to trace direct forward() calls."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self, wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        """Intercepts Module.forward() calls."""
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        # Re-entry detection: Skip if we're already instrumenting this instance
        # This prevents duplicate spans from DSPy's internal decorator chains
        instance_id = id(instance)
        instrumenting = _instrumenting_modules.get()

        if instance_id in instrumenting:
            # Already instrumenting this instance, skip to avoid duplicate span
            return wrapped(*args, **kwargs)

        # Get the module class name (e.g., "UpperModule", "ReAct", "ChainOfThought")
        module_class_name = instance.__class__.__name__
        operation_name = f"DSPy.{module_class_name}.forward"

        # Mark this instance as being instrumented
        new_instrumenting = instrumenting.copy()
        new_instrumenting.add(instance_id)
        token_ctx = _instrumenting_modules.set(new_instrumenting)

        try:
            with self._tracer.start_as_current_span(
                operation_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "dspy.span_kind": SPAN_KIND_CHAIN,
                    "dspy.operation": "forward",
                    "dspy.module_type": module_class_name,
                },
            ) as span:
                # Call the actual forward method with suppression to avoid duplicate spans
                from opentelemetry.context import attach, detach, set_value

                token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
                try:
                    response = wrapped(*args, **kwargs)
                finally:
                    detach(token)

                def _record_io() -> None:
                    if args or kwargs:
                        _set_input(span, safe_json_dumps(convert_to_dict({"args": args, "kwargs": kwargs})))
                    if response is not None:
                        _set_output(span, response)
                    span.set_status(Status(StatusCode.OK))

                _record(span, "input/output", _record_io)
                return response
        finally:
            # Clean up: remove this instance from the tracking set
            _instrumenting_modules.reset(token_ctx)


class ToolCallWrapper:
    """Wrapper for dspy.Tool.__call__() to trace synchronous tool execution."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self, wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        """Intercepts Tool.__call__() calls."""
        if should_suppress_instrumentation():
            return wrapped(*args, **kwargs)

        tool_name = getattr(instance, "name", "unknown_tool")
        operation_name = f"DSPy.Tool.{tool_name}"

        tool_desc = getattr(instance, "desc", None) or ""

        with self._tracer.start_as_current_span(
            operation_name,
            kind=SpanKind.INTERNAL,
            attributes={
                "dspy.span_kind": "tool",
                "dspy.operation": "tool_call",
                "dspy.tool.name": tool_name,
                "dspy.tool.description": tool_desc,
            },
        ) as span:
            if kwargs:
                _record(span, "input", lambda: _set_input(span, safe_json_dumps(convert_to_dict(kwargs))))

            response = wrapped(*args, **kwargs)

            def _record_output() -> None:
                if response is not None:
                    _set_output(span, response)
                span.set_status(Status(StatusCode.OK))

            _record(span, "output", _record_output)
            return response


class ToolAsyncCallWrapper:
    """Wrapper for dspy.Tool.acall() to trace asynchronous tool execution."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self, wrapped: Callable[..., Any], instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        """Intercepts Tool.acall() calls."""
        if should_suppress_instrumentation():
            return await wrapped(*args, **kwargs)

        tool_name = getattr(instance, "name", "unknown_tool")
        operation_name = f"DSPy.Tool.{tool_name}"
        tool_desc = getattr(instance, "desc", None) or ""

        with self._tracer.start_as_current_span(
            operation_name,
            kind=SpanKind.INTERNAL,
            attributes={
                "dspy.span_kind": "tool",
                "dspy.operation": "tool_acall",
                "dspy.tool.name": tool_name,
                "dspy.tool.description": tool_desc,
            },
        ) as span:
            if kwargs:
                _record(span, "input", lambda: _set_input(span, safe_json_dumps(convert_to_dict(kwargs))))

            response = await wrapped(*args, **kwargs)

            def _record_output() -> None:
                if response is not None:
                    _set_output(span, response)
                span.set_status(Status(StatusCode.OK))

            _record(span, "output", _record_output)
            return response
