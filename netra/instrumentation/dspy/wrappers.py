import logging
from contextvars import ContextVar
from copy import copy, deepcopy
from typing import Any, Callable, Mapping, Set, Tuple

from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import BoundFunctionWrapper, FunctionWrapper

from netra.instrumentation.dspy.utils import (
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
    safe_json_dumps,
    should_suppress_instrumentation,
)

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

        # Build span attributes
        span_attributes = dict(
            flatten_attributes(
                {
                    SpanAttributes.LLM_SYSTEM: "dspy",
                    "gen_ai.operation.name": "call",
                    "dspy.span_kind": SPAN_KIND_LLM,
                }
            )
        )

        # Add model name
        if model_name := get_llm_model_name(instance):
            span_attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
        ) as span:
            try:
                # Set request attributes
                span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")

                # Set input messages
                for key, value in extract_llm_input_messages(arguments):
                    span.set_attribute(key, value)

                # Set invocation parameters
                invocation_params = get_llm_invocation_parameters(instance, kwargs)
                if invocation_params:
                    span.set_attribute(SpanAttributes.LLM_HEADERS, safe_json_dumps(invocation_params))

                    # Extract temperature and max_tokens if present
                    if (temp := invocation_params.get("temperature")) is not None:
                        span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, temp)
                    if (max_tokens := invocation_params.get("max_tokens")) is not None:
                        span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

                # Set raw input
                span.set_attribute("gen_ai.request.input", safe_json_dumps(convert_to_dict(arguments)))

                # Call the wrapped method
                response = wrapped(*args, **kwargs)

                # Set response attributes
                response_dict = convert_to_dict(response)

                # Set response model if available
                if isinstance(response_dict, dict) and (resp_model := response_dict.get("model")):
                    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, resp_model)

                # Set output messages
                for key, value in extract_llm_output_messages(response):
                    span.set_attribute(key, value)

                # Set usage information
                for key, value in extract_usage_info(response):
                    span.set_attribute(key, value)

                # Set raw output
                span.set_attribute("gen_ai.response.output", safe_json_dumps(response_dict))

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in LMCallWrapper: {e}", exc_info=True)
                return None


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

        # Build span attributes
        span_attributes = dict(
            flatten_attributes(
                {
                    SpanAttributes.LLM_SYSTEM: "dspy",
                    "gen_ai.operation.name": "acall",
                    "dspy.span_kind": SPAN_KIND_LLM,
                }
            )
        )

        # Add model name
        if model_name := get_llm_model_name(instance):
            span_attributes[SpanAttributes.LLM_REQUEST_MODEL] = model_name

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
        ) as span:
            try:
                # Set request attributes
                span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, "chat")

                # Set input messages
                for key, value in extract_llm_input_messages(arguments):
                    span.set_attribute(key, value)

                # Set invocation parameters
                invocation_params = get_llm_invocation_parameters(instance, kwargs)
                if invocation_params:
                    span.set_attribute(SpanAttributes.LLM_HEADERS, safe_json_dumps(invocation_params))

                    # Extract temperature and max_tokens if present
                    if (temp := invocation_params.get("temperature")) is not None:
                        span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, temp)
                    if (max_tokens := invocation_params.get("max_tokens")) is not None:
                        span.set_attribute(SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)

                # Set raw input
                span.set_attribute("gen_ai.request.input", safe_json_dumps(convert_to_dict(arguments)))

                # Call the wrapped method
                response = await wrapped(*args, **kwargs)

                # Set response attributes
                response_dict = convert_to_dict(response)

                # Set response model if available
                if isinstance(response_dict, dict) and (resp_model := response_dict.get("model")):
                    span.set_attribute(SpanAttributes.LLM_RESPONSE_MODEL, resp_model)

                # Set output messages
                for key, value in extract_llm_output_messages(response):
                    span.set_attribute(key, value)

                # Set usage information
                for key, value in extract_usage_info(response):
                    span.set_attribute(key, value)

                # Set raw output
                span.set_attribute("gen_ai.response.output", safe_json_dumps(response_dict))

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in LMAsyncCallWrapper: {e}", exc_info=True)
                return None


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
            try:
                # Set input
                input_value = get_input_value_from_method(wrapped, *args, **kwargs)
                span.set_attribute("gen_ai.request.input", input_value)
                span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                # Call the wrapped method
                prediction = wrapped(*args, **kwargs)

                # Set output
                output_dict = (
                    prediction_to_output_dict(prediction, signature) if signature else convert_to_dict(prediction)
                )
                span.set_attribute("gen_ai.response.output", safe_json_dumps(output_dict))
                span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                return prediction

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in PredictForwardWrapper: {e}", exc_info=True)
                return None


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
                try:
                    # Set input
                    forward_method = getattr(instance.__class__, "forward", None)
                    if forward_method:
                        input_value = get_input_value_from_method(forward_method, *args, **kwargs)
                    else:
                        input_value = safe_json_dumps(arguments)

                    span.set_attribute("gen_ai.request.input", input_value)
                    span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                    # Call the wrapped method
                    result = wrapped(*args, **kwargs)

                    # Set output
                    span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(result)))
                    span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(f"Exception in ModuleForwardWrapper: {e}", exc_info=True)
                    return None
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
                try:
                    # Set input
                    # Try to get the forward method signature for input extraction
                    forward_method = getattr(instance.__class__, "forward", None)
                    if forward_method:
                        input_value = get_input_value_from_method(forward_method, *args, **kwargs)
                    else:
                        input_value = safe_json_dumps(arguments)

                    span.set_attribute("gen_ai.request.input", input_value)
                    span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                    # Call the wrapped method
                    result = await wrapped(*args, **kwargs)

                    # Set output
                    span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(result)))
                    span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(f"Exception in ModuleAsyncCallWrapper: {e}", exc_info=True)
                    return None
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
            try:
                # Set input
                input_value = get_input_value_from_method(wrapped, *args, **kwargs)
                span.set_attribute("gen_ai.request.input", input_value)
                span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                # Call the wrapped method
                prediction = wrapped(*args, **kwargs)

                # Extract documents from prediction
                if isinstance(prediction, dict) and (passages := prediction.get("passages")):
                    for i, passage_text in enumerate(passages):
                        span.set_attribute(f"retrieval.documents.{i}.content", str(passage_text))

                # Set output
                span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(prediction)))
                span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                return prediction

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in RetrieverForwardWrapper: {e}", exc_info=True)
                return None


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
            try:
                # Set model name
                if model_name := getattr(instance, "name", None):
                    span.set_attribute("gen_ai.request.model", str(model_name))

                # Set input texts
                if isinstance(input_texts, list):
                    for i, text in enumerate(input_texts):
                        span.set_attribute(f"gen_ai.embeddings.{i}.text", str(text))

                span.set_attribute("gen_ai.request.input", safe_json_dumps(input_texts))
                span.set_attribute("gen_ai.request.input.mime_type", "text/plain")

                # Call the wrapped method
                response = wrapped(*args, **kwargs)

                # Set embedding outputs (vectors)
                if isinstance(response, list):
                    for i, embedding in enumerate(response):
                        if hasattr(embedding, "tolist"):
                            embedding_vector = embedding.tolist()
                            span.set_attribute(f"gen_ai.embeddings.{i}.vector", str(embedding_vector))

                span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(response)))
                span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in EmbedderCallWrapper: {e}", exc_info=True)
                return None


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
                try:
                    # Call the actual forward method with suppression to avoid duplicate spans
                    from opentelemetry.context import attach, detach, set_value

                    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
                    try:
                        response = wrapped(*args, **kwargs)
                    finally:
                        detach(token)

                    # Extract input arguments
                    if args or kwargs:
                        span.set_attribute(
                            "gen_ai.request.input", safe_json_dumps(convert_to_dict({"args": args, "kwargs": kwargs}))
                        )
                        span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                    # Extract output
                    if response is not None:
                        span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(response)))
                        span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                    span.set_status(Status(StatusCode.OK))
                    return response

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(f"Exception in ModuleForwardSyncWrapper: {e}", exc_info=True)
                    return None
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
        print(f"🔍 DEBUG: Tool.{tool_name} called with kwargs={kwargs}")
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
            try:
                # Extract input arguments
                if kwargs:
                    span.set_attribute("gen_ai.request.input", safe_json_dumps(convert_to_dict(kwargs)))
                    span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                # Call the actual tool function
                response = wrapped(*args, **kwargs)

                # Extract output
                if response is not None:
                    span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(response)))
                    span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in ToolCallWrapper: {e}", exc_info=True)
                return None


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
            try:
                # Extract input arguments
                if kwargs:
                    span.set_attribute("gen_ai.request.input", safe_json_dumps(convert_to_dict(kwargs)))
                    span.set_attribute("gen_ai.request.input.mime_type", "application/json")

                # Call the actual async tool function
                response = await wrapped(*args, **kwargs)

                # Extract output
                if response is not None:
                    span.set_attribute("gen_ai.response.output", safe_json_dumps(convert_to_dict(response)))
                    span.set_attribute("gen_ai.response.output.mime_type", "application/json")

                span.set_status(Status(StatusCode.OK))
                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Exception in ToolAsyncCallWrapper: {e}", exc_info=True)
                return None
