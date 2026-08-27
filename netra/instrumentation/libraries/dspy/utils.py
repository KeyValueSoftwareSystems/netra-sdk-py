import json
from inspect import signature
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.util.types import AttributeValue

# Constants for span kinds
SPAN_KIND_LLM = "llm"
SPAN_KIND_CHAIN = "chain"
SPAN_KIND_RETRIEVER = "retriever"
SPAN_KIND_EMBEDDING = "embedding"


def should_suppress_instrumentation() -> bool:
    """Check if instrumentation should be suppressed"""
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


class DSPyJSONEncoder(json.JSONEncoder):
    """
    Provides support for non-JSON-serializable objects in DSPy.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            try:
                from dspy.primitives.example import Example

                if hasattr(o, "_asdict"):
                    # convert namedtuples to dictionaries
                    return o._asdict()
                if isinstance(o, Example):
                    # handles Prediction objects and other sub-classes of Example
                    return getattr(o, "_store", {})
            except ImportError:
                pass

            # Fallback for other objects
            if hasattr(o, "dict") and callable(o.dict):
                return o.dict()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return repr(o)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    Safely serialize an object to JSON, handling non-serializable objects.
    """
    if "cls" not in kwargs:
        kwargs["cls"] = DSPyJSONEncoder
    try:
        return json.dumps(obj, **kwargs)
    except (TypeError, ValueError):
        return repr(obj)


def flatten_attributes(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Flatten nested dictionaries and lists into dot-notation attribute key-value pairs.

    Example:
        {"a": {"b": 1}, "c": [{"d": 2}]} -> [("a.b", 1), ("c.0.d", 2)]
    """
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in flatten_attributes(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                if isinstance(sub_mapping, Mapping):
                    for sub_key, sub_value in flatten_attributes(sub_mapping):
                        yield f"{key}.{index}.{sub_key}", sub_value
                else:
                    yield f"{key}.{index}", sub_mapping
        else:
            # Convert to string if it's not a primitive type
            if isinstance(value, (str, int, float, bool)):
                yield key, value
            else:
                yield key, str(value)


def bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Bind arguments to method signature, handling both positional and keyword arguments.
    This ensures consistent argument extraction regardless of how they're passed.
    """
    try:
        method_signature = signature(method)
        bound_args = method_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)
    except Exception:
        # Fallback if signature binding fails
        return kwargs


def convert_to_dict(obj: Any) -> Any:
    """
    Recursively converts objects to dicts if they have conversion methods.
    Handles nested lists and dictionaries.
    """
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_dict(item) for item in obj]
    return obj


def get_llm_model_name(lm: Any) -> Optional[str]:
    """
    Extract model name from DSPy LM instance.
    DSPy stores model in various attributes depending on version.
    """
    # Try different attribute names
    for attr in ["model_name", "model", "name"]:
        if (model_name := getattr(lm, attr, None)) is not None:
            return str(model_name)
    return None


def get_llm_invocation_parameters(lm: Any, call_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge LM instance kwargs with call-time kwargs to get full invocation parameters.
    """
    lm_kwargs = getattr(lm, "kwargs", {})
    if not isinstance(lm_kwargs, dict):
        lm_kwargs = {}

    # Filter out internal parameters
    filtered_call_kwargs = {
        k: v for k, v in call_kwargs.items() if k not in ["self", "kwargs", "wrapped", "instance", "args"]
    }

    # Merge with call-level kwargs taking precedence
    return {**lm_kwargs, **filtered_call_kwargs}


def extract_llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extract and format input messages for LLM calls.
    Handles both prompt (string) and messages (list) formats.
    """
    if isinstance(prompt := arguments.get("prompt"), str):
        yield f"{SpanAttributes.LLM_PROMPTS}.0.role", "user"
        yield f"{SpanAttributes.LLM_PROMPTS}.0.content", prompt
    elif isinstance(messages := arguments.get("messages"), list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if (role := message.get("role")) is not None:
                yield f"{SpanAttributes.LLM_PROMPTS}.{i}.role", role
            if (content := message.get("content")) is not None:
                yield f"{SpanAttributes.LLM_PROMPTS}.{i}.content", str(content)


def extract_llm_output_messages(response: Any) -> Iterator[Tuple[str, Any]]:
    """
    Extract and format output messages from LLM response.
    """
    # Handle list of string responses
    if isinstance(response, list):
        for i, message in enumerate(response):
            if isinstance(message, str):
                yield f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", "assistant"
                yield f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", message
    # Handle single string response
    elif isinstance(response, str):
        yield f"{SpanAttributes.LLM_COMPLETIONS}.0.role", "assistant"
        yield f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response
    # Handle structured response objects
    elif hasattr(response, "choices"):
        try:
            response_dict = convert_to_dict(response)
            choices = response_dict.get("choices", [])
            for i, choice in enumerate(choices):
                if isinstance(choice, dict):
                    if message := choice.get("message"):
                        if role := message.get("role"):
                            yield f"{SpanAttributes.LLM_COMPLETIONS}.{i}.role", role
                        if content := message.get("content"):
                            yield f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", str(content)
                    if finish_reason := choice.get("finish_reason"):
                        yield f"{SpanAttributes.LLM_COMPLETIONS}.{i}.finish_reason", finish_reason
        except Exception:
            pass


def extract_usage_info(response: Any) -> Iterator[Tuple[str, Any]]:
    """
    Extract token usage information from LLM response.
    """
    try:
        response_dict = convert_to_dict(response)
        if usage := response_dict.get("usage"):
            if isinstance(usage, dict):
                # Handle various usage field names
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")

                if prompt_tokens is not None:
                    yield f"{SpanAttributes.LLM_USAGE_PROMPT_TOKENS}", prompt_tokens
                if completion_tokens is not None:
                    yield f"{SpanAttributes.LLM_USAGE_COMPLETION_TOKENS}", completion_tokens
                if total_tokens is not None:
                    yield f"{SpanAttributes.LLM_USAGE_TOTAL_TOKENS}", total_tokens
    except Exception:
        pass


def get_predict_span_name(instance: Any) -> str:
    """
    Gets the name for the Predict span, combining class name with signature name.
    Example: "Predict(UserDefinedSignature).forward"
    """
    class_name = instance.__class__.__name__
    if (signature := getattr(instance, "signature", None)) and (signature_name := get_signature_name(signature)):
        return f"DSPy.{class_name}({signature_name}).forward"
    return f"DSPy.{class_name}.forward"


def get_signature_name(signature: Any) -> Optional[str]:
    """
    A best-effort attempt to get the name of a DSPy signature.
    """
    if (qual_name := getattr(signature, "__qualname__", None)) is None:
        return None
    return str(qual_name.split(".")[-1])


def get_input_value_from_method(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string.
    Ensures consistent output regardless of positional vs keyword arguments.
    """
    try:
        method_signature = signature(method)
        first_parameter_name = next(iter(method_signature.parameters), None)
        signature_contains_self = first_parameter_name in ["self"]

        bound_arguments = method_signature.bind(
            *([None] if signature_contains_self else []),
            *args,
            **kwargs,
        )

        return safe_json_dumps(
            {
                **{
                    argument_name: argument_value
                    for argument_name, argument_value in bound_arguments.arguments.items()
                    if argument_name not in ["self", "kwargs"]
                },
                **bound_arguments.arguments.get("kwargs", {}),
            }
        )
    except Exception:
        # Fallback to simple serialization
        return safe_json_dumps({"args": args, "kwargs": kwargs})


def prediction_to_output_dict(prediction: Any, signature: Any) -> Dict[str, Any]:
    """
    Parse the prediction to extract output fields based on signature.
    """
    output = {}
    try:
        if hasattr(signature, "output_fields"):
            for output_field_name in signature.output_fields:
                if hasattr(prediction, "get") and callable(prediction.get):
                    if (prediction_value := prediction.get(output_field_name)) is not None:
                        output[output_field_name] = prediction_value
                elif hasattr(prediction, output_field_name):
                    output[output_field_name] = getattr(prediction, output_field_name)
        else:
            # Fallback: try to convert prediction to dict
            output = convert_to_dict(prediction)
    except Exception:
        output = {"result": str(prediction)}

    return output
