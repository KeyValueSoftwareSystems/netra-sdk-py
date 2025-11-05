import logging
import os
import traceback
from functools import wraps
from importlib.metadata import version
from typing import Any, Callable, Dict, TypeVar, cast

from opentelemetry import context as context_api
from opentelemetry.instrumentation.groq.config import Config
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Span

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_GROQ = "groq"

_PYDANTIC_VERSION = version("pydantic")

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def set_span_attribute(span: Span, name: str, value: Any) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)


def should_send_prompts() -> bool:
    return (os.getenv(TRACELOOP_TRACE_CONTENT) or "true").lower() == "true" or bool(
        context_api.get_value("override_enable_content_tracing")
    )


R = TypeVar("R")


def dont_throw(func: Callable[..., R]) -> Callable[..., R]:
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    """
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:  # type:ignore[unused-ignore]
        try:
            return cast(R, func(*args, **kwargs))  # type:ignore[redundant-cast]
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)
            return cast(R, None)

    return wrapper


@dont_throw
def shared_metrics_attributes(response: Any) -> Dict[str, Any]:
    response_dict = model_as_dict(response)

    common_attributes = Config.get_common_metrics_attributes()

    return {
        **common_attributes,
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_GROQ,
        SpanAttributes.LLM_RESPONSE_MODEL: response_dict.get("model"),
    }


@dont_throw
def error_metrics_attributes(exception: BaseException) -> Dict[str, Any]:
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_GROQ,
        "error.type": exception.__class__.__name__,
    }


def model_as_dict(model: Any) -> Any:
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):
        return model_as_dict(model.parse())
    else:
        return model


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """

    return not Config.use_legacy_attributes
