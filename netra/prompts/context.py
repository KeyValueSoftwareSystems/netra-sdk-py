import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

from opentelemetry import context as context_api
from opentelemetry import trace

from netra.config import Config
from netra.netra_logger import get_logger

logger = get_logger(__name__)

# JSON object: {"id", "version", and optionally "name", "label"}.
ATTR_PROMPT = f"{Config.LIBRARY_NAME}.prompt"

# Prompt identity tagged onto every span started inside `use_prompt`. Stored in the
# OTel context (not a bare ContextVar) so the threading instrumentation carries it
# into ThreadPoolExecutor workers.
_ACTIVE_PROMPT_KEY = "netra.active_prompt"
# Mutable list attached by the test-run runner; prompts fetched/used during the task land here
# so the runner can tag the root span after the fact (span attributes are otherwise start-only).
_COLLECTOR_KEY = "netra.prompt_collector"


def get_active_prompt_attributes() -> Dict[str, Any]:
    """
    Build span attributes for the prompt set by `use_prompt`.

    Returns:
        Span attributes, or an empty dict when no prompt is active or it carries no identity
    """
    return prompt_attributes(context_api.get_value(_ACTIVE_PROMPT_KEY))


def prompt_attributes(prompt: Any) -> Dict[str, Any]:
    """Build span attributes from a prompt mapping; empty dict when it carries no identity."""
    if not isinstance(prompt, Mapping):
        return {}

    prompt_id = prompt.get("promptId")
    version = prompt.get("versionNumber")
    if not prompt_id or version is None:
        logger.debug("Active prompt has no promptId/versionNumber; spans will not be linked: %s", list(prompt))
        return {}

    value: Dict[str, Any] = {"id": prompt_id, "version": version}
    if prompt.get("promptName"):
        value["name"] = prompt["promptName"]
    if prompt.get("label"):
        value["label"] = prompt["label"]
    return {ATTR_PROMPT: json.dumps(value)}


def _collect(attributes: Dict[str, Any]) -> None:
    collector = context_api.get_value(_COLLECTOR_KEY)
    if attributes and isinstance(collector, list):
        collector.append(attributes)


def record_prompt(prompt: Any) -> None:
    """Tag the current span with the prompt and report it to any active collector."""
    attributes = prompt_attributes(prompt)
    if not attributes:
        return
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(attributes)
    _collect(attributes)


@contextmanager
def collect_prompts() -> Iterator[List[Dict[str, Any]]]:
    """Collect attributes of every prompt fetched or used inside the block (last one wins)."""
    collected: List[Dict[str, Any]] = []
    token = context_api.attach(context_api.set_value(_COLLECTOR_KEY, collected))
    try:
        yield collected
    finally:
        context_api.detach(token)


@contextmanager
def use_prompt(prompt: Optional[Mapping[str, Any]]) -> Iterator[None]:
    """
    Link every span started inside the block to a prompt version.

    The prompt lives in the OTel context, so it reaches worker threads only when
    the threading instrumentation propagates that context (e.g. `ThreadPoolExecutor`).

    Args:
        prompt: The value returned by `Netra.prompts.get_prompt`. A missing or
            empty value is accepted and links nothing.
    """
    if not isinstance(prompt, Mapping):
        logger.debug("use_prompt called without a prompt mapping (%s); nothing linked", type(prompt).__name__)
    _collect(prompt_attributes(prompt))
    token = context_api.attach(context_api.set_value(_ACTIVE_PROMPT_KEY, prompt))
    try:
        yield
    finally:
        context_api.detach(token)
