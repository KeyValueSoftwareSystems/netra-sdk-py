import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

from opentelemetry import context as context_api

from netra.config import Config

logger = logging.getLogger(__name__)

ATTR_PROMPT_ID = f"{Config.LIBRARY_NAME}.prompt.id"
ATTR_PROMPT_NAME = f"{Config.LIBRARY_NAME}.prompt.name"
ATTR_PROMPT_VERSION = f"{Config.LIBRARY_NAME}.prompt.version"
ATTR_PROMPT_LABEL = f"{Config.LIBRARY_NAME}.prompt.label"

# Prompt identity tagged onto every span started inside `use_prompt`. Stored in the
# OTel context (not a bare ContextVar) so the threading instrumentation carries it
# into ThreadPoolExecutor workers.
_ACTIVE_PROMPT_KEY = "netra.active_prompt"


def get_active_prompt_attributes() -> Dict[str, Any]:
    """
    Build span attributes for the prompt set by `use_prompt`.

    Returns:
        Span attributes, or an empty dict when no prompt is active or it carries no identity
    """
    prompt = context_api.get_value(_ACTIVE_PROMPT_KEY)
    if not isinstance(prompt, Mapping):
        return {}

    prompt_id = prompt.get("promptId")
    version = prompt.get("versionNumber")
    if not prompt_id or version is None:
        logger.debug("Active prompt has no promptId/versionNumber; spans will not be linked: %s", list(prompt))
        return {}

    attributes: Dict[str, Any] = {ATTR_PROMPT_ID: prompt_id, ATTR_PROMPT_VERSION: version}
    if prompt.get("promptName"):
        attributes[ATTR_PROMPT_NAME] = prompt["promptName"]
    if prompt.get("label"):
        attributes[ATTR_PROMPT_LABEL] = prompt["label"]
    return attributes


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
    token = context_api.attach(context_api.set_value(_ACTIVE_PROMPT_KEY, prompt))
    try:
        yield
    finally:
        context_api.detach(token)
