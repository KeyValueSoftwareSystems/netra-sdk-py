from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Mapping, Optional

from netra.config import Config

ATTR_PROMPT_ID = f"{Config.LIBRARY_NAME}.prompt.id"
ATTR_PROMPT_NAME = f"{Config.LIBRARY_NAME}.prompt.name"
ATTR_PROMPT_VERSION = f"{Config.LIBRARY_NAME}.prompt.version"
ATTR_PROMPT_LABEL = f"{Config.LIBRARY_NAME}.prompt.label"

# Prompt identity tagged onto every span started inside `use_prompt`.
_active_prompt: ContextVar[Optional[Mapping[str, Any]]] = ContextVar("netra_active_prompt", default=None)


def get_active_prompt_attributes() -> Dict[str, Any]:
    """
    Build span attributes for the prompt set by `use_prompt`.

    Returns:
        Span attributes, or an empty dict when no prompt is active or it carries no identity
    """
    prompt = _active_prompt.get()
    if not isinstance(prompt, Mapping):
        return {}

    prompt_id = prompt.get("promptId")
    version = prompt.get("versionNumber")
    if not prompt_id or version is None:
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

    Context variables do not cross into threads started with `threading.Thread`;
    spans created there are not linked.

    Args:
        prompt: The value returned by `Netra.prompts.get_prompt`. A missing or
            empty value is accepted and links nothing.
    """
    token = _active_prompt.set(prompt)
    try:
        yield
    finally:
        _active_prompt.reset(token)
