from typing import Any, Dict


def build_prompt_version_payload(
    prompt_name: str,
    label: str,
) -> Dict[str, Any]:
    """Build the JSON payload for the prompt-version endpoint.

    Args:
        prompt_name: Name of the prompt.
        label: Label of the prompt version.

    Returns:
        Dictionary payload for the POST request.
    """
    return {"promptName": prompt_name, "label": label}
