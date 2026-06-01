from typing import Dict, Optional


def build_model_pricing_params(
    name: Optional[str] = None,
) -> Dict[str, str]:
    """Build query parameters for the model-pricing endpoint.

    Args:
        name: Optional model name to filter results.

    Returns:
        Dictionary of query parameters.
    """
    params: Dict[str, str] = {}
    if name:
        params["name"] = name
    return params
