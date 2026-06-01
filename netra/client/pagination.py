import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _empty_page() -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Return a fresh empty-page tuple to avoid shared mutable state.

    Returns:
        A tuple of ([], False, None).
    """
    return ([], False, None)


def parse_paginated_response(
    result: Dict[str, Any],
    items_key: str = "data",
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Parse a standard paginated API response into items, has_next_page, and next_cursor.

    Args:
        result: The raw JSON response dict from the API.
        items_key: Key within "data" that holds the list of items.

    Returns:
        A tuple of (items, has_next_page, next_cursor).
    """
    data_block = result.get("data")
    if data_block is None:
        return _empty_page()

    if not isinstance(data_block, dict):
        logger.error("netra: Unexpected paginated response shape; 'data' is not a dict")
        return _empty_page()

    items = data_block.get(items_key, [])
    if items is None:
        items = []

    if not isinstance(items, list):
        logger.error("netra: Unexpected paginated response shape; '%s' is not a list", items_key)
        return _empty_page()

    page_info = data_block.get("pageInfo", {}) or {}
    has_next_page = bool(page_info.get("hasNextPage", False)) if isinstance(page_info, dict) else False

    next_cursor: Optional[str] = None
    if items:
        last_item = items[-1]
        if isinstance(last_item, dict):
            next_cursor = last_item.get("cursor")

    return items, has_next_page, next_cursor
