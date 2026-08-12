import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span

from netra.config import get_attribute_max_len
from netra.instrumentation.honcho import constants as attrs
from netra.utils import truncate_string

logger = logging.getLogger(__name__)

# Type aliases for attribute-setter callbacks used by wrappers and PatchSpec.
RequestAttrFn = Callable[[Span, Any, Tuple[Any, ...], Dict[str, Any]], None]
ResponseAttrFn = Callable[[Span, Any], None]

_SKIP_PRIVATE = frozenset({"honcho"})


class _SafeEncoder(json.JSONEncoder):
    """Encodes non-serializable objects via ``str()`` instead of raising."""

    def default(self, o: Any) -> Any:
        try:
            return str(o)
        except Exception:
            return repr(o)


def should_suppress_instrumentation() -> bool:
    return context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True


def _safe_set(span: Span, key: str, value: Any) -> None:
    if value is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        logger.debug("Failed to set span attribute '%s'", key, exc_info=True)


def _extract_id(obj: Any) -> Optional[str]:
    """Extract an id string from a peer/session object or passthrough strings."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return getattr(obj, "id", str(obj))


def _jsonify_value(v: Any, _depth: int = 0) -> Any:
    """Convert a single value to a JSON-safe type, recursing into objects."""
    if _depth >= attrs.MAX_SERIALIZE_DEPTH:
        return str(v)
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _jsonify_value(val, _depth + 1) for k, val in v.items() if val is not None}
    if isinstance(v, list):
        return [_jsonify_value(item, _depth + 1) for item in v]
    try:
        nested = _serialize_obj(v, _depth + 1)
        return nested if nested is not None else str(v)
    except Exception:
        return str(v)


def _serialize_obj(obj: Any, _depth: int = 0) -> Optional[Dict[str, Any]]:
    """Serialize any Honcho response object to a JSON-safe dict.

    Tries ``model_dump(mode='json')`` for Pydantic types, then falls
    back to ``vars()`` for plain classes (Message, Conclusion).  For
    Pydantic types that store extra data in private attrs (Peer,
    Session) the private attrs are included automatically.

    A *_depth* counter prevents unbounded recursion on circular or
    deeply nested object graphs (capped at ``MAX_SERIALIZE_DEPTH``).
    """
    if _depth >= attrs.MAX_SERIALIZE_DEPTH:
        return None
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return None
    if isinstance(obj, dict):
        return {k: _jsonify_value(v, _depth + 1) for k, v in obj.items() if v is not None} or None

    data: Dict[str, Any] = {}

    if hasattr(obj, "model_dump"):
        try:
            data = obj.model_dump(mode="json")
        except Exception:
            try:
                data = obj.model_dump()
            except Exception:
                logger.debug("model_dump failed for %s", type(obj).__name__, exc_info=True)
        for k, v in (getattr(obj, "__pydantic_private__", None) or {}).items():
            clean = k.lstrip("_")
            if v is not None and clean not in data and clean not in _SKIP_PRIVATE:
                data[clean] = v

    if not data and hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("_") and v is not None:
                data[k] = v

    if not data:
        return None

    try:
        return {k: _jsonify_value(v, _depth + 1) for k, v in data.items() if v is not None}
    except Exception:
        logger.debug("Failed to serialize object %s", type(obj).__name__, exc_info=True)
        return None


def _safe_json(data: Any) -> str:
    return json.dumps(data, cls=_SafeEncoder)


def _set_input(span: Span, data: Dict[str, Any]) -> None:
    """Set the ``input`` span attribute as a JSON string, omitting None values."""
    filtered = {k: v for k, v in data.items() if v is not None}
    if filtered:
        serialized = _safe_json(filtered)
        _safe_set(span, attrs.INPUT, truncate_string(serialized, get_attribute_max_len()))


def _set_output(span: Span, data: Any) -> None:
    """Set the ``output`` span attribute as a JSON string."""
    if data is None:
        return
    try:
        max_len = get_attribute_max_len()
        if isinstance(data, str):
            _safe_set(span, attrs.OUTPUT, truncate_string(data, max_len))
        elif isinstance(data, (dict, list)):
            _safe_set(span, attrs.OUTPUT, truncate_string(_safe_json(data), max_len))
        else:
            _safe_set(span, attrs.OUTPUT, truncate_string(str(data), max_len))
    except Exception:
        logger.debug("Failed to serialize output for span", exc_info=True)


def _is_page(obj: Any) -> bool:
    """Check if an object is a Honcho paginated response (SyncPage / AsyncPage).

    Prefers ``isinstance`` against the real Honcho page classes when
    importable.  Falls back to structural duck-typing (``items`` +
    ``page`` attributes) so the check survives SDK class renames,
    subclassing, or test mocks.
    """
    try:
        from honcho.pagination import AsyncPage, SyncPage

        if isinstance(obj, (SyncPage, AsyncPage)):
            return True
    except Exception:
        pass
    return hasattr(obj, "items") and hasattr(obj, "page")


def _coerce_to_items(response: Any) -> Optional[List[Any]]:
    """Extract a plain list of items from a response.

    Handles ``list``, ``SyncPage``, and ``AsyncPage``.  For paginated
    types we use the ``.items`` property (current page only) so we never
    trigger additional API calls.  Returns ``None`` when the response
    cannot be converted.
    """
    if isinstance(response, list):
        return response
    if _is_page(response):
        return getattr(response, "items", None)
    return None


def _get_item_count(response: Any) -> Optional[int]:
    """Return the total item count, preferring ``.total`` for paginated types."""
    if isinstance(response, list):
        return len(response)
    if _is_page(response):
        total = getattr(response, "total", None)
        if total is not None:
            return int(total)
        items = getattr(response, "items", None)
        return len(items) if items is not None else None
    return None


def _serialize_items(items: Any) -> Optional[List[Dict[str, Any]]]:
    """Serialize a list of Honcho objects into a list of JSON-safe dicts."""
    if not isinstance(items, list):
        return None
    result = []
    for item in items:
        data = _serialize_obj(item)
        if data:
            result.append(data)
    return result or None


def _add_page_info(output: Dict[str, Any], response: Any) -> None:
    """Append pagination metadata to *output* when *response* is a paginated type."""
    if not _is_page(response):
        return
    for attr in ("page", "size", "pages"):
        val = getattr(response, attr, None)
        if val is not None:
            output[attr] = val


def set_common_attributes(span: Span, instance: Any, operation: str) -> None:
    _safe_set(span, attrs.PROVIDER_NAME, attrs.PROVIDER_VALUE)
    _safe_set(span, attrs.OPERATION_NAME, operation)
    _safe_set(span, attrs.MEMORY_STORE_ID, getattr(instance, "workspace_id", None))


def set_peer_attributes(span: Span, instance: Any) -> None:
    _safe_set(span, attrs.AGENT_ID, getattr(instance, "id", None))


def set_session_attributes(span: Span, instance: Any) -> None:
    _safe_set(span, attrs.CONVERSATION_ID, getattr(instance, "id", None))


def set_conclusion_scope_attributes(span: Span, instance: Any) -> None:
    _safe_set(span, attrs.CONCLUSION_OBSERVER, getattr(instance, "observer", None))
    _safe_set(span, attrs.CONCLUSION_OBSERVED, getattr(instance, "observed", None))


def set_chat_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_CHAT)
    set_peer_attributes(span, instance)
    query = args[0] if args else kwargs.get("query")
    target = _extract_id(kwargs.get("target"))
    session = _extract_id(kwargs.get("session"))
    _safe_set(span, attrs.MEMORY_QUERY_TEXT, query)
    _safe_set(span, attrs.PEER_TARGET, target)
    _safe_set(span, attrs.CONVERSATION_ID, session)
    _safe_set(span, attrs.REQUEST_REASONING_LEVEL, kwargs.get("reasoning_level"))
    _set_input(
        span,
        {
            "query": query,
            "peer_id": getattr(instance, "id", None),
            "target": target,
            "session": session,
        },
    )


def set_chat_response_attrs(span: Span, response: Any) -> None:
    if isinstance(response, str):
        _safe_set(span, attrs.RESPONSE_LENGTH, len(response))
        _set_output(span, response)


def set_add_messages_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_CREATE_MEMORY)
    set_session_attributes(span, instance)
    messages = args[0] if args else kwargs.get("messages", [])
    if isinstance(messages, list):
        _safe_set(span, attrs.MESSAGE_COUNT, len(messages))
    else:
        _safe_set(span, attrs.MESSAGE_COUNT, 1)
    serialized = _serialize_items(messages) if isinstance(messages, list) else None
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "message_count": len(messages) if isinstance(messages, list) else 1,
            "messages": serialized,
        },
    )


def set_add_messages_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is not None:
        _safe_set(span, attrs.RESPONSE_MESSAGE_COUNT, len(items))
        _set_output(span, {"message_count": len(items), "messages": _serialize_items(items)})


def set_upload_file_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_UPLOAD_FILE)
    set_session_attributes(span, instance)
    peer = args[1] if len(args) > 1 else kwargs.get("peer")
    _safe_set(span, attrs.AGENT_ID, _extract_id(peer))
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "peer_id": _extract_id(peer),
        },
    )


def set_upload_file_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    count = len(items)
    _safe_set(span, attrs.RESPONSE_MESSAGE_COUNT, count)
    _set_output(span, {"message_count": count, "messages": _serialize_items(items)})


def set_session_context_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_CONTEXT)
    set_session_attributes(span, instance)
    _safe_set(span, attrs.REQUEST_TOP_K, kwargs.get("tokens"))
    _safe_set(span, attrs.PEER_TARGET, kwargs.get("peer_target"))
    _safe_set(span, attrs.PEER_PERSPECTIVE, kwargs.get("peer_perspective"))
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "tokens": kwargs.get("tokens"),
            "summary": kwargs.get("summary"),
        },
    )


def set_session_context_response_attrs(span: Span, response: Any) -> None:
    data = _serialize_obj(response)
    if not data:
        return
    messages = data.get("messages")
    if isinstance(messages, list):
        _safe_set(span, attrs.RESPONSE_MESSAGE_COUNT, len(messages))
    _safe_set(span, attrs.RESPONSE_HAS_SUMMARY, data.get("summary") is not None)
    _safe_set(span, attrs.RESPONSE_HAS_REPRESENTATION, data.get("peer_representation") is not None)
    _set_output(span, data)


def set_peer_context_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_CONTEXT)
    set_peer_attributes(span, instance)
    target = _extract_id(kwargs.get("target"))
    _safe_set(span, attrs.PEER_TARGET, target)
    _safe_set(span, attrs.RETRIEVAL_QUERY_TEXT, kwargs.get("search_query"))
    _safe_set(span, attrs.RETRIEVAL_TOP_K, kwargs.get("search_top_k"))
    _set_input(
        span,
        {
            "peer_id": getattr(instance, "id", None),
            "target": target,
            "search_query": kwargs.get("search_query"),
        },
    )


def set_peer_context_response_attrs(span: Span, response: Any) -> None:
    data = _serialize_obj(response)
    if not data:
        return
    _safe_set(span, attrs.RESPONSE_HAS_REPRESENTATION, data.get("representation") is not None)
    peer_card = data.get("peer_card")
    if isinstance(peer_card, list):
        _safe_set(span, attrs.RESPONSE_PEER_CARD_COUNT, len(peer_card))
    _set_output(span, data)


def set_representation_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_REPRESENTATION)
    input_data: Dict[str, Any] = {}
    if hasattr(instance, "model_fields") and "id" in instance.model_fields:
        obj_type = type(instance).__name__
        if "Peer" in obj_type:
            set_peer_attributes(span, instance)
            input_data["peer_id"] = getattr(instance, "id", None)
        elif "Session" in obj_type:
            set_session_attributes(span, instance)
            peer = args[0] if args else kwargs.get("peer")
            _safe_set(span, attrs.AGENT_ID, _extract_id(peer))
            input_data["session_id"] = getattr(instance, "id", None)
            input_data["peer_id"] = _extract_id(peer)
    target = _extract_id(kwargs.get("target"))
    _safe_set(span, attrs.PEER_TARGET, target)
    _safe_set(span, attrs.RETRIEVAL_QUERY_TEXT, kwargs.get("search_query"))
    _safe_set(span, attrs.RETRIEVAL_TOP_K, kwargs.get("search_top_k"))
    input_data["target"] = target
    _set_input(span, input_data)


def set_search_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_SEARCH_MEMORY)
    obj_type = type(instance).__name__
    input_data: Dict[str, Any] = {}
    if "Peer" in obj_type:
        set_peer_attributes(span, instance)
        input_data["peer_id"] = getattr(instance, "id", None)
    elif "Session" in obj_type:
        set_session_attributes(span, instance)
        input_data["session_id"] = getattr(instance, "id", None)
    query = args[0] if args else kwargs.get("query")
    _safe_set(span, attrs.MEMORY_QUERY_TEXT, query)
    _safe_set(span, attrs.RETRIEVAL_TOP_K, kwargs.get("limit"))
    input_data["query"] = query
    input_data["limit"] = kwargs.get("limit")
    _set_input(span, input_data)


def set_search_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    count = _get_item_count(response)
    if count is not None:
        _safe_set(span, attrs.RESPONSE_RESULT_COUNT, count)
    output: Dict[str, Any] = {"result_count": count, "results": _serialize_items(items)}
    _add_page_info(output, response)
    _set_output(span, output)


def set_get_card_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_GET_CARD)
    set_peer_attributes(span, instance)
    target = _extract_id(kwargs.get("target") or (args[0] if args else None))
    _safe_set(span, attrs.PEER_TARGET, target)
    _set_input(span, {"peer_id": getattr(instance, "id", None), "target": target})


def set_card_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is not None:
        _safe_set(span, attrs.RESPONSE_CARD_ITEM_COUNT, len(items))
        _set_output(span, {"card_item_count": len(items), "items": _serialize_items(items)})


def set_set_card_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_SET_CARD)
    set_peer_attributes(span, instance)
    items = args[0] if args else kwargs.get("items", [])
    if isinstance(items, list):
        _safe_set(span, attrs.CARD_ITEM_COUNT, len(items))
    target = _extract_id(kwargs.get("target"))
    _safe_set(span, attrs.PEER_TARGET, target)
    _set_input(
        span,
        {
            "peer_id": getattr(instance, "id", None),
            "card_item_count": len(items) if isinstance(items, list) else None,
            "target": target,
        },
    )


def set_conclusions_create_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_CREATE_MEMORY)
    set_conclusion_scope_attributes(span, instance)
    conclusions = args[0] if args else kwargs.get("conclusions", [])
    count = len(conclusions) if isinstance(conclusions, list) else None
    if count is not None:
        _safe_set(span, attrs.CONCLUSION_COUNT, count)
    _set_input(
        span,
        {
            "observer": getattr(instance, "observer", None),
            "observed": getattr(instance, "observed", None),
            "conclusion_count": count,
        },
    )


def set_conclusions_create_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    count = len(items)
    _safe_set(span, attrs.RESPONSE_CONCLUSION_COUNT, count)
    _set_output(span, {"conclusion_count": count, "conclusions": _serialize_items(items)})


def set_conclusions_response_attrs(span: Span, response: Any) -> None:
    """Response setter for ConclusionScope.list / ConclusionScope.query."""
    items = _coerce_to_items(response)
    if items is None:
        return
    count = _get_item_count(response)
    if count is not None:
        _safe_set(span, attrs.RESPONSE_CONCLUSION_COUNT, count)
    output: Dict[str, Any] = {"conclusion_count": count, "conclusions": _serialize_items(items)}
    _add_page_info(output, response)
    _set_output(span, output)


def set_conclusions_list_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_SEARCH_MEMORY)
    set_conclusion_scope_attributes(span, instance)
    _set_input(
        span,
        {
            "observer": getattr(instance, "observer", None),
            "observed": getattr(instance, "observed", None),
        },
    )


def set_conclusions_query_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_SEARCH_MEMORY)
    set_conclusion_scope_attributes(span, instance)
    query = args[0] if args else kwargs.get("query")
    _safe_set(span, attrs.MEMORY_QUERY_TEXT, query)
    _safe_set(span, attrs.REQUEST_TOP_K, kwargs.get("top_k"))
    _set_input(
        span,
        {
            "observer": getattr(instance, "observer", None),
            "observed": getattr(instance, "observed", None),
            "query": query,
            "top_k": kwargs.get("top_k"),
        },
    )


def set_conclusions_delete_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_DELETE_MEMORY)
    set_conclusion_scope_attributes(span, instance)
    conclusion_id = args[0] if args else kwargs.get("conclusion_id")
    _safe_set(span, attrs.MEMORY_RECORD_ID, _extract_id(conclusion_id))
    _set_input(span, {"conclusion_id": _extract_id(conclusion_id)})


def set_get_or_create_peer_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_CREATE_PEER)
    peer_id = args[0] if args else kwargs.get("id")
    _safe_set(span, attrs.AGENT_ID, peer_id)
    _set_input(span, {"peer_id": peer_id})


def set_get_or_create_session_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_CREATE_SESSION)
    session_id = args[0] if args else kwargs.get("id")
    _safe_set(span, attrs.CONVERSATION_ID, session_id)
    _set_input(span, {"session_id": session_id})


def set_add_peers_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_ADD_PEERS)
    set_session_attributes(span, instance)
    peers = args[0] if args else kwargs.get("peers", [])
    peer_ids = [_extract_id(p) for p in peers] if isinstance(peers, list) else []
    if isinstance(peers, list):
        _safe_set(span, attrs.PEER_COUNT, len(peers))
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "peer_ids": peer_ids or None,
        },
    )


def set_messages_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_LIST_MESSAGES)
    set_session_attributes(span, instance)
    _safe_set(span, attrs.PAGE, kwargs.get("page"))
    _safe_set(span, attrs.PAGE_SIZE, kwargs.get("size"))
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "page": kwargs.get("page"),
            "size": kwargs.get("size"),
        },
    )


def set_queue_status_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_QUEUE_STATUS)
    obj_type = type(instance).__name__
    input_data: Dict[str, Any] = {}
    if "Session" in obj_type:
        set_session_attributes(span, instance)
        input_data["session_id"] = getattr(instance, "id", None)
    _set_input(span, input_data)


def set_list_peers_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_LIST_PEERS)
    _set_input(span, {"workspace_id": getattr(instance, "workspace_id", None)})


def set_list_peers_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    total = _get_item_count(response)
    _safe_set(span, attrs.RESPONSE_PEER_COUNT, total)
    output: Dict[str, Any] = {"peer_count": total, "peers": _serialize_items(items)}
    _add_page_info(output, response)
    _set_output(span, output)


def set_session_peers_request_attrs(span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    set_common_attributes(span, instance, attrs.OP_SESSION_PEERS)
    set_session_attributes(span, instance)
    _set_input(span, {"session_id": getattr(instance, "id", None)})


def set_session_peers_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    total = _get_item_count(response)
    _safe_set(span, attrs.RESPONSE_PEER_COUNT, total)
    _set_output(span, {"peer_count": total, "peers": _serialize_items(items)})


def set_session_set_metadata_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_SET_METADATA)
    set_session_attributes(span, instance)
    metadata = args[0] if args else kwargs.get("metadata", {})
    _set_input(
        span,
        {
            "session_id": getattr(instance, "id", None),
            "metadata": metadata if isinstance(metadata, dict) else str(metadata),
        },
    )


def set_peer_set_metadata_request_attrs(
    span: Span, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> None:
    set_common_attributes(span, instance, attrs.OP_SET_METADATA)
    set_peer_attributes(span, instance)
    metadata = args[0] if args else kwargs.get("metadata", {})
    _set_input(
        span,
        {
            "peer_id": getattr(instance, "id", None),
            "metadata": metadata if isinstance(metadata, dict) else str(metadata),
        },
    )


def set_representation_response_attrs(span: Span, response: Any) -> None:
    if isinstance(response, str):
        _set_output(span, response)


def set_get_or_create_peer_response_attrs(span: Span, response: Any) -> None:
    data = _serialize_obj(response)
    if not data:
        return
    _safe_set(span, attrs.AGENT_ID, data.get("id"))
    _set_output(span, data)


def set_get_or_create_session_response_attrs(span: Span, response: Any) -> None:
    data = _serialize_obj(response)
    if not data:
        return
    _safe_set(span, attrs.CONVERSATION_ID, data.get("id"))
    if "is_active" in data:
        _safe_set(span, attrs.SESSION_IS_ACTIVE, data["is_active"])
    _set_output(span, data)


def set_messages_response_attrs(span: Span, response: Any) -> None:
    items = _coerce_to_items(response)
    if items is None:
        return
    total = _get_item_count(response)
    _safe_set(span, attrs.RESPONSE_MESSAGE_COUNT, total)
    output: Dict[str, Any] = {"message_count": total, "messages": _serialize_items(items)}
    _add_page_info(output, response)
    _set_output(span, output)


def set_queue_status_response_attrs(span: Span, response: Any) -> None:
    data = _serialize_obj(response)
    if data:
        _set_output(span, data)


# No-op response attributes for wrappers
def _noop_response_attrs(span: Span, response: Any) -> None:
    pass
