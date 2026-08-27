"""Centralized constants for Honcho instrumentation.

All instrumentation-specific constants live here so that other modules
import from a single source of truth.
"""

LOG_PREFIX = "netra.instrumentation.libraries.honcho"

MAX_SERIALIZE_DEPTH = 5

# SDK module / class mapping (used by PatchSpec for monkey-patching)
ASYNC_MODULE = "honcho.aio"

ASYNC_CLASS_MAP: dict[str, str] = {
    "Session": "SessionAio",
    "Peer": "PeerAio",
    "Honcho": "HonchoAio",
    "ConclusionScope": "ConclusionScopeAio",
}

SYNC_MODULE_MAP: dict[str, str] = {
    "Session": "honcho.session",
    "Peer": "honcho.peer",
    "Honcho": "honcho.client",
    "ConclusionScope": "honcho.conclusions",
}

# OTel GenAI standard span attribute keys
PROVIDER_NAME = "gen_ai.provider.name"
OPERATION_NAME = "gen_ai.operation.name"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_STREAM = "gen_ai.request.stream"
RESPONSE_ID = "gen_ai.response.id"
CONVERSATION_ID = "gen_ai.conversation.id"
REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"
REQUEST_TOP_K = "gen_ai.request.top_k"

# Memory-specific (OTel GenAI Memory semconv)
MEMORY_STORE_ID = "gen_ai.memory.store.id"
MEMORY_RECORD_ID = "gen_ai.memory.record.id"
MEMORY_RECORD_COUNT = "gen_ai.memory.record.count"
MEMORY_QUERY_TEXT = "gen_ai.memory.query.text"

# Retrieval-specific (OTel GenAI Retrieval semconv)
RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"

# Agent identity (OTel GenAI Agent semconv)
AGENT_ID = "gen_ai.agent.id"

# Standard I/O (used across Netra instrumentations)
INPUT = "input"
OUTPUT = "output"

# Honcho-specific span attribute keys (gen_ai.honcho.*)
PEER_TARGET = "gen_ai.honcho.peer.target"
PEER_PERSPECTIVE = "gen_ai.honcho.peer.perspective"
PEER_COUNT = "gen_ai.honcho.peer.count"
CONCLUSION_OBSERVER = "gen_ai.honcho.conclusion.observer"
CONCLUSION_OBSERVED = "gen_ai.honcho.conclusion.observed"
SESSION_IS_ACTIVE = "gen_ai.honcho.session.is_active"
RESPONSE_LENGTH = "gen_ai.honcho.response.length"
RESPONSE_MESSAGE_COUNT = "gen_ai.honcho.response.message_count"
RESPONSE_RESULT_COUNT = "gen_ai.honcho.response.result_count"
RESPONSE_PEER_COUNT = "gen_ai.honcho.response.peer_count"
RESPONSE_CARD_ITEM_COUNT = "gen_ai.honcho.response.card_item_count"
RESPONSE_CONCLUSION_COUNT = "gen_ai.honcho.response.conclusion_count"
RESPONSE_PEER_CARD_COUNT = "gen_ai.honcho.response.peer_card_count"
RESPONSE_HAS_SUMMARY = "gen_ai.honcho.response.has_summary"
RESPONSE_HAS_REPRESENTATION = "gen_ai.honcho.response.has_representation"
MESSAGE_COUNT = "gen_ai.honcho.message_count"
CARD_ITEM_COUNT = "gen_ai.honcho.card_item_count"
CONCLUSION_COUNT = "gen_ai.honcho.conclusion_count"
PAGE = "gen_ai.honcho.page"
PAGE_SIZE = "gen_ai.honcho.page_size"

# Provider value
PROVIDER_VALUE = "honcho"

# Operation name values (gen_ai.operation.name values)
OP_CHAT = "honcho.chat"
OP_CREATE_MEMORY = "create_memory"
OP_SEARCH_MEMORY = "search_memory"
OP_DELETE_MEMORY = "delete_memory"
OP_RETRIEVAL = "retrieval"
OP_UPLOAD_FILE = "honcho.upload_file"
OP_CONTEXT = "honcho.context"
OP_REPRESENTATION = "honcho.representation"
OP_GET_CARD = "honcho.get_card"
OP_SET_CARD = "honcho.set_card"
OP_CREATE_PEER = "honcho.peer.get_or_create"
OP_CREATE_SESSION = "honcho.session.get_or_create"
OP_ADD_PEERS = "honcho.add_peers"
OP_LIST_MESSAGES = "honcho.messages"
OP_QUEUE_STATUS = "honcho.queue_status"
OP_LIST_PEERS = "honcho.peers.list"
OP_SESSION_PEERS = "honcho.session.peers"
OP_SET_METADATA = "honcho.set_metadata"

# Span names
SPAN_ADD_MESSAGES = "honcho.create_memory"
SPAN_UPLOAD_FILE = "honcho.create_memory"
SPAN_SET_CARD = "honcho.create_memory"
SPAN_CONCLUSIONS_CREATE = "honcho.create_memory"
SPAN_CHAT = "honcho.chat"
SPAN_CHAT_STREAM = "honcho.chat.stream"
SPAN_SESSION_CONTEXT = "honcho.retrieval"
SPAN_PEER_CONTEXT = "honcho.retrieval"
SPAN_PEER_REPRESENTATION = "honcho.retrieval"
SPAN_SESSION_REPRESENTATION = "honcho.retrieval"
SPAN_PEER_SEARCH = "honcho.search_memory"
SPAN_SESSION_SEARCH = "honcho.search_memory"
SPAN_WORKSPACE_SEARCH = "honcho.search_memory"
SPAN_GET_CARD = "honcho.retrieval"
SPAN_CONCLUSIONS_LIST = "honcho.search_memory"
SPAN_CONCLUSIONS_QUERY = "honcho.search_memory"
SPAN_CONCLUSIONS_DELETE = "honcho.delete_memory"
SPAN_GET_OR_CREATE_PEER = "honcho.peer.get_or_create"
SPAN_GET_OR_CREATE_SESSION = "honcho.session.get_or_create"
SPAN_ADD_PEERS = "honcho.session.add_peers"
SPAN_LIST_MESSAGES = "honcho.session.messages"
SPAN_QUEUE_STATUS = "honcho.queue_status"
SPAN_SESSION_QUEUE_STATUS = "honcho.session.queue_status"
SPAN_LIST_PEERS = "honcho.peers.list"
SPAN_SESSION_PEERS = "honcho.session.peers"
SPAN_SESSION_SET_METADATA = "honcho.session.set_metadata"
SPAN_PEER_SET_METADATA = "honcho.peer.set_metadata"
