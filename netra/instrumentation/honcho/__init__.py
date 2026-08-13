import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from netra.instrumentation.honcho import constants as attrs
from netra.instrumentation.honcho.utils import (
    _noop_response_attrs,
    set_add_messages_request_attrs,
    set_add_messages_response_attrs,
    set_add_peers_request_attrs,
    set_card_response_attrs,
    set_chat_request_attrs,
    set_chat_response_attrs,
    set_conclusions_create_request_attrs,
    set_conclusions_create_response_attrs,
    set_conclusions_delete_request_attrs,
    set_conclusions_list_request_attrs,
    set_conclusions_query_request_attrs,
    set_conclusions_response_attrs,
    set_get_card_request_attrs,
    set_get_or_create_peer_request_attrs,
    set_get_or_create_peer_response_attrs,
    set_get_or_create_session_request_attrs,
    set_get_or_create_session_response_attrs,
    set_list_peers_request_attrs,
    set_list_peers_response_attrs,
    set_messages_request_attrs,
    set_messages_response_attrs,
    set_peer_context_request_attrs,
    set_peer_context_response_attrs,
    set_peer_set_metadata_request_attrs,
    set_queue_status_request_attrs,
    set_queue_status_response_attrs,
    set_representation_request_attrs,
    set_representation_response_attrs,
    set_search_request_attrs,
    set_search_response_attrs,
    set_session_context_request_attrs,
    set_session_context_response_attrs,
    set_session_peers_request_attrs,
    set_session_peers_response_attrs,
    set_session_set_metadata_request_attrs,
    set_set_card_request_attrs,
    set_upload_file_request_attrs,
    set_upload_file_response_attrs,
)
from netra.instrumentation.honcho.version import __version__
from netra.instrumentation.honcho.wrappers import (
    make_async_wrapper,
    make_chat_stream_async_wrapper,
    make_chat_stream_sync_wrapper,
    make_sync_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("honcho-ai >= 2.0.0",)

_NOOP = _noop_response_attrs


@dataclass(frozen=True, slots=True)
class PatchSpec:
    """Declares a single Honcho method to instrument.

    By default each spec produces **two** patches (sync + async) derived
    from ``cls_method`` via ``_SYNC_MODULE_MAP`` / ``_ASYNC_CLASS_MAP``.

    Override fields let you handle SDK deviations without touching the
    derivation logic:

    * ``sync_module_override`` / ``async_module_override`` — use a
      non-standard module path for one side.
    * ``async_cls_method_override`` — use a different async class or
      method name than the convention (``<Class>Aio.<method>``).
    * ``patch_sync`` / ``patch_async`` — set to ``False`` to skip
      patching one side entirely (e.g. an async-only API).
    * ``streaming`` — use the streaming wrapper factory instead of the
      standard request/response one.
    """

    cls_method: str  # e.g. "Session.add_messages"
    span_name: str  # e.g. "honcho.session.add_messages"
    request_attrs: Callable[..., None]
    response_attrs: Callable[..., None] = field(default=_NOOP)
    streaming: bool = False

    sync_module_override: str | None = None
    async_module_override: str | None = None
    async_cls_method_override: str | None = None

    patch_sync: bool = True
    patch_async: bool = True

    @property
    def _cls_name(self) -> str:
        return self.cls_method.split(".")[0]

    @property
    def sync_module(self) -> str:
        return self.sync_module_override or attrs.SYNC_MODULE_MAP[self._cls_name]

    @property
    def async_module(self) -> str:
        return self.async_module_override or attrs.ASYNC_MODULE

    @property
    def async_cls_method(self) -> str:
        if self.async_cls_method_override:
            return self.async_cls_method_override
        cls_name, method = self.cls_method.split(".", 1)
        return f"{attrs.ASYNC_CLASS_MAP[cls_name]}.{method}"


PATCH_SPECS: tuple[PatchSpec, ...] = (
    # Memory Ingestion
    PatchSpec(
        "Session.add_messages", attrs.SPAN_ADD_MESSAGES, set_add_messages_request_attrs, set_add_messages_response_attrs
    ),
    PatchSpec(
        "Session.upload_file", attrs.SPAN_UPLOAD_FILE, set_upload_file_request_attrs, set_upload_file_response_attrs
    ),
    PatchSpec("Peer.set_card", attrs.SPAN_SET_CARD, set_set_card_request_attrs, set_card_response_attrs),
    PatchSpec(
        "ConclusionScope.create",
        attrs.SPAN_CONCLUSIONS_CREATE,
        set_conclusions_create_request_attrs,
        set_conclusions_create_response_attrs,
    ),
    # Memory Retrieval
    PatchSpec("Peer.chat", attrs.SPAN_CHAT, set_chat_request_attrs, set_chat_response_attrs),
    PatchSpec("Peer.chat_stream", attrs.SPAN_CHAT_STREAM, set_chat_request_attrs, streaming=True),
    PatchSpec(
        "Session.context",
        attrs.SPAN_SESSION_CONTEXT,
        set_session_context_request_attrs,
        set_session_context_response_attrs,
    ),
    PatchSpec("Peer.context", attrs.SPAN_PEER_CONTEXT, set_peer_context_request_attrs, set_peer_context_response_attrs),
    PatchSpec(
        "Peer.representation",
        attrs.SPAN_PEER_REPRESENTATION,
        set_representation_request_attrs,
        set_representation_response_attrs,
    ),
    PatchSpec(
        "Session.representation",
        attrs.SPAN_SESSION_REPRESENTATION,
        set_representation_request_attrs,
        set_representation_response_attrs,
    ),
    PatchSpec("Peer.search", attrs.SPAN_PEER_SEARCH, set_search_request_attrs, set_search_response_attrs),
    PatchSpec("Session.search", attrs.SPAN_SESSION_SEARCH, set_search_request_attrs, set_search_response_attrs),
    PatchSpec("Honcho.search", attrs.SPAN_WORKSPACE_SEARCH, set_search_request_attrs, set_search_response_attrs),
    PatchSpec("Peer.get_card", attrs.SPAN_GET_CARD, set_get_card_request_attrs, set_card_response_attrs),
    PatchSpec(
        "ConclusionScope.list",
        attrs.SPAN_CONCLUSIONS_LIST,
        set_conclusions_list_request_attrs,
        set_conclusions_response_attrs,
    ),
    PatchSpec(
        "ConclusionScope.query",
        attrs.SPAN_CONCLUSIONS_QUERY,
        set_conclusions_query_request_attrs,
        set_conclusions_response_attrs,
    ),
    PatchSpec("ConclusionScope.delete", attrs.SPAN_CONCLUSIONS_DELETE, set_conclusions_delete_request_attrs),
    # Setup / Lifecycle
    PatchSpec(
        "Honcho.peer",
        attrs.SPAN_GET_OR_CREATE_PEER,
        set_get_or_create_peer_request_attrs,
        set_get_or_create_peer_response_attrs,
    ),
    PatchSpec(
        "Honcho.session",
        attrs.SPAN_GET_OR_CREATE_SESSION,
        set_get_or_create_session_request_attrs,
        set_get_or_create_session_response_attrs,
    ),
    PatchSpec("Session.add_peers", attrs.SPAN_ADD_PEERS, set_add_peers_request_attrs),
    PatchSpec("Session.messages", attrs.SPAN_LIST_MESSAGES, set_messages_request_attrs, set_messages_response_attrs),
    PatchSpec(
        "Honcho.queue_status", attrs.SPAN_QUEUE_STATUS, set_queue_status_request_attrs, set_queue_status_response_attrs
    ),
    PatchSpec(
        "Session.queue_status",
        attrs.SPAN_SESSION_QUEUE_STATUS,
        set_queue_status_request_attrs,
        set_queue_status_response_attrs,
    ),
    # Peers & Metadata
    PatchSpec("Honcho.peers", attrs.SPAN_LIST_PEERS, set_list_peers_request_attrs, set_list_peers_response_attrs),
    PatchSpec(
        "Session.peers", attrs.SPAN_SESSION_PEERS, set_session_peers_request_attrs, set_session_peers_response_attrs
    ),
    PatchSpec("Session.set_metadata", attrs.SPAN_SESSION_SET_METADATA, set_session_set_metadata_request_attrs),
    PatchSpec("Peer.set_metadata", attrs.SPAN_PEER_SET_METADATA, set_peer_set_metadata_request_attrs),
)


class NetraHonchoInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """OpenTelemetry instrumentor for the Honcho memory SDK (honcho-ai)."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            tracer_provider = kwargs.get("tracer_provider")
            tracer = get_tracer(__name__, __version__, tracer_provider)
        except Exception as e:
            logger.error("Failed to initialize Honcho tracer: %s", e)
            return

        for spec in PATCH_SPECS:
            self._apply_patch(tracer, spec)

    def _apply_patch(self, tracer: Any, spec: PatchSpec) -> None:
        if spec.streaming:
            sync_factory = make_chat_stream_sync_wrapper(tracer, spec.span_name, spec.request_attrs)
            async_factory = make_chat_stream_async_wrapper(tracer, spec.span_name, spec.request_attrs)
        else:
            sync_factory = make_sync_wrapper(tracer, spec.span_name, spec.request_attrs, spec.response_attrs)
            async_factory = make_async_wrapper(tracer, spec.span_name, spec.request_attrs, spec.response_attrs)

        if spec.patch_sync:
            try:
                wrap_function_wrapper(spec.sync_module, spec.cls_method, sync_factory)
            except Exception as e:
                logger.debug("Failed to instrument %s.%s: %s", spec.sync_module, spec.cls_method, e)

        if spec.patch_async:
            try:
                wrap_function_wrapper(spec.async_module, spec.async_cls_method, async_factory)
            except Exception as e:
                logger.debug("Failed to instrument %s.%s: %s", spec.async_module, spec.async_cls_method, e)

    def _uninstrument(self, **kwargs: Any) -> None:
        for spec in PATCH_SPECS:
            targets: list[tuple[str, str]] = []
            if spec.patch_sync:
                targets.append((spec.sync_module, spec.cls_method))
            if spec.patch_async:
                targets.append((spec.async_module, spec.async_cls_method))

            for module, cls_method in targets:
                try:
                    unwrap(module, cls_method)
                except (AttributeError, ModuleNotFoundError):
                    logger.debug("Failed to uninstrument %s.%s", module, cls_method)
