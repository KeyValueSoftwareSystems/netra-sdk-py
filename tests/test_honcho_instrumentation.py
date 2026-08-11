"""
Unit tests for NetraHonchoInstrumentor class and wrappers.
"""

import asyncio
from typing import Collection
from unittest.mock import MagicMock, Mock, patch

import pytest

from netra.instrumentation.honcho import NetraHonchoInstrumentor
from netra.instrumentation.honcho import constants as attrs
from netra.instrumentation.honcho.utils import should_suppress_instrumentation


class _FakeObj:
    """Lightweight stand-in for Honcho domain objects in tests.

    Unlike Mock, ``vars()`` only contains the explicitly passed fields,
    which lets ``_serialize_obj`` work the same way it does with real
    Honcho objects (Message, Conclusion, etc.).
    """

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestNetraHonchoInstrumentor:
    """Test NetraHonchoInstrumentor core functionality."""

    def test_initialization(self):
        instrumentor = NetraHonchoInstrumentor()
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")
        assert hasattr(instrumentor, "instrumentation_dependencies")

    def test_instrumentation_dependencies(self):
        instrumentor = NetraHonchoInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()
        assert isinstance(dependencies, Collection)
        assert "honcho-ai >= 2.0.0" in dependencies

    @patch("netra.instrumentation.honcho.get_tracer")
    @patch("netra.instrumentation.honcho.wrap_function_wrapper")
    def test_instrument_patches_all_methods(self, mock_wrap, mock_get_tracer):
        instrumentor = NetraHonchoInstrumentor()
        mock_get_tracer.return_value = Mock()

        instrumentor._instrument()

        mock_get_tracer.assert_called_once()
        # 27 PatchSpecs × 2 (sync + async) = 54
        assert mock_wrap.call_count == 54

    @patch("netra.instrumentation.honcho.get_tracer")
    @patch("netra.instrumentation.honcho.wrap_function_wrapper")
    def test_instrument_with_custom_tracer_provider(self, mock_wrap, mock_get_tracer):
        instrumentor = NetraHonchoInstrumentor()
        mock_tracer_provider = Mock()
        mock_get_tracer.return_value = Mock()

        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        mock_get_tracer.assert_called_once_with(
            "netra.instrumentation.honcho",
            mock_get_tracer.call_args[0][1],
            mock_tracer_provider,
        )

    @patch("netra.instrumentation.honcho.unwrap")
    def test_uninstrument(self, mock_unwrap):
        instrumentor = NetraHonchoInstrumentor()
        instrumentor._uninstrument()
        # 27 PatchSpecs × 2 (sync + async) = 54
        assert mock_unwrap.call_count == 54

    @patch("netra.instrumentation.honcho.get_tracer")
    @patch("netra.instrumentation.honcho.wrap_function_wrapper")
    def test_instrument_tracer_error_returns_early(self, mock_wrap, mock_get_tracer):
        instrumentor = NetraHonchoInstrumentor()
        mock_get_tracer.side_effect = RuntimeError("tracer init failed")

        instrumentor._instrument()

        mock_wrap.assert_not_called()

    @patch("netra.instrumentation.honcho.get_tracer")
    @patch("netra.instrumentation.honcho.wrap_function_wrapper")
    def test_instrument_wrap_failure_continues(self, mock_wrap, mock_get_tracer):
        """If one wrap fails, the rest should still be attempted."""
        instrumentor = NetraHonchoInstrumentor()
        mock_get_tracer.return_value = Mock()

        call_count = [0]

        def selective_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ImportError("module not found")

        mock_wrap.side_effect = selective_fail

        instrumentor._instrument()

        assert mock_wrap.call_count == 54


class TestSyncWrapper:
    """Test sync wrapper functionality."""

    def test_non_streaming_wrapper_creates_span_and_returns_result(self):
        from netra.instrumentation.honcho.wrappers import make_sync_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        req_fn = Mock()
        resp_fn = Mock()
        wrapped = Mock(return_value=["msg1", "msg2"])
        instance = Mock()

        wrapper = make_sync_wrapper(mock_tracer, "honcho.test.op", req_fn, resp_fn)
        result = wrapper(wrapped, instance, (), {"key": "val"})

        wrapped.assert_called_once_with(key="val")
        mock_tracer.start_as_current_span.assert_called_once()
        req_fn.assert_called_once_with(mock_span, instance, (), {"key": "val"})
        resp_fn.assert_called_once_with(mock_span, ["msg1", "msg2"])
        assert result == ["msg1", "msg2"]

    def test_non_streaming_wrapper_records_error(self):
        from netra.instrumentation.honcho.wrappers import make_sync_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        error = ValueError("API error")
        wrapped = Mock(side_effect=error)

        wrapper = make_sync_wrapper(mock_tracer, "honcho.test.op", Mock(), Mock())

        with pytest.raises(ValueError, match="API error"):
            wrapper(wrapped, Mock(), (), {})

        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once_with(error)

    @patch("netra.instrumentation.honcho.utils.context_api.get_value", return_value=True)
    def test_non_streaming_wrapper_suppressed(self, mock_get_value):
        from netra.instrumentation.honcho.wrappers import make_sync_wrapper

        mock_tracer = Mock()
        wrapped = Mock(return_value="result")

        wrapper = make_sync_wrapper(mock_tracer, "honcho.test.op", Mock(), Mock())
        result = wrapper(wrapped, Mock(), (), {})

        mock_tracer.start_as_current_span.assert_not_called()
        assert result == "result"

    def test_wrapper_tolerates_request_attr_failure(self):
        from netra.instrumentation.honcho.wrappers import make_sync_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        req_fn = Mock(side_effect=TypeError("attr extraction failed"))
        resp_fn = Mock()
        wrapped = Mock(return_value="ok")

        wrapper = make_sync_wrapper(mock_tracer, "honcho.test.op", req_fn, resp_fn)
        result = wrapper(wrapped, Mock(), (), {})

        assert result == "ok"
        wrapped.assert_called_once()

    def test_wrapper_tolerates_response_attr_failure(self):
        from netra.instrumentation.honcho.wrappers import make_sync_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        req_fn = Mock()
        resp_fn = Mock(side_effect=TypeError("attr extraction failed"))
        wrapped = Mock(return_value="ok")

        wrapper = make_sync_wrapper(mock_tracer, "honcho.test.op", req_fn, resp_fn)
        result = wrapper(wrapped, Mock(), (), {})

        assert result == "ok"
        wrapped.assert_called_once()


class TestAsyncWrapper:
    """Test async wrapper functionality."""

    def test_async_wrapper_creates_span_and_returns_result(self):
        from netra.instrumentation.honcho.wrappers import make_async_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        req_fn = Mock()
        resp_fn = Mock()

        async def mock_wrapped(*args, **kwargs):
            return ["msg1"]

        async def run():
            wrapper = make_async_wrapper(mock_tracer, "honcho.test.op", req_fn, resp_fn)
            return await wrapper(mock_wrapped, Mock(), (), {"key": "val"})

        result = asyncio.run(run())

        mock_tracer.start_as_current_span.assert_called_once()
        assert result == ["msg1"]

    def test_async_wrapper_records_error(self):
        from netra.instrumentation.honcho.wrappers import make_async_wrapper

        mock_tracer = Mock()
        mock_span_ctx = MagicMock()
        mock_span = Mock()
        mock_span_ctx.__enter__.return_value = mock_span
        mock_tracer.start_as_current_span.return_value = mock_span_ctx

        async def mock_wrapped(*args, **kwargs):
            raise ConnectionError("network error")

        async def run():
            wrapper = make_async_wrapper(mock_tracer, "honcho.test.op", Mock(), Mock())
            return await wrapper(mock_wrapped, Mock(), (), {})

        with pytest.raises(ConnectionError, match="network error"):
            asyncio.run(run())

        mock_span.set_status.assert_called_once()
        mock_span.record_exception.assert_called_once()


class TestStreamingChatWrapper:
    """Test streaming chat wrappers."""

    def test_sync_streaming_wrapper_iterates_and_finalizes_span(self):
        from netra.instrumentation.honcho.wrappers import StreamingChatWrapper

        mock_span = Mock()
        chunks = ["Hello", " ", "world"]

        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter(chunks))
        mock_response.__next__ = Mock(side_effect=chunks + [StopIteration()])
        mock_response.get_final_response.return_value = {"content": "Hello world"}
        mock_response.is_complete = True

        class FakeStream:
            def __init__(self):
                self._chunks = iter(chunks)
                self._acc = []
                self.is_complete = False

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    c = next(self._chunks)
                    self._acc.append(c)
                    return c
                except StopIteration:
                    self.is_complete = True
                    raise

            def get_final_response(self):
                return {"content": "".join(self._acc)}

        fake_stream = FakeStream()
        wrapper = StreamingChatWrapper(mock_span, fake_stream)

        collected = list(wrapper)

        assert collected == ["Hello", " ", "world"]
        mock_span.set_attribute.assert_any_call(attrs.RESPONSE_LENGTH, 11)
        mock_span.end.assert_called_once()

    def test_sync_streaming_wrapper_handles_error(self):
        from netra.instrumentation.honcho.wrappers import StreamingChatWrapper

        mock_span = Mock()

        class ErrorStream:
            def __init__(self):
                self._call = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._call += 1
                if self._call == 1:
                    return "chunk"
                raise RuntimeError("stream error")

            def get_final_response(self):
                return {"content": "chunk"}

        wrapper = StreamingChatWrapper(mock_span, ErrorStream())

        with pytest.raises(RuntimeError, match="stream error"):
            list(wrapper)

        mock_span.record_exception.assert_called_once()
        mock_span.end.assert_called_once()

    def test_sync_chat_stream_wrapper_factory(self):
        from netra.instrumentation.honcho.wrappers import make_chat_stream_sync_wrapper

        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span

        class FakeStream:
            def __iter__(self):
                return iter(["chunk"])

            def get_final_response(self):
                return {"content": "chunk"}

            @property
            def is_complete(self):
                return True

        wrapped = Mock(return_value=FakeStream())
        req_fn = Mock()

        wrapper = make_chat_stream_sync_wrapper(mock_tracer, "honcho.peer.chat_stream", req_fn)
        result = wrapper(wrapped, Mock(), (), {"query": "hello"})

        mock_tracer.start_span.assert_called_once()
        mock_span.set_attribute.assert_any_call(attrs.REQUEST_STREAM, True)
        assert hasattr(result, "__iter__")

    def test_async_streaming_wrapper_iterates_and_finalizes_span(self):
        from netra.instrumentation.honcho.wrappers import AsyncStreamingChatWrapper

        mock_span = Mock()
        chunks = ["Hello", " ", "world"]

        class FakeAsyncStream:
            def __init__(self):
                self._chunks = iter(chunks)
                self._acc = []
                self.is_complete = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    c = next(self._chunks)
                    self._acc.append(c)
                    return c
                except StopIteration:
                    self.is_complete = True
                    raise StopAsyncIteration

            def get_final_response(self):
                return {"content": "".join(self._acc)}

        async def run():
            fake_stream = FakeAsyncStream()
            wrapper = AsyncStreamingChatWrapper(mock_span, fake_stream)
            collected = []
            async for chunk in wrapper:
                collected.append(chunk)
            return collected

        collected = asyncio.run(run())

        assert collected == ["Hello", " ", "world"]
        mock_span.set_attribute.assert_any_call(attrs.RESPONSE_LENGTH, 11)
        mock_span.end.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("netra.instrumentation.honcho.utils.context_api.get_value")
    def test_should_suppress_instrumentation_true(self, mock_get_value):
        mock_get_value.return_value = True
        assert should_suppress_instrumentation() is True

    @patch("netra.instrumentation.honcho.utils.context_api.get_value")
    def test_should_suppress_instrumentation_false(self, mock_get_value):
        mock_get_value.return_value = False
        assert should_suppress_instrumentation() is False


class TestRequestAttributes:
    """Test attribute setter functions from utils."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    def test_set_chat_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_chat_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "peer-1"
        instance.workspace_id = "ws-1"

        set_chat_request_attrs(span, instance, ("What is AI?",), {"reasoning_level": "high", "target": "bob"})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CHAT
        assert captured[attrs.AGENT_ID] == "peer-1"
        assert captured[attrs.MEMORY_STORE_ID] == "ws-1"
        assert captured[attrs.MEMORY_QUERY_TEXT] == "What is AI?"
        assert captured[attrs.REQUEST_REASONING_LEVEL] == "high"
        assert captured[attrs.PEER_TARGET] == "bob"

    def test_set_add_messages_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_add_messages_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        messages = [Mock(), Mock(), Mock()]
        set_add_messages_request_attrs(span, instance, (messages,), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CREATE_MEMORY
        assert captured[attrs.CONVERSATION_ID] == "session-1"
        assert captured[attrs.MESSAGE_COUNT] == 3

    def test_set_session_context_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_session_context_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        set_session_context_request_attrs(span, instance, (), {"tokens": 2000, "summary": True, "peer_target": "alice"})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CONTEXT
        assert captured[attrs.REQUEST_TOP_K] == 2000
        assert captured[attrs.PEER_TARGET] == "alice"

    def test_set_search_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_search_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"
        type(instance).__name__ = "Session"

        set_search_request_attrs(span, instance, ("weather",), {"limit": 20})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_SEARCH_MEMORY
        assert captured[attrs.MEMORY_QUERY_TEXT] == "weather"
        assert captured[attrs.RETRIEVAL_TOP_K] == 20

    def test_set_conclusions_create_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_conclusions_create_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"
        instance.observer = "alice"
        instance.observed = "bob"

        conclusions = [{"content": "fact1"}, {"content": "fact2"}]
        set_conclusions_create_request_attrs(span, instance, (conclusions,), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CREATE_MEMORY
        assert captured[attrs.CONCLUSION_OBSERVER] == "alice"
        assert captured[attrs.CONCLUSION_OBSERVED] == "bob"
        assert captured[attrs.CONCLUSION_COUNT] == 2

    def test_set_upload_file_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_upload_file_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        set_upload_file_request_attrs(span, instance, (Mock(), "peer-1"), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_UPLOAD_FILE
        assert captured[attrs.CONVERSATION_ID] == "session-1"
        assert captured[attrs.AGENT_ID] == "peer-1"

    def test_set_get_or_create_peer_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_get_or_create_peer_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"

        set_get_or_create_peer_request_attrs(span, instance, ("user-123",), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CREATE_PEER
        assert captured[attrs.AGENT_ID] == "user-123"

    def test_set_get_or_create_session_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_get_or_create_session_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"

        set_get_or_create_session_request_attrs(span, instance, ("conv-1",), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_CREATE_SESSION
        assert captured[attrs.CONVERSATION_ID] == "conv-1"

    def test_set_get_card_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_get_card_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "alice"
        instance.workspace_id = "ws-1"

        set_get_card_request_attrs(span, instance, (), {"target": "bob"})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_GET_CARD
        assert captured[attrs.AGENT_ID] == "alice"
        assert captured[attrs.PEER_TARGET] == "bob"

    def test_set_set_card_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_set_card_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "alice"
        instance.workspace_id = "ws-1"

        set_set_card_request_attrs(span, instance, (["fact1", "fact2"],), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_SET_CARD
        assert captured[attrs.CARD_ITEM_COUNT] == 2


class TestResponseAttributes:
    """Test response attribute setters."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    def test_set_chat_response_attrs(self):
        from netra.instrumentation.honcho.utils import set_chat_response_attrs

        span, captured = self._capture_span()
        set_chat_response_attrs(span, "Hello world")
        assert captured[attrs.RESPONSE_LENGTH] == 11

    def test_set_chat_response_attrs_none(self):
        from netra.instrumentation.honcho.utils import set_chat_response_attrs

        span, captured = self._capture_span()
        set_chat_response_attrs(span, None)
        assert attrs.RESPONSE_LENGTH not in captured

    def test_set_add_messages_response_attrs_captures_message_details(self):
        from netra.instrumentation.honcho.utils import set_add_messages_response_attrs

        span, captured = self._capture_span()
        msg1 = _FakeObj(id="msg-1", content="hello", peer_id="alice", session_id="sess-1", token_count=5)
        msg2 = _FakeObj(id="msg-2", content="world", peer_id="bob", session_id="sess-1", token_count=3)
        set_add_messages_response_attrs(span, [msg1, msg2])
        assert captured[attrs.RESPONSE_MESSAGE_COUNT] == 2
        import json

        output = json.loads(captured["output"])
        assert output["message_count"] == 2
        assert len(output["messages"]) == 2
        assert output["messages"][0]["id"] == "msg-1"
        assert output["messages"][0]["content"] == "hello"
        assert output["messages"][0]["token_count"] == 5

    def test_set_session_context_response_attrs_captures_all_fields(self):
        from netra.instrumentation.honcho.utils import set_session_context_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="ctx message", peer_id="alice", session_id="sess-1", token_count=10)
        summary = _FakeObj(
            content="A concise summary",
            message_id="msg-0",
            summary_type="short",
            token_count=15,
            created_at="2026-01-01T00:00:00",
        )
        response = _FakeObj(
            session_id="sess-1",
            messages=[msg],
            summary=summary,
            peer_representation="Alice prefers dark mode",
            peer_card=["fact1", "fact2"],
        )
        set_session_context_response_attrs(span, response)

        assert captured[attrs.RESPONSE_MESSAGE_COUNT] == 1
        assert captured[attrs.RESPONSE_HAS_SUMMARY] is True
        assert captured[attrs.RESPONSE_HAS_REPRESENTATION] is True
        import json

        output = json.loads(captured["output"])
        assert output["session_id"] == "sess-1"
        assert output["summary"]["content"] == "A concise summary"
        assert output["summary"]["summary_type"] == "short"
        assert output["peer_representation"] == "Alice prefers dark mode"
        assert output["peer_card"] == ["fact1", "fact2"]

    def test_set_search_response_attrs_captures_message_details(self):
        from netra.instrumentation.honcho.utils import set_search_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="search result", peer_id="alice", session_id="sess-1", token_count=8)
        set_search_response_attrs(span, [msg])
        assert captured[attrs.RESPONSE_RESULT_COUNT] == 1
        import json

        output = json.loads(captured["output"])
        assert output["results"][0]["content"] == "search result"
        assert output["results"][0]["id"] == "msg-1"

    def test_set_card_response_attrs(self):
        from netra.instrumentation.honcho.utils import set_card_response_attrs

        span, captured = self._capture_span()
        set_card_response_attrs(span, ["fact1", "fact2"])
        assert captured[attrs.RESPONSE_CARD_ITEM_COUNT] == 2

    def test_set_peer_context_response_attrs_captures_all_fields(self):
        from netra.instrumentation.honcho.utils import set_peer_context_response_attrs

        span, captured = self._capture_span()
        response = _FakeObj(
            peer_id="alice",
            target_id="bob",
            representation="Alice is a software engineer",
            peer_card=["fact1", "fact2", "fact3"],
        )
        set_peer_context_response_attrs(span, response)

        assert captured[attrs.RESPONSE_HAS_REPRESENTATION] is True
        assert captured[attrs.RESPONSE_PEER_CARD_COUNT] == 3
        import json

        output = json.loads(captured["output"])
        assert output["peer_id"] == "alice"
        assert output["target_id"] == "bob"
        assert output["representation"] == "Alice is a software engineer"
        assert output["peer_card"] == ["fact1", "fact2", "fact3"]


class TestPaginatedResponseHandling:
    """Test that response attribute setters handle SyncPage objects correctly."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    @staticmethod
    def _make_sync_page(items, total=None, page=1, size=10, pages=1):
        """Create a mock SyncPage object."""
        pg = Mock()
        type(pg).__name__ = "SyncPage"
        pg.items = items
        pg.total = total if total is not None else len(items)
        pg.page = page
        pg.size = size
        pg.pages = pages
        pg.__len__ = Mock(return_value=len(items))
        return pg

    def test_search_response_with_sync_page(self):
        from netra.instrumentation.honcho.utils import set_search_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="found", peer_id="alice", session_id="sess-1", token_count=3)
        page = self._make_sync_page([msg], total=10)
        set_search_response_attrs(span, page)
        assert captured[attrs.RESPONSE_RESULT_COUNT] == 10
        import json

        output = json.loads(captured["output"])
        assert output["results"][0]["id"] == "msg-1"
        assert output["results"][0]["content"] == "found"

    def test_search_response_with_list(self):
        from netra.instrumentation.honcho.utils import set_search_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="result", peer_id="alice", session_id="sess-1", token_count=4)
        set_search_response_attrs(span, [msg])
        assert captured[attrs.RESPONSE_RESULT_COUNT] == 1

    def test_list_peers_response_with_sync_page_captures_peer_data(self):
        from netra.instrumentation.honcho.utils import set_list_peers_response_attrs

        span, captured = self._capture_span()
        peer1 = _FakeObj(id="alice", workspace_id="ws-1", metadata={"role": "user"})
        peer2 = _FakeObj(id="bob", workspace_id="ws-1")
        page = self._make_sync_page([peer1, peer2], total=5, page=1, size=10, pages=1)
        set_list_peers_response_attrs(span, page)
        assert captured[attrs.RESPONSE_PEER_COUNT] == 5
        import json

        output = json.loads(captured["output"])
        assert len(output["peers"]) == 2
        assert output["peers"][0]["id"] == "alice"
        assert output["peers"][0]["workspace_id"] == "ws-1"
        assert output["page"] == 1

    def test_messages_response_with_sync_page_captures_pagination(self):
        from netra.instrumentation.honcho.utils import set_messages_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="hello", peer_id="alice", session_id="sess-1", token_count=5)
        page = self._make_sync_page([msg], total=20, page=2, size=10, pages=2)
        set_messages_response_attrs(span, page)
        assert captured[attrs.RESPONSE_MESSAGE_COUNT] == 20
        import json

        output = json.loads(captured["output"])
        assert output["page"] == 2
        assert output["size"] == 10
        assert output["pages"] == 2
        assert output["messages"][0]["id"] == "msg-1"

    def test_add_messages_response_with_list(self):
        from netra.instrumentation.honcho.utils import set_add_messages_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="stored", peer_id="alice", session_id="sess-1", token_count=4)
        set_add_messages_response_attrs(span, [msg])
        assert captured[attrs.RESPONSE_MESSAGE_COUNT] == 1
        assert "output" in captured

    def test_conclusions_create_response_captures_details(self):
        from netra.instrumentation.honcho.utils import set_conclusions_create_response_attrs

        span, captured = self._capture_span()
        c1 = _FakeObj(
            id="conc-1",
            content="Alice likes coffee",
            observer_id="alice",
            observed_id="bob",
            session_id="sess-1",
            level="explicit",
        )
        set_conclusions_create_response_attrs(span, [c1])
        assert captured[attrs.RESPONSE_CONCLUSION_COUNT] == 1
        import json

        output = json.loads(captured["output"])
        assert output["conclusions"][0]["id"] == "conc-1"
        assert output["conclusions"][0]["content"] == "Alice likes coffee"
        assert output["conclusions"][0]["level"] == "explicit"

    def test_conclusions_response_attrs_with_sync_page(self):
        from netra.instrumentation.honcho.utils import set_conclusions_response_attrs

        span, captured = self._capture_span()
        c1 = _FakeObj(id="conc-1", content="fact", observer_id="alice", observed_id="bob", level="deductive")
        page = self._make_sync_page([c1], total=5, page=1, size=10, pages=1)
        set_conclusions_response_attrs(span, page)
        assert captured[attrs.RESPONSE_CONCLUSION_COUNT] == 5
        import json

        output = json.loads(captured["output"])
        assert output["conclusions"][0]["level"] == "deductive"
        assert output["page"] == 1

    def test_response_attrs_handle_none(self):
        from netra.instrumentation.honcho.utils import set_search_response_attrs

        span, captured = self._capture_span()
        set_search_response_attrs(span, None)
        assert attrs.RESPONSE_RESULT_COUNT not in captured

    def test_representation_response_attrs(self):
        from netra.instrumentation.honcho.utils import set_representation_response_attrs

        span, captured = self._capture_span()
        set_representation_response_attrs(span, "Alice likes dark mode and prefers concise answers.")
        assert captured["output"] == "Alice likes dark mode and prefers concise answers."

    def test_get_or_create_peer_response_captures_full_data(self):
        from netra.instrumentation.honcho.utils import set_get_or_create_peer_response_attrs

        span, captured = self._capture_span()
        peer = _FakeObj(id="alice", workspace_id="ws-1", metadata={"role": "assistant"})
        set_get_or_create_peer_response_attrs(span, peer)
        assert captured[attrs.AGENT_ID] == "alice"
        import json

        output = json.loads(captured["output"])
        assert output["id"] == "alice"
        assert output["workspace_id"] == "ws-1"
        assert output["metadata"] == {"role": "assistant"}

    def test_get_or_create_session_response_captures_full_data(self):
        from netra.instrumentation.honcho.utils import set_get_or_create_session_response_attrs

        span, captured = self._capture_span()
        session = _FakeObj(id="session-1", workspace_id="ws-1", metadata={"topic": "greetings"}, is_active=True)
        set_get_or_create_session_response_attrs(span, session)
        assert captured[attrs.CONVERSATION_ID] == "session-1"
        assert captured[attrs.SESSION_IS_ACTIVE] is True
        import json

        output = json.loads(captured["output"])
        assert output["id"] == "session-1"
        assert output["is_active"] is True
        assert output["metadata"] == {"topic": "greetings"}

    def test_queue_status_response_captures_all_fields(self):
        from netra.instrumentation.honcho.utils import set_queue_status_response_attrs

        span, captured = self._capture_span()
        session_status = _FakeObj(
            session_id="sess-1",
            total_work_units=5,
            completed_work_units=3,
            in_progress_work_units=1,
            pending_work_units=1,
        )
        response = _FakeObj(
            total_work_units=10,
            completed_work_units=7,
            in_progress_work_units=1,
            pending_work_units=2,
            sessions={"sess-1": session_status},
        )
        set_queue_status_response_attrs(span, response)
        import json

        output = json.loads(captured["output"])
        assert output["total_work_units"] == 10
        assert output["in_progress_work_units"] == 1
        assert output["sessions"]["sess-1"]["session_id"] == "sess-1"


class TestUploadFileResponse:
    """Test upload_file response captures message details."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    def test_upload_file_response_captures_message_data(self):
        from netra.instrumentation.honcho.utils import set_upload_file_response_attrs

        span, captured = self._capture_span()
        msg = _FakeObj(id="msg-1", content="file content", peer_id="alice", session_id="sess-1", token_count=50)
        set_upload_file_response_attrs(span, [msg])
        assert captured[attrs.RESPONSE_MESSAGE_COUNT] == 1
        import json

        output = json.loads(captured["output"])
        assert output["messages"][0]["id"] == "msg-1"
        assert output["messages"][0]["content"] == "file content"
        assert output["messages"][0]["token_count"] == 50


class TestNewOperationAttributes:
    """Test attribute setters for newly added operations."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    def test_set_list_peers_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_list_peers_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"

        set_list_peers_request_attrs(span, instance, (), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_LIST_PEERS
        assert captured[attrs.MEMORY_STORE_ID] == "ws-1"
        assert "input" in captured

    def test_set_list_peers_response_attrs(self):
        from netra.instrumentation.honcho.utils import set_list_peers_response_attrs

        span, captured = self._capture_span()
        peer1 = _FakeObj(id="alice", workspace_id="ws-1", metadata={"role": "user"})
        peer2 = _FakeObj(id="bob", workspace_id="ws-1")
        set_list_peers_response_attrs(span, [peer1, peer2])

        assert captured[attrs.RESPONSE_PEER_COUNT] == 2
        import json

        output = json.loads(captured["output"])
        assert output["peers"][0]["id"] == "alice"
        assert output["peers"][0]["workspace_id"] == "ws-1"

    def test_set_session_peers_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_session_peers_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        set_session_peers_request_attrs(span, instance, (), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_SESSION_PEERS
        assert captured[attrs.CONVERSATION_ID] == "session-1"
        assert "input" in captured

    def test_set_session_peers_response_attrs(self):
        from netra.instrumentation.honcho.utils import set_session_peers_response_attrs

        span, captured = self._capture_span()
        peer1 = _FakeObj(id="alice", workspace_id="ws-1")
        set_session_peers_response_attrs(span, [peer1])

        assert captured[attrs.RESPONSE_PEER_COUNT] == 1
        import json

        output = json.loads(captured["output"])
        assert output["peers"][0]["id"] == "alice"

    def test_set_session_set_metadata_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_session_set_metadata_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        set_session_set_metadata_request_attrs(span, instance, ({"topic": "greetings"},), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_SET_METADATA
        assert captured[attrs.CONVERSATION_ID] == "session-1"
        assert "input" in captured

    def test_set_peer_set_metadata_request_attrs(self):
        from netra.instrumentation.honcho.utils import set_peer_set_metadata_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "alice"
        instance.workspace_id = "ws-1"

        set_peer_set_metadata_request_attrs(span, instance, ({"role": "assistant"},), {})

        assert captured[attrs.OPERATION_NAME] == attrs.OP_SET_METADATA
        assert captured[attrs.AGENT_ID] == "alice"
        assert "input" in captured


class TestInputOutputAttributes:
    """Test that input/output span attributes are set by all attr functions."""

    @staticmethod
    def _capture_span():
        span = Mock()
        span.is_recording.return_value = True
        captured = {}
        span.set_attribute.side_effect = lambda k, v: captured.__setitem__(k, v)
        return span, captured

    def test_chat_sets_input_and_output(self):
        from netra.instrumentation.honcho.utils import set_chat_request_attrs, set_chat_response_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "peer-1"
        instance.workspace_id = "ws-1"

        set_chat_request_attrs(span, instance, ("What is AI?",), {})
        assert "input" in captured
        assert "What is AI?" in captured["input"]

        set_chat_response_attrs(span, "AI is artificial intelligence.")
        assert "output" in captured
        assert captured["output"] == "AI is artificial intelligence."

    def test_add_messages_sets_input_and_output(self):
        from netra.instrumentation.honcho.utils import set_add_messages_request_attrs, set_add_messages_response_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        msg = _FakeObj(id="msg-1", content="hello", peer_id="alice", session_id="session-1", token_count=5)
        set_add_messages_request_attrs(span, instance, ([msg],), {})
        assert "input" in captured

        resp_msg = _FakeObj(id="msg-2", content="stored", peer_id="alice", session_id="session-1", token_count=4)
        set_add_messages_response_attrs(span, [resp_msg])
        assert "output" in captured

    def test_search_sets_input_and_output(self):
        from netra.instrumentation.honcho.utils import set_search_request_attrs, set_search_response_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"
        type(instance).__name__ = "Peer"
        instance.id = "alice"

        set_search_request_attrs(span, instance, ("weather",), {"limit": 5})
        assert "input" in captured
        assert "weather" in captured["input"]

        set_search_response_attrs(span, [_FakeObj(id="r1", content="result")])
        assert "output" in captured

    def test_get_or_create_peer_sets_input(self):
        from netra.instrumentation.honcho.utils import set_get_or_create_peer_request_attrs

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"

        set_get_or_create_peer_request_attrs(span, instance, ("alice",), {})
        assert "input" in captured
        assert "alice" in captured["input"]

    def test_conclusions_create_sets_input_and_output(self):
        from netra.instrumentation.honcho.utils import (
            set_conclusions_create_request_attrs,
            set_conclusions_create_response_attrs,
        )

        span, captured = self._capture_span()
        instance = Mock()
        instance.workspace_id = "ws-1"
        instance.observer = "alice"
        instance.observed = "bob"

        set_conclusions_create_request_attrs(span, instance, ([{"content": "fact"}],), {})
        assert "input" in captured

        set_conclusions_create_response_attrs(span, [_FakeObj(id="c1", content="fact")])
        assert "output" in captured

    def test_session_context_sets_input_and_output(self):
        from netra.instrumentation.honcho.utils import (
            set_session_context_request_attrs,
            set_session_context_response_attrs,
        )

        span, captured = self._capture_span()
        instance = Mock()
        instance.id = "session-1"
        instance.workspace_id = "ws-1"

        set_session_context_request_attrs(span, instance, (), {"tokens": 2000})
        assert "input" in captured

        msg = _FakeObj(id="msg-1", content="ctx", peer_id="alice", session_id="session-1", token_count=3)
        response = _FakeObj(session_id="session-1", messages=[msg], summary="A summary")
        set_session_context_response_attrs(span, response)
        assert "output" in captured

    def test_streaming_sets_output(self):
        from netra.instrumentation.honcho.wrappers import StreamingChatWrapper

        span, captured = self._capture_span()

        class FakeStream:
            def __init__(self):
                self._chunks = iter(["Hello", " ", "world"])
                self._acc = []

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    c = next(self._chunks)
                    self._acc.append(c)
                    return c
                except StopIteration:
                    raise

            def get_final_response(self):
                return {"content": "".join(self._acc)}

        wrapper = StreamingChatWrapper(span, FakeStream())
        list(wrapper)

        assert captured.get("output") == "Hello world"


class TestIntegration:
    """Integration tests verifying instrument/uninstrument cycle."""

    def test_instrument_and_uninstrument_cycle(self):
        instrumentor = NetraHonchoInstrumentor()

        instrumentor.instrument()
        assert instrumentor.is_instrumented_by_opentelemetry

        from honcho.session import Session

        assert hasattr(Session.add_messages, "__wrapped__")

        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented_by_opentelemetry

    def test_idempotent_instrument(self):
        """Calling instrument() twice should not double-wrap methods."""
        instrumentor = NetraHonchoInstrumentor()
        instrumentor.instrument()

        from honcho.peer import Peer

        Peer.chat.__wrapped__ if hasattr(Peer.chat, "__wrapped__") else None

        instrumentor2 = NetraHonchoInstrumentor()
        # BaseInstrumentor guards against double-instrumentation
        assert instrumentor2.is_instrumented_by_opentelemetry

        instrumentor.uninstrument()
