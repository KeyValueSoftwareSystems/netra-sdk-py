"""Tests for netra.instrumentation.capture.stream_utils stream wrappers.

Covers both sync and async wrappers, verifying:
- True iterators (single-pass streams) are wrapped correctly.
- Re-iterable collections (lists, etc.) are handled eagerly, not wrapped.
- ``_commit`` fires on full exhaustion, early ``break``, and context-manager exit.
- ``_netra_extractor`` and ``_generic_extractor`` paths.
"""

import asyncio
from typing import Any, AsyncIterator, Iterator, List
from unittest.mock import MagicMock

import pytest

from netra.instrumentation.capture.stream_utils import (
    RootOutputAsyncStreamWrapper,
    RootOutputSyncStreamWrapper,
    _generic_extractor,
    _netra_extractor,
    wrap_stream_for_root_output,
)


def _make_commit_fn() -> MagicMock:
    """Return a mock callable to use as the ``commit_fn`` callback."""
    return MagicMock()


class _SyncIterable:
    """Plain iterable (has ``__iter__`` but NOT ``__next__``)."""

    def __init__(self, items: List[Any]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)


class _SyncIterator:
    """Iterator (has both ``__iter__`` and ``__next__``)."""

    def __init__(self, items: List[Any]) -> None:
        self._items = iter(items)

    def __iter__(self) -> "_SyncIterator":
        return self

    def __next__(self) -> Any:
        return next(self._items)


class _AsyncIterable:
    """Async iterable (has ``__aiter__`` but NOT ``__anext__``)."""

    def __init__(self, items: List[Any]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[Any]:
        return _AsyncIterator(self._items)


class _AsyncIterator:
    """Async iterator (has both ``__aiter__`` and ``__anext__``)."""

    def __init__(self, items: List[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncIterator":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class _NetraSyncStream:
    """Simulates a Netra-instrumented sync stream with ``_netra_output``."""

    _netra_stream_wrapper = True

    def __init__(self, items: List[Any], output: Any) -> None:
        self._items = items
        self._netra_output = output

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)


class _NetraAsyncStream:
    """Simulates a Netra-instrumented async stream with ``_netra_output``."""

    _netra_stream_wrapper = True

    def __init__(self, items: List[Any], output: Any) -> None:
        self._items = items
        self._netra_output = output

    def __aiter__(self) -> AsyncIterator[Any]:
        return _AsyncIterator(self._items)


# Sync wrapper tests


class TestRootOutputSyncStreamWrapper:

    def test_iterable_full_exhaustion(self) -> None:
        """A plain iterable (not an iterator) is consumed and commit fires."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable(["a", "b", "c"]), commit_fn, _generic_extractor)
        result = list(wrapper)
        assert result == ["a", "b", "c"]
        assert wrapper._committed is True
        commit_fn.assert_called_once()

    def test_iterator_full_exhaustion(self) -> None:
        """A plain iterator is consumed and commit fires."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterator(["x", "y"]), commit_fn, _generic_extractor)
        result = list(wrapper)
        assert result == ["x", "y"]
        assert wrapper._committed is True

    def test_break_triggers_commit(self) -> None:
        """``for x in stream: break`` must trigger ``_commit``."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable([1, 2, 3]), commit_fn, _generic_extractor)
        for _ in wrapper:
            break
        assert wrapper._committed is True

    def test_break_records_partial_output(self) -> None:
        """Only chunks yielded before the break are recorded."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable(["a", "b", "c"]), commit_fn, _generic_extractor)
        collected = []
        for chunk in wrapper:
            collected.append(chunk)
            break
        assert collected == ["a"]
        assert wrapper._chunks == ["a"]
        assert wrapper._committed is True

    def test_next_calls_work(self) -> None:
        """Direct ``next(wrapper)`` calls work and commit on StopIteration."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterator(["only"]), commit_fn, _generic_extractor)
        assert next(wrapper) == "only"
        with pytest.raises(StopIteration):
            next(wrapper)
        assert wrapper._committed is True

    def test_context_manager_commit(self) -> None:
        """Exiting a ``with`` block triggers ``_commit``."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable(["a"]), commit_fn, _generic_extractor)
        with wrapper:
            pass
        assert wrapper._committed is True

    def test_netra_extractor_path(self) -> None:
        """When wrapping a Netra-instrumented stream, ``_netra_extractor`` reads ``_netra_output``."""
        commit_fn = _make_commit_fn()
        inner = _NetraSyncStream(["chunk1"], output="full_output_value")
        wrapper = RootOutputSyncStreamWrapper(inner, commit_fn, _netra_extractor)
        list(wrapper)
        assert wrapper._committed is True
        commit_fn.assert_called_once_with("full_output_value")

    def test_generic_extractor_concatenates(self) -> None:
        """Generic extractor concatenates str(chunk) values."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable(["a", "b"]), commit_fn, _generic_extractor)
        list(wrapper)
        assert _generic_extractor(wrapper) == "ab"

    def test_getattr_proxies_to_stream(self) -> None:
        """Unknown attributes are proxied to the underlying stream."""
        commit_fn = _make_commit_fn()
        inner = _NetraSyncStream([], output="x")
        wrapper = RootOutputSyncStreamWrapper(inner, commit_fn, _netra_extractor)
        assert wrapper._netra_stream_wrapper is True
        assert wrapper._netra_output == "x"

    def test_commit_is_idempotent(self) -> None:
        """Calling ``_commit`` multiple times only invokes commit_fn once."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable(["a"]), commit_fn, _generic_extractor)
        list(wrapper)
        assert wrapper._committed is True
        call_count_after_first = commit_fn.call_count
        wrapper._commit()
        assert commit_fn.call_count == call_count_after_first

    def test_empty_stream(self) -> None:
        """An empty iterable commits with no chunks."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputSyncStreamWrapper(_SyncIterable([]), commit_fn, _generic_extractor)
        assert list(wrapper) == []
        assert wrapper._committed is True
        assert wrapper._chunks == []


# Async wrapper tests


class TestRootOutputAsyncStreamWrapper:

    def test_async_iterable_full_exhaustion(self) -> None:
        """An async iterable (not iterator) is consumed and commit fires."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterable(["a", "b"]), commit_fn, _generic_extractor)

        async def _consume() -> List[Any]:
            result = []
            async for chunk in wrapper:
                result.append(chunk)
            return result

        result = asyncio.run(_consume())
        assert result == ["a", "b"]
        assert wrapper._committed is True

    def test_async_iterator_full_exhaustion(self) -> None:
        """An async iterator is consumed and commit fires."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterator(["x"]), commit_fn, _generic_extractor)

        async def _consume() -> List[Any]:
            result = []
            async for chunk in wrapper:
                result.append(chunk)
            return result

        result = asyncio.run(_consume())
        assert result == ["x"]
        assert wrapper._committed is True

    def test_async_break_triggers_commit(self) -> None:
        """``async for x in stream: break`` must trigger ``_commit``.

        This test passes because ``asyncio.run()`` force-finalizes async
        generators on shutdown (via ``loop.shutdown_asyncgens()``).  In a
        long-lived event loop (e.g. a web server), the ``finally``/commit
        is deferred to a future event loop iteration, which may run after
        the root span has already ended — see the
        ``RootOutputAsyncStreamWrapper`` class docstring for details.
        """
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterable([1, 2, 3]), commit_fn, _generic_extractor)

        async def _break_early() -> None:
            async for _ in wrapper:
                break

        asyncio.run(_break_early())
        assert wrapper._committed is True

    def test_async_break_records_partial_output(self) -> None:
        """Only chunks yielded before the async break are recorded."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterable(["a", "b", "c"]), commit_fn, _generic_extractor)

        async def _break_early() -> List[Any]:
            collected = []
            async for chunk in wrapper:
                collected.append(chunk)
                break
            return collected

        result = asyncio.run(_break_early())
        assert result == ["a"]
        assert wrapper._chunks == ["a"]
        assert wrapper._committed is True

    def test_async_anext_calls_work(self) -> None:
        """Direct ``await wrapper.__anext__()`` calls work."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterator(["only"]), commit_fn, _generic_extractor)

        async def _direct() -> Any:
            val = await wrapper.__anext__()
            with pytest.raises(StopAsyncIteration):
                await wrapper.__anext__()
            return val

        result = asyncio.run(_direct())
        assert result == "only"
        assert wrapper._committed is True

    def test_async_context_manager_commit(self) -> None:
        """Exiting an ``async with`` block triggers ``_commit``."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterable(["a"]), commit_fn, _generic_extractor)

        async def _ctx() -> None:
            async with wrapper:
                pass

        asyncio.run(_ctx())
        assert wrapper._committed is True

    def test_async_netra_extractor_path(self) -> None:
        """Netra extractor reads ``_netra_output`` from the inner async stream."""
        commit_fn = _make_commit_fn()
        inner = _NetraAsyncStream(["c"], output="async_full_output")
        wrapper = RootOutputAsyncStreamWrapper(inner, commit_fn, _netra_extractor)

        async def _consume() -> None:
            async for _ in wrapper:
                pass

        asyncio.run(_consume())
        assert wrapper._committed is True
        commit_fn.assert_called_once_with("async_full_output")

    def test_async_empty_stream(self) -> None:
        """An empty async iterable commits with no chunks."""
        commit_fn = _make_commit_fn()
        wrapper = RootOutputAsyncStreamWrapper(_AsyncIterable([]), commit_fn, _generic_extractor)

        async def _consume() -> List[Any]:
            return [x async for x in wrapper]

        result = asyncio.run(_consume())
        assert result == []
        assert wrapper._committed is True


# wrap_stream_for_root_output tests


class TestWrapStreamForRootOutput:

    def test_sync_iterator_produces_sync_wrapper(self) -> None:
        """A true sync iterator (has ``__next__``) is wrapped."""
        commit_fn = _make_commit_fn()
        wrapped = wrap_stream_for_root_output(_SyncIterator([1]), commit_fn)
        assert isinstance(wrapped, RootOutputSyncStreamWrapper)

    def test_async_iterator_produces_async_wrapper(self) -> None:
        """A true async iterator (has ``__anext__``) is wrapped."""
        commit_fn = _make_commit_fn()
        wrapped = wrap_stream_for_root_output(_AsyncIterator([1]), commit_fn)
        assert isinstance(wrapped, RootOutputAsyncStreamWrapper)

    def test_sync_iterable_sets_output_eagerly(self) -> None:
        """A re-iterable (list-like, no ``__next__``) commits eagerly and is returned unchanged."""
        commit_fn = _make_commit_fn()
        original = _SyncIterable(["a", "b"])
        result = wrap_stream_for_root_output(original, commit_fn)
        assert result is original
        commit_fn.assert_called_once_with(original)

    def test_list_sets_output_eagerly(self) -> None:
        """A plain list is not wrapped — output is committed eagerly."""
        commit_fn = _make_commit_fn()
        data = ["item1", "item2", "item3"]
        result = wrap_stream_for_root_output(data, commit_fn)
        assert result is data
        commit_fn.assert_called_once_with(data)

    def test_async_iterable_sets_output_eagerly(self) -> None:
        """A re-iterable async object (no ``__anext__``) commits eagerly."""
        commit_fn = _make_commit_fn()
        original = _AsyncIterable(["a"])
        result = wrap_stream_for_root_output(original, commit_fn)
        assert result is original
        commit_fn.assert_called_once_with(original)

    def test_non_iterable_returned_unchanged(self) -> None:
        commit_fn = _make_commit_fn()
        obj = 42
        result = wrap_stream_for_root_output(obj, commit_fn)
        assert result is obj
        commit_fn.assert_not_called()

    def test_netra_stream_uses_netra_extractor(self) -> None:
        """Netra-wrapped objects are always wrapped, even without ``__next__``."""
        commit_fn = _make_commit_fn()
        inner = _NetraSyncStream(["c"], output="netra_out")
        wrapped = wrap_stream_for_root_output(inner, commit_fn)
        assert isinstance(wrapped, RootOutputSyncStreamWrapper)
        assert wrapped._extractor is _netra_extractor

    def test_generator_is_wrapped(self) -> None:
        """A generator (has ``__next__``) is treated as a stream and wrapped."""
        commit_fn = _make_commit_fn()
        gen = (x for x in [1, 2, 3])
        wrapped = wrap_stream_for_root_output(gen, commit_fn)
        assert isinstance(wrapped, RootOutputSyncStreamWrapper)
