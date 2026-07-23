"""
Utilities for wrapping **stream** (single-pass iterator) objects so that when
iteration completes, the accumulated output is committed via an injected
callback.

Only true streams — objects that implement the **iterator** protocol
(``__next__`` / ``__anext__``) — are wrapped.  Re-iterable collections such
as ``list``, ``tuple``, or ``set`` are **not** streams: their output is
committed eagerly via the callback and the original object is returned
unchanged.

Supported flows:

    1. Netra-wrapped stream (``_netra_stream_wrapper = True``)
        The inner instrumentation wrapper has already accumulated the output
        in ``_netra_output``.  The outer tap simply delegates iteration and
        reads that attribute once the inner wrapper signals exhaustion.

    2. Generic / unknown single-pass stream
        Any iterator whose type Netra does not know about.  Chunks are
        converted to strings via ``str(chunk)`` and concatenated.

    3. Re-iterable collections (``list``, ``tuple``, etc.)
        Output is committed eagerly via the callback.  A warning is logged
        directing the caller to ``Netra.set_root_output()`` instead.

    4. Objects that carry no iterator protocol are returned unchanged with a
        warning log.
"""

import logging
from typing import Any, AsyncIterator, Callable, Generator, Iterator, List, Union

logger = logging.getLogger(__name__)


# Extractors — injected at construction time, kept stateless
def _netra_extractor(wrapper: Union["RootOutputSyncStreamWrapper", "RootOutputAsyncStreamWrapper"]) -> Any:
    """Read accumulated output from the inner Netra wrapper."""
    inner = wrapper._stream
    output = getattr(inner, "_netra_output", None)
    if output is not None:
        return output
    # Nested wrapping: inner is another RootOutput* wrapper with _chunks
    chunks = getattr(inner, "_chunks", None)
    if chunks is not None:
        return "".join(chunks)
    return None


def _generic_extractor(wrapper: Union["RootOutputSyncStreamWrapper", "RootOutputAsyncStreamWrapper"]) -> Any:
    """Return the concatenated stringified chunks."""
    return "".join(wrapper._chunks)


# Sync wrapper
class RootOutputSyncStreamWrapper:
    """Wraps a **single-pass** sync iterator; on exhaustion commits the output
    via the injected ``commit_fn`` callback.

    This wrapper is intended for true streams (objects with ``__next__``) such
    as LLM streaming responses, generators, and Netra-instrumented wrappers.
    It must **not** be used for re-iterable collections (``list``, ``tuple``,
    etc.) — use ``Netra.set_root_output()`` for those.

    Internally delegates to a generator with a ``finally`` block so that
    ``_commit`` fires reliably on full exhaustion, early ``break``, or
    explicit ``.close()`` — not only on ``StopIteration``.
    """

    _netra_stream_wrapper = True

    def __init__(self, stream: Any, commit_fn: Callable[[Any], None], extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._iterator: Iterator[Any] = iter(stream)
        self._commit_fn = commit_fn
        self._extractor = extractor
        self._chunks: List[str] = []
        self._track_chunks: bool = extractor is _generic_extractor
        self._committed = False

    def _iter_gen(self) -> Generator[Any, None, None]:
        try:
            for chunk in self._iterator:
                if self._track_chunks:
                    self._chunks.append(str(chunk))
                yield chunk
        finally:
            self._commit()

    def __iter__(self) -> Generator[Any, None, None]:
        return self._iter_gen()

    def __next__(self) -> Any:
        try:
            chunk = next(self._iterator)
            if self._track_chunks:
                self._chunks.append(str(chunk))
            return chunk
        except StopIteration:
            self._commit()
            raise

    def __enter__(self) -> "RootOutputSyncStreamWrapper":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None:
            self._commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        if not self._committed:
            self._commit()

    def _commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        try:
            self._commit_fn(self._extractor(self))
        except Exception:
            logger.debug("RootOutputSyncWrapper: failed to commit output", exc_info=True)


# Async wrapper
class RootOutputAsyncStreamWrapper:
    """Wraps a **single-pass** async iterator; on exhaustion commits the output
    via the injected ``commit_fn`` callback.

    This wrapper is intended for true async streams (objects with
    ``__anext__``) such as async LLM streaming responses and async generators.
    It must **not** be used for re-iterable async collections — use
    ``Netra.set_root_output()`` for those.

    Uses an internal async generator with ``finally`` so that ``_commit``
    fires on full exhaustion, early ``break`` (via ``aclose()``), or explicit
    close — mirroring the sync wrapper's behaviour.
    """

    _netra_stream_wrapper = True

    def __init__(self, stream: Any, commit_fn: Callable[[Any], None], extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._aiterator: AsyncIterator[Any] = aiter(stream)
        self._commit_fn = commit_fn
        self._extractor = extractor
        self._chunks: List[str] = []
        self._track_chunks: bool = extractor is _generic_extractor
        self._committed = False

    async def _aiter_gen(self) -> Any:
        try:
            async for chunk in self._aiterator:
                if self._track_chunks:
                    self._chunks.append(str(chunk))
                yield chunk
        finally:
            self._commit()

    def __aiter__(self) -> Any:
        return self._aiter_gen()

    async def __anext__(self) -> Any:
        try:
            chunk = await self._aiterator.__anext__()
            if self._track_chunks:
                self._chunks.append(str(chunk))
            return chunk
        except StopAsyncIteration:
            self._commit()
            raise

    async def __aenter__(self) -> "RootOutputAsyncStreamWrapper":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(exc_type, exc_val, exc_tb)
        if exc_type is None:
            self._commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __del__(self) -> None:
        if not self._committed:
            self._commit()

    def _commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        try:
            self._commit_fn(self._extractor(self))
        except Exception:
            logger.debug("RootOutputAsyncWrapper: failed to commit output", exc_info=True)


def _is_stream(obj: Any) -> bool:
    """Return ``True`` if *obj* is a single-pass iterator (has ``__next__`` or ``__anext__``)."""
    return hasattr(obj, "__next__") or hasattr(obj, "__anext__")


def wrap_stream_for_root_output(stream: Any, commit_fn: Callable[[Any], None]) -> Any:
    """Wrap *stream* so the accumulated output is committed via *commit_fn* when
    iteration ends.

    Only **single-pass iterators** (objects with ``__next__`` / ``__anext__``)
    and Netra-instrumented wrappers are wrapped.  Re-iterable collections such
    as ``list`` or ``tuple`` are not streams — their output is committed
    **eagerly** via *commit_fn* and the original object is returned unchanged.

    Detection order:
        1. ``_netra_stream_wrapper`` attribute — always wrapped (Netra-instrumented).
        2. Has ``__next__`` / ``__anext__`` — single-pass stream, wrapped.
        3. Has only ``__iter__`` / ``__aiter__`` (no ``__next__`` / ``__anext__``)
           — re-iterable collection; output committed eagerly, returned unchanged.
        4. Not iterable at all — returned unchanged with a warning.

    Args:
        stream:     The stream or value to wrap.  May be sync or async.
        commit_fn:  Callback invoked with the extracted output when iteration
                    completes (or eagerly for re-iterables).  The caller
                    defines what "commit" means (e.g. serialize and set an
                    attribute on a span).

    Returns:
        A :class:`RootOutputSyncStreamWrapper`, :class:`RootOutputAsyncStreamWrapper`,
        or the original *stream* unchanged.
    """
    is_netra = getattr(stream, "_netra_stream_wrapper", False)

    # Async path
    if hasattr(stream, "__aiter__"):
        if is_netra or _is_stream(stream):
            extractor = _netra_extractor if is_netra else _generic_extractor
            return RootOutputAsyncStreamWrapper(stream, commit_fn, extractor)
        # Re-iterable async object — commit eagerly.
        logger.warning(
            "set_root_output_stream: %s is a re-iterable, not a single-pass stream; "
            "output set eagerly on root span. Prefer Netra.set_root_output() for static values.",
            type(stream).__name__,
        )
        commit_fn(stream)
        return stream

    # Sync path
    if hasattr(stream, "__iter__"):
        if is_netra or _is_stream(stream):
            extractor = _netra_extractor if is_netra else _generic_extractor
            return RootOutputSyncStreamWrapper(stream, commit_fn, extractor)
        # Re-iterable collection (list, tuple, set, …) — commit eagerly.
        logger.warning(
            "set_root_output_stream: %s is a re-iterable, not a single-pass stream; "
            "output set eagerly on root span. Prefer Netra.set_root_output() for static values.",
            type(stream).__name__,
        )
        commit_fn(stream)
        return stream

    # Not iterable at all
    logger.warning(
        "set_root_output_stream: passed object of type %s is not iterable; returning unchanged.",
        type(stream).__name__,
    )
    return stream
