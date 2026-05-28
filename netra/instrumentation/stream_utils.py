"""
Utilities for wrapping stream objects so that when iteration completes, the
accumulated output is automatically set on the root span of the current trace.

Two flows are supported:

    Flow 1 — Netra-wrapped stream (``_netra_stream_wrapper = True``)
        The inner instrumentation wrapper has already accumulated the output
        in ``_netra_output``.  The outer tap simply delegates iteration and
        reads that attribute once the inner wrapper signals exhaustion.

    Flow 3 — Generic / unknown stream
        Any iterable whose type Netra does not know about.  Chunks are
        converted to strings via ``str(chunk)`` and concatenated.

Objects that carry no iterator protocol are returned unchanged with a
warning log (Flow 4).
"""

import logging
from typing import Any, Callable, List, Union

from opentelemetry.trace import Span

from netra.session_manager import NETRA_USER_OUTPUT
from netra.utils import serialize_value

logger = logging.getLogger(__name__)


def _set_output_on_root(root_span: Span, output: Any) -> None:
    """Write serialized *output* to *root_span* as ``NETRA_USER_OUTPUT``."""
    try:
        serialized = serialize_value(output)
        if serialized:
            root_span.set_attribute(NETRA_USER_OUTPUT, serialized)
    except Exception:
        logger.warning("root_output_stream: failed to set output on root span", exc_info=True)


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
    """Wraps a sync iterable; on exhaustion sets the output on the root span."""

    _netra_stream_wrapper = True

    def __init__(self, stream: Any, root_span: Span, extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._root_span = root_span
        self._extractor = extractor
        self._chunks: List[str] = []
        self._track_chunks: bool = extractor is _generic_extractor
        self._committed = False

    def __iter__(self) -> "RootOutputSyncStreamWrapper":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
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
            _set_output_on_root(self._root_span, self._extractor(self))
        except Exception:
            logger.debug("RootOutputSyncWrapper: failed to commit output to root span", exc_info=True)


# Async wrapper
class RootOutputAsyncStreamWrapper:
    """Wraps an async iterable; on exhaustion sets the output on the root span."""

    _netra_stream_wrapper = True

    def __init__(self, stream: Any, root_span: Span, extractor: Callable[[Any], Any]) -> None:
        self._stream = stream
        self._root_span = root_span
        self._extractor = extractor
        self._chunks: List[str] = []
        self._track_chunks: bool = extractor is _generic_extractor
        self._committed = False

    def __aiter__(self) -> "RootOutputAsyncStreamWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
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
            _set_output_on_root(self._root_span, self._extractor(self))
        except Exception:
            logger.debug("RootOutputAsyncWrapper: failed to commit output to root span", exc_info=True)


def wrap_stream_for_root_output(stream: Any, root_span: Span) -> Any:
    """Wrap *stream* so the accumulated output is set on *root_span* when iteration ends.

    Detection order:
        1. ``_netra_stream_wrapper`` attribute present (Netra-wrapped)
        2. Has ``__aiter__`` or ``__iter__`` (generic)
        3. Not iterable (return unchanged)

    Args:
        stream:    The stream to wrap.  May be sync or async.
        root_span: The root OTel span that will receive the ``NETRA_USER_OUTPUT`` attribute.

    Returns:
        A :class:`RootOutputSyncWrapper`, :class:`RootOutputAsyncWrapper`, or the
        original *stream* unchanged if it is not iterable.
    """
    is_netra = getattr(stream, "_netra_stream_wrapper", False)
    extractor: Callable[[Union["RootOutputSyncStreamWrapper", "RootOutputAsyncStreamWrapper"]], Any] = (
        _netra_extractor if is_netra else _generic_extractor
    )

    if hasattr(stream, "__aiter__"):
        return RootOutputAsyncStreamWrapper(stream, root_span, extractor)

    if hasattr(stream, "__iter__"):
        return RootOutputSyncStreamWrapper(stream, root_span, extractor)

    logger.warning(
        "set_root_output_stream: passed object of type %s is not iterable; returning unchanged",
        type(stream).__name__,
    )
    return stream
