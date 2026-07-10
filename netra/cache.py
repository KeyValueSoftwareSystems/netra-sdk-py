import threading
import time
from typing import Dict, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """In-memory TTL cache for SDK read API responses."""

    def __init__(self, default_ttl: int = 60) -> None:
        self._default_ttl = default_ttl
        self._store: Dict[str, Tuple[T, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        ttl_seconds = self._default_ttl if ttl is None else ttl
        if ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._store[key] = (value, expires_at)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
