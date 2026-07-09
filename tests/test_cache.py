from unittest.mock import patch

import pytest

from netra.cache import TTLCache


class TestTTLCache:
    def test_get_returns_none_for_missing_key(self) -> None:
        cache = TTLCache[str]()
        assert cache.get("missing") is None

    def test_set_and_get_returns_stored_value_before_ttl_expires(self) -> None:
        cache = TTLCache[str](default_ttl=60)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_returns_none_after_ttl_expires(self) -> None:
        with patch("netra.cache.time.monotonic", side_effect=[0.0, 1.1]):
            cache = TTLCache[str](default_ttl=1)
            cache.set("key", "value")
            assert cache.get("key") is None

    def test_per_entry_ttl_override_expires_independently_of_default(self) -> None:
        with patch("netra.cache.time.monotonic", side_effect=[0.0, 0.0, 1.1, 1.1]):
            cache = TTLCache[str](default_ttl=60)
            cache.set("short", "a", ttl=1)
            cache.set("long", "b", ttl=60)
            assert cache.get("short") is None
            assert cache.get("long") == "b"

    def test_clear_removes_all_entries(self) -> None:
        cache = TTLCache[str]()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_invalidate_removes_single_entry(self) -> None:
        cache = TTLCache[str]()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") == "2"

    def test_thread_safe_concurrent_access(self) -> None:
        cache = TTLCache[int](default_ttl=60)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                cache.set(f"key-{i}", i)
                assert cache.get(f"key-{i}") == i
            except Exception as exc:
                errors.append(exc)

        import threading

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
