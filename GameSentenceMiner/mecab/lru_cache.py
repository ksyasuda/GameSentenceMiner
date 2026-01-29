# Copyright: Ajatt-Tools and contributors; https://github.com/Ajatt-Tools
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache for results of calls to mecab.translate().
    Uses RLock to allow recursive locking from the same thread.
    """

    _cache: OrderedDict[K, V]
    _capacity: int
    _lock: threading.RLock

    def __init__(self, capacity: int = 0) -> None:
        self._capacity = capacity
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, key: K) -> V:
        with self._lock:
            value = self._cache[key]
            self._cache.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._clear_old_items()

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def set_capacity(self, capacity: int) -> None:
        with self._lock:
            self._capacity = capacity
            self._clear_old_items()

    def _clear_old_items(self) -> None:
        # Note: Called while holding the lock
        if self._capacity > 0:
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def setdefault(self, key: K, value: V) -> V:
        with self._lock:
            value = self._cache.setdefault(key, value)
            self._cache.move_to_end(key)
            self._clear_old_items()
            return value

    def get(self, key: K, default: V = None) -> V:
        """Get a value with an optional default, without raising KeyError."""
        with self._lock:
            if key in self._cache:
                value = self._cache[key]
                self._cache.move_to_end(key)
                return value
            return default

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
