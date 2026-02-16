"""
Cache manager with LRU caching and TTL support.

This module provides thread-safe caching utilities for:
- LRU cache with configurable size
- TTL-based cache expiration
- Thread-safe operations
- Cache statistics and management
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheEntry:
    """Single cache entry with value and TTL."""

    def __init__(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """
        Initialize cache entry.

        Args:
            key: Cache key
            value: Cached value
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = time.time() - self.created_at
        return age > self.ttl_seconds

    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    def __repr__(self) -> str:
        return f"CacheEntry(key={self.key!r}, value_type={type(self.value).__name__}, ttl={self.ttl_seconds})"


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Attributes:
        max_size: Maximum number of entries in cache
        default_ttl: Default TTL in seconds (None = no expiration)
    """

    def __init__(self, max_size: int = 100, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                self._expirations += 1
                logger.debug(f"Cache expired: {key}")
                return None

            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Custom TTL in seconds (uses default if None)
        """
        with self._lock:
            # Use provided TTL or default
            if ttl is None:
                ttl = self.default_ttl

            # Create new entry
            entry = CacheEntry(key, value, ttl)

            # If key exists, replace and move to end
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = entry
            else:
                # Add new entry
                self._cache[key] = entry

                # Evict oldest if at capacity
                if len(self._cache) > self.max_size:
                    oldest_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    logger.debug(f"Cache evicted: {oldest_key} (size={len(self._cache)})")

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def has(self, key: str) -> bool:
        """
        Check if key exists in cache (and is not expired).

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "hit_rate_percent": round(hit_rate, 2),
                "default_ttl": self.default_ttl,
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def get_entries(self) -> List[Tuple[str, Any, float]]:
        """
        Get all cache entries (key, value, age).

        Returns:
            List of tuples (key, value, age_seconds)
        """
        with self._lock:
            return [
                (key, entry.value, entry.age_seconds())
                for key, entry in self._cache.items()
            ]

    def get_oldest_entry(self) -> Optional[Tuple[str, Any, float]]:
        """
        Get oldest cache entry.

        Returns:
            Tuple (key, value, age_seconds) or None if cache is empty
        """
        with self._lock:
            if not self._cache:
                return None

            # First item is oldest (FIFO in OrderedDict)
            key, entry = next(iter(self._cache.items()))
            return (key, entry.value, entry.age_seconds())

    def get_newest_entry(self) -> Optional[Tuple[str, Any, float]]:
        """
        Get newest cache entry.

        Returns:
            Tuple (key, value, age_seconds) or None if cache is empty
        """
        with self._lock:
            if not self._cache:
                return None

            # Last item is newest
            key, entry = next(reversed(self._cache.items()))
            return (key, entry.value, entry.age_seconds())


class CacheManager:
    """
    Centralized cache manager with multiple named caches.

    Provides a simple interface for managing multiple caches
    with different configurations.
    """

    def __init__(self):
        """Initialize cache manager."""
        self._caches: Dict[str, LRUCache] = {}
        self._lock = threading.RLock()

        # Default cache configurations
        self._default_configs: Dict[str, Dict[str, Any]] = {
            "default": {"max_size": 100, "default_ttl": 300},
            "api": {"max_size": 50, "default_ttl": 60},
            "market_data": {"max_size": 200, "default_ttl": 30},
            "indicators": {"max_size": 500, "default_ttl": 600},
            "signals": {"max_size": 100, "default_ttl": 120},
        }

        # Initialize default caches
        for cache_name, config in self._default_configs.items():
            self._caches[cache_name] = LRUCache(**config)

    def get_cache(self, name: str = "default") -> LRUCache:
        """
        Get or create a cache.

        Args:
            name: Cache name

        Returns:
            LRUCache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = LRUCache()
                logger.info(f"Created new cache: {name}")

            return self._caches[name]

    def create_cache(
        self,
        name: str,
        max_size: int = 100,
        default_ttl: Optional[float] = None
    ) -> LRUCache:
        """
        Create a new cache with specific configuration.

        Args:
            name: Cache name
            max_size: Maximum cache size
            default_ttl: Default TTL in seconds

        Returns:
            LRUCache instance
        """
        with self._lock:
            if name in self._caches:
                logger.warning(f"Cache {name} already exists, returning existing")
                return self._caches[name]

            cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
            self._caches[name] = cache
            logger.info(f"Created cache: {name} (max_size={max_size}, ttl={default_ttl})")

            return cache

    def remove_cache(self, name: str) -> bool:
        """
        Remove a cache.

        Args:
            name: Cache name

        Returns:
            True if cache was removed, False if not found
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                logger.info(f"Removed cache: {name}")
                return True
            return False

    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("Cleared all caches")

    def cleanup_all_expired(self) -> int:
        """
        Clean up expired entries in all caches.

        Returns:
            Total number of entries removed
        """
        with self._lock:
            total = 0
            for name, cache in self._caches.items():
                removed = cache.cleanup_expired()
                total += removed
                if removed:
                    logger.debug(f"Cache {name}: removed {removed} expired entries")
            return total

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary mapping cache names to statistics
        """
        with self._lock:
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all caches.

        Returns:
            Dictionary with cache summary
        """
        with self._lock:
            total_size = sum(cache.size() for cache in self._caches.values())
            total_hits = sum(cache._hits for cache in self._caches.values())
            total_misses = sum(cache._misses for cache in self._caches.values())

            total_requests = total_hits + total_misses
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "cache_count": len(self._caches),
                "total_size": total_size,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "overall_hit_rate_percent": round(overall_hit_rate, 2),
                "caches": list(self._caches.keys()),
            }


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None
_global_manager_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        CacheManager instance
    """
    global _global_cache_manager

    with _global_manager_lock:
        if _global_cache_manager is None:
            _global_cache_manager = CacheManager()
            logger.info("Initialized global cache manager")

        return _global_cache_manager


def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_prefix: Optional[str] = None
) -> Callable[..., Callable[..., T]]:
    """
    Decorator to cache function results.

    Args:
        cache_name: Name of cache to use
        ttl: TTL for cached results (uses cache default if None)
        key_prefix: Custom key prefix (uses function name if None)

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)

            # Generate cache key
            prefix = key_prefix or func.__name__
            # Convert args and kwargs to a hashable key
            key_parts = [prefix]

            # Add args to key (skip self for methods)
            if args and hasattr(args[0], '__class__'):
                # Skip 'self' for methods
                key_parts.extend(str(arg) for arg in args[1:])
            else:
                key_parts.extend(str(arg) for arg in args)

            # Add sorted kwargs to key
            for k in sorted(kwargs.keys()):
                key_parts.append(f"{k}={kwargs[k]}")

            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_value

            # Compute value
            logger.debug(f"Cache miss for {prefix}")
            result = func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result, ttl=ttl)

            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return wrapper

    return decorator
