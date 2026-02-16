"""
Generic types for reusable containers and data structures.

Generic types enable type-safe containers that work with any type,
providing better type inference and IDE support.
"""

from __future__ import annotations

import queue
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

# Type variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
E = TypeVar("E", bound=Exception)


class UpdateBus(Generic[T]):
    """
    Generic thread-safe update bus for type-safe message passing.
    
    Type parameter T specifies the type of messages in the bus.
    
    Example:
        bus: UpdateBus[SignalPayload] = UpdateBus()
        bus.publish({"signal_id": "123", ...})  # Type checked
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize the update bus.
        
        Args:
            max_size: Maximum queue size (prevents unbounded growth)
        """
        self._queue: queue.Queue[T] = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()
        self._dropped_count = 0

    def publish(self, update: T) -> bool:
        """
        Publish an update to the bus (called from worker threads).
        
        Args:
            update: Update payload of type T
            
        Returns:
            True if published successfully, False if queue is full
        """
        try:
            self._queue.put_nowait(update)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
            return False

    def drain(self, max_updates: Optional[int] = None) -> List[T]:
        """
        Drain updates from the bus (called from main thread).
        
        Args:
            max_updates: Maximum number of updates to drain (None = drain all)
            
        Returns:
            List of update payloads of type T
        """
        updates: List[T] = []
        count = 0

        while True:
            if max_updates is not None and count >= max_updates:
                break

            try:
                update = self._queue.get_nowait()
                updates.append(update)
                count += 1
            except queue.Empty:
                break

        return updates

    def has_updates(self) -> bool:
        """Check if there are pending updates."""
        return not self._queue.empty()

    def get_dropped_count(self) -> int:
        """Get the number of dropped updates."""
        with self._lock:
            return self._dropped_count

    def reset_dropped_count(self) -> None:
        """Reset the dropped update counter."""
        with self._lock:
            self._dropped_count = 0

    def clear(self) -> None:
        """Clear all pending updates."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def size(self) -> int:
        """Get the current queue size."""
        return self._queue.qsize()


class Result(Generic[T, E]):
    """
    Generic result type for operations that can succeed or fail.
    
    Similar to Rust's Result type, providing a type-safe way to
    handle success and error cases.
    
    Type parameters:
    - T: Success value type
    - E: Error type (default Exception)
    
    Example:
        def fetch_data() -> Result[KlineData, RequestException]:
            try:
                data = api.fetch()
                return Result.ok(data)
            except RequestException as e:
                return Result.err(e)
    """

    def __init__(
        self,
        is_ok: bool,
        value: Optional[T] = None,
        error: Optional[E] = None,
    ) -> None:
        self._is_ok = is_ok
        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> Result[T, Any]:
        """Create a successful result."""
        return cls(is_ok=True, value=value, error=None)

    @classmethod
    def err(cls, error: E) -> Result[Any, E]:
        """Create an error result."""
        return cls(is_ok=False, value=None, error=error)

    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._is_ok

    @property
    def is_err(self) -> bool:
        """Check if result is an error."""
        return not self._is_ok

    def unwrap(self) -> T:
        """
        Get the success value, raising if error.
        
        Raises:
            RuntimeError: If result is an error
        """
        if self._is_ok and self._value is not None:
            return self._value
        raise RuntimeError(f"Cannot unwrap error result: {self._error}")

    def unwrap_or(self, default: T) -> T:
        """Get the success value or return default."""
        if self._is_ok and self._value is not None:
            return self._value
        return default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Get the success value or compute from error."""
        if self._is_ok and self._value is not None:
            return self._value
        if self._error is not None:
            return f(self._error)
        raise RuntimeError("Both value and error are None")

    def map(self, f: Callable[[T], Any]) -> Result[Any, E]:
        """Transform success value with function."""
        if self._is_ok and self._value is not None:
            return Result.ok(f(self._value))
        return Result(is_ok=False, value=None, error=self._error)

    def map_err(self, f: Callable[[E], Any]) -> Result[T, Any]:
        """Transform error value with function."""
        if not self._is_ok and self._error is not None:
            return Result.err(f(self._error))
        return self

    def expect(self, msg: str) -> T:
        """
        Get success value with custom error message.
        
        Raises:
            RuntimeError: If result is an error
        """
        if self._is_ok and self._value is not None:
            return self._value
        raise RuntimeError(f"{msg}: {self._error}")

    @property
    def error(self) -> Optional[E]:
        """Get the error value (None if success)."""
        return self._error


class DataStore(Generic[K, V]):
    """
    Generic thread-safe key-value data store.
    
    Type parameters:
    - K: Key type
    - V: Value type
    
    Example:
        store: DataStore[str, KlineData] = DataStore()
        store.set("BTCUSDT:1h", kline_data)
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        """
        Initialize the data store.
        
        Args:
            max_size: Maximum number of entries (None = unlimited)
        """
        self._data: Dict[K, V] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._access_count: Dict[K, int] = {}

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value by key.
        
        Args:
            key: Key to look up
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        with self._lock:
            if key in self._data:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._data[key]
            return default

    def set(self, key: K, value: V) -> None:
        """
        Set value by key.
        
        Args:
            key: Key to set
            value: Value to store
        """
        with self._lock:
            if self._max_size is not None and len(self._data) >= self._max_size:
                # Evict least accessed entry
                if self._data:
                    lru_key = min(
                        self._access_count.keys(),
                        key=lambda k: self._access_count.get(k, 0)
                    )
                    del self._data[lru_key]
                    del self._access_count[lru_key]

            self._data[key] = value
            self._access_count[key] = self._access_count.get(key, 0) + 1

    def delete(self, key: K) -> bool:
        """
        Delete value by key.
        
        Args:
            key: Key to delete
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._access_count:
                    del self._access_count[key]
                return True
            return False

    def contains(self, key: K) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._data

    def keys(self) -> List[K]:
        """Get all keys."""
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[V]:
        """Get all values."""
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[tuple[K, V]]:
        """Get all key-value pairs."""
        with self._lock:
            return list(self._data.items())

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._data.clear()
            self._access_count.clear()

    def size(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._data)

    def get_or_compute(
        self,
        key: K,
        compute: Callable[[], V],
    ) -> V:
        """
        Get value or compute and store if not exists.
        
        Args:
            key: Key to look up
            compute: Function to compute value if not exists
            
        Returns:
            Existing or computed value
        """
        with self._lock:
            if key in self._data:
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._data[key]

        # Compute outside lock
        value = compute()

        with self._lock:
            self._data[key] = value
            self._access_count[key] = 1
            return value


class PaginatedList(Generic[T]):
    """
    Generic paginated list for handling large datasets.
    
    Type parameter T specifies the item type.
    
    Example:
        items: PaginatedList[KlineData] = PaginatedList(klines, page_size=100)
        for page in items.pages():
            process(page)
    """

    def __init__(
        self,
        items: List[T],
        page_size: int = 100,
        total_count: Optional[int] = None,
    ) -> None:
        """
        Initialize paginated list.
        
        Args:
            items: List of items
            page_size: Number of items per page
            total_count: Total count (if different from len(items))
        """
        self._items = items
        self._page_size = page_size
        self._total_count = total_count or len(items)

    @property
    def total_count(self) -> int:
        """Get total item count."""
        return self._total_count

    @property
    def page_size(self) -> int:
        """Get page size."""
        return self._page_size

    @property
    def page_count(self) -> int:
        """Get number of pages."""
        return (self._total_count + self._page_size - 1) // self._page_size

    def get_page(self, page_number: int) -> List[T]:
        """
        Get specific page (0-indexed).
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            List of items for that page
        """
        start = page_number * self._page_size
        end = start + self._page_size
        return self._items[start:end]

    def pages(self):
        """Iterate over all pages."""
        for i in range(self.page_count):
            yield self.get_page(i)

    def __iter__(self):
        """Iterate over all items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Get total item count."""
        return self._total_count

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        """Get item by index or slice."""
        return self._items[index]


class LazyValue(Generic[T]):
    """
    Generic lazy value that computes on first access.
    
    Type parameter T specifies the value type.
    
    Example:
        expensive_data: LazyValue[DataFrame] = LazyValue(lambda: load_large_dataset())
        # Computation happens only on first access
        df = expensive_data.value
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        """
        Initialize lazy value.
        
        Args:
            factory: Function to compute the value
        """
        self._factory = factory
        self._value: Optional[T] = None
        self._computed = False
        self._lock = threading.RLock()

    @property
    def value(self) -> T:
        """Get the value (computes on first access)."""
        if not self._computed:
            with self._lock:
                if not self._computed:
                    self._value = self._factory()
                    self._computed = True
        # Type assertion: after computation, _value is not None
        assert self._value is not None
        return self._value

    @property
    def is_computed(self) -> bool:
        """Check if value has been computed."""
        return self._computed

    def reset(self) -> None:
        """Reset to force re-computation on next access."""
        with self._lock:
            self._value = None
            self._computed = False
