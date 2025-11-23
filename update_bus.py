"""
Thread-safe UpdateBus for worker-to-main-thread communication.

This pattern ensures no Streamlit API calls from worker threads.
Workers publish Dict payloads with a "type" field; main thread drains and applies updates.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UpdateBus:
    """Thread-safe message bus for worker-to-main-thread updates."""
    
    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize the update bus.
        
        Args:
            max_size: Maximum queue size (prevents unbounded growth)
        """
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()
        self._dropped_count = 0
    
    def publish(self, update: Dict[str, Any]) -> bool:
        """
        Publish an update to the bus (called from worker threads).
        
        Args:
            update: Update payload (must contain "type" field)
            
        Returns:
            True if published successfully, False if queue is full
        """
        if not isinstance(update, dict):
            logger.warning(f"Invalid update payload: {type(update)}")
            return False
        
        if "type" not in update:
            logger.warning(f"Update payload missing 'type' field: {update}")
            return False
        
        try:
            self._queue.put_nowait(update)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
                if self._dropped_count % 10 == 1:  # Log every 10th drop
                    logger.warning(
                        f"UpdateBus queue full, dropped {self._dropped_count} updates "
                        f"(type={update.get('type')})"
                    )
            return False
    
    def drain(self, max_updates: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Drain updates from the bus (called from main thread).
        
        Args:
            max_updates: Maximum number of updates to drain (None = drain all)
            
        Returns:
            List of update payloads
        """
        updates: List[Dict[str, Any]] = []
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
