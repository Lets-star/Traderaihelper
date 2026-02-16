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

from logging_config import get_structured_logger

# Optional metrics import
try:
    from metrics import update_bus_messages, update_bus_dropped, update_bus_queue_size
    from metrics.collectors import get_update_bus_collector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = get_structured_logger(__name__)


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
            logger.warning("Invalid update payload", payload_type=type(update).__name__)
            return False
        
        if "type" not in update:
            logger.warning("Update payload missing 'type' field", update=update)
            return False
        
        msg_type = update.get("type", "unknown")
        
        try:
            self._queue.put_nowait(update)
            
            # Record metrics
            if METRICS_AVAILABLE:
                update_bus_messages.labels(message_type=msg_type).inc()
                update_bus_queue_size.set(self._queue.qsize())
                
                collector = get_update_bus_collector()
                collector.record_publish(msg_type)
            
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
                if self._dropped_count % 10 == 1:  # Log every 10th drop
                    logger.warning(
                        "UpdateBus queue full, dropping updates",
                        dropped_count=self._dropped_count,
                        message_type=msg_type
                    )
            
            # Record dropped metric
            if METRICS_AVAILABLE:
                update_bus_dropped.labels(message_type=msg_type, reason="queue_full").inc()
                
                collector = get_update_bus_collector()
                collector.record_dropped(msg_type, "queue_full")
            
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
        
        # Update queue size metric after draining
        if METRICS_AVAILABLE:
            update_bus_queue_size.set(self._queue.qsize())
        
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
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        # Record dropped metric for cleared messages
        if METRICS_AVAILABLE and cleared > 0:
            update_bus_dropped.labels(message_type="all", reason="clear").inc(cleared)
            update_bus_queue_size.set(0)
    
    def size(self) -> int:
        """Get the current queue size."""
        size = self._queue.qsize()
        
        # Update metric
        if METRICS_AVAILABLE:
            update_bus_queue_size.set(size)
        
        return size
