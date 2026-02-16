"""
Metric collector classes for aggregating and exporting metrics.

Provides high-level collectors for different subsystems that handle
metric recording in a thread-safe manner.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from metrics import (
    websocket_connections,
    websocket_reconnections,
    websocket_latency,
    websocket_messages,
    websocket_active_connections,
    websocket_errors,
    signal_executions,
    signal_execution_latency,
    signal_validation_errors,
    active_signals,
    api_requests,
    api_latency,
    api_rate_limits,
    api_errors,
    cache_hits,
    cache_misses,
    cache_size,
    cache_evictions,
    worker_starts,
    worker_stops,
    worker_errors,
    worker_processing_time,
    update_bus_messages,
    update_bus_dropped,
    update_bus_queue_size,
)


@dataclass
class SignalExecutionRecord:
    """Record of a signal execution."""
    signal_id: str
    symbol: str
    status: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    error_msg: str = ""


class SignalExecutionCollector:
    """Collector for signal execution metrics.
    
    Maintains a rolling window of recent executions for analysis.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[SignalExecutionRecord] = []
        self._lock = threading.RLock()
        self._stats_cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0
    
    def record(
        self,
        signal_id: str,
        symbol: str,
        status: str,
        latency_ms: float,
        error_msg: str = ""
    ):
        """Record a signal execution."""
        record = SignalExecutionRecord(
            signal_id=signal_id,
            symbol=symbol,
            status=status,
            latency_ms=latency_ms,
            error_msg=error_msg
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
        
        # Update Prometheus metrics
        success = status == "filled" or status == "success"
        error_type = ""
        if not success and error_msg:
            if "validation" in error_msg.lower():
                error_type = "validation"
            elif "network" in error_msg.lower():
                error_type = "network"
            elif "timeout" in error_msg.lower():
                error_type = "timeout"
            else:
                error_type = "other"
        
        signal_executions.labels(
            symbol=symbol,
            status="success" if success else "error",
            error_type=error_type or "none"
        ).inc()
        signal_execution_latency.labels(
            symbol=symbol,
            status="success" if success else "error"
        ).observe(latency_ms / 1000)
    
    def record_validation_error(self, symbol: str, field: str):
        """Record a validation error."""
        signal_validation_errors.labels(symbol=symbol, error_field=field).inc()
    
    def get_stats(self, window_seconds: float = 300) -> Dict[str, Any]:
        """Get execution statistics for the recent window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Statistics dictionary
        """
        # Simple caching to avoid recalculating frequently
        now = time.time()
        if now - self._cache_timestamp < 1.0 and self._stats_cache:
            return self._stats_cache
        
        cutoff = now - window_seconds
        
        with self._lock:
            recent = [r for r in self._history if r.timestamp >= cutoff]
        
        if not recent:
            return {
                "total": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }
        
        total = len(recent)
        successful = sum(1 for r in recent if r.status in ("filled", "success"))
        latencies = [r.latency_ms for r in recent]
        
        latencies_sorted = sorted(latencies)
        
        stats = {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": latencies_sorted[len(latencies_sorted) // 2],
            "p95_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)],
            "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)],
        }
        
        self._stats_cache = stats
        self._cache_timestamp = now
        return stats


@dataclass
class WebSocketConnectionRecord:
    """Record of a WebSocket connection event."""
    symbol: str
    interval: str
    event: str  # connect, disconnect, reconnect, error
    timestamp: float = field(default_factory=time.time)
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None


class WebSocketMetricsCollector:
    """Collector for WebSocket connection metrics."""
    
    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self._history: List[WebSocketConnectionRecord] = []
        self._lock = threading.RLock()
        self._active_connections: Dict[str, int] = defaultdict(int)
    
    def record_connect(self, symbol: str, interval: str, success: bool, latency_ms: float = 0):
        """Record a connection attempt."""
        status = "success" if success else "failed"
        websocket_connections.labels(symbol=symbol, interval=interval, status=status).inc()
        
        if latency_ms > 0:
            websocket_latency.labels(symbol=symbol, interval=interval).observe(latency_ms / 1000)
        
        record = WebSocketConnectionRecord(
            symbol=symbol,
            interval=interval,
            event="connect" if success else "connect_failed",
            latency_ms=latency_ms
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            
            if success:
                key = f"{symbol}_{interval}"
                self._active_connections[key] = 1
                websocket_active_connections.labels(symbol=symbol, interval=interval).set(1)
    
    def record_disconnect(self, symbol: str, interval: str, reason: str = ""):
        """Record a disconnection."""
        record = WebSocketConnectionRecord(
            symbol=symbol,
            interval=interval,
            event=f"disconnect_{reason}" if reason else "disconnect"
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            
            key = f"{symbol}_{interval}"
            self._active_connections[key] = 0
            websocket_active_connections.labels(symbol=symbol, interval=interval).set(0)
    
    def record_reconnect(self, symbol: str, interval: str, attempt: int):
        """Record a reconnection attempt."""
        websocket_reconnections.labels(symbol=symbol, interval=interval).inc()
        
        record = WebSocketConnectionRecord(
            symbol=symbol,
            interval=interval,
            event=f"reconnect_attempt_{attempt}"
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
    
    def record_error(self, symbol: str, interval: str, error_type: str):
        """Record a WebSocket error."""
        websocket_errors.labels(symbol=symbol, interval=interval, error_type=error_type).inc()
        
        record = WebSocketConnectionRecord(
            symbol=symbol,
            interval=interval,
            event="error",
            error_type=error_type
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
    
    def record_message(self, symbol: str, interval: str, msg_type: str):
        """Record a message receipt."""
        websocket_messages.labels(symbol=symbol, interval=interval, type=msg_type).inc()
    
    def get_active_connections(self) -> int:
        """Get total number of active connections."""
        with self._lock:
            return sum(self._active_connections.values())
    
    def get_connection_stats(self, window_seconds: float = 300) -> Dict[str, Any]:
        """Get connection statistics."""
        cutoff = time.time() - window_seconds
        
        with self._lock:
            recent = [r for r in self._history if r.timestamp >= cutoff]
        
        if not recent:
            return {
                "total_events": 0,
                "connects": 0,
                "disconnects": 0,
                "reconnects": 0,
                "errors": 0,
                "active_connections": 0,
            }
        
        connects = sum(1 for r in recent if r.event == "connect")
        disconnects = sum(1 for r in recent if r.event.startswith("disconnect"))
        reconnects = sum(1 for r in recent if r.event.startswith("reconnect"))
        errors = sum(1 for r in recent if r.event == "error")
        
        return {
            "total_events": len(recent),
            "connects": connects,
            "disconnects": disconnects,
            "reconnects": reconnects,
            "errors": errors,
            "active_connections": self.get_active_connections(),
        }


@dataclass
class APIRequestRecord:
    """Record of an API request."""
    endpoint: str
    method: str
    status: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    error_code: str = ""


class APIMetricsCollector:
    """Collector for API request metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[APIRequestRecord] = []
        self._lock = threading.RLock()
        self._active_requests: Dict[str, int] = defaultdict(int)
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status: str,
        latency_ms: float,
        error_code: str = ""
    ):
        """Record an API request."""
        record = APIRequestRecord(
            endpoint=endpoint,
            method=method,
            status=status,
            latency_ms=latency_ms,
            error_code=error_code
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
        
        # Update Prometheus metrics
        api_requests.labels(endpoint=endpoint, method=method, status=status).inc()
        api_latency.labels(endpoint=endpoint, method=method, status=status).observe(latency_ms / 1000)
        
        if error_code:
            api_errors.labels(endpoint=endpoint, method=method, error_code=error_code).inc()
    
    def record_rate_limit(self, endpoint: str, method: str):
        """Record a rate limit hit."""
        api_rate_limits.labels(endpoint=endpoint, method=method).inc()
        
        # Also record in history for stats tracking
        record = APIRequestRecord(
            endpoint=endpoint,
            method=method,
            status="rate_limited",
            latency_ms=0,
            error_code="RATE_LIMIT"
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
    
    def get_stats(self, window_seconds: float = 300) -> Dict[str, Any]:
        """Get API statistics."""
        cutoff = time.time() - window_seconds
        
        with self._lock:
            recent = [r for r in self._history if r.timestamp >= cutoff]
        
        if not recent:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "rate_limit_hits": 0,
            }
        
        total = len(recent)
        successful = sum(1 for r in recent if r.status == "success")
        rate_limit_hits = sum(1 for r in recent if r.status == "rate_limited")
        latencies = [r.latency_ms for r in recent]
        
        return {
            "total_requests": total,
            "successful": successful,
            "failed": total - successful,
            "rate_limit_hits": rate_limit_hits,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
        }


class CacheMetricsCollector:
    """Collector for cache metrics."""
    
    def __init__(self):
        self._sizes: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def record_hit(self, cache_name: str, key_type: str = "default"):
        """Record a cache hit."""
        cache_hits.labels(cache_name=cache_name, key_type=key_type).inc()
    
    def record_miss(self, cache_name: str, key_type: str = "default"):
        """Record a cache miss."""
        cache_misses.labels(cache_name=cache_name, key_type=key_type).inc()
    
    def update_size(self, cache_name: str, size: int):
        """Update cache size gauge."""
        with self._lock:
            self._sizes[cache_name] = size
            cache_size.labels(cache_name=cache_name).set(size)
    
    def record_eviction(self, cache_name: str, reason: str):
        """Record a cache eviction."""
        cache_evictions.labels(cache_name=cache_name, reason=reason).inc()
    
    def get_hit_ratio(self, cache_name: str) -> float:
        """Calculate hit ratio for a cache."""
        # Note: This is a simplified calculation. In production, you'd want
        # to track hits/misses over a specific time window.
        return 0.0  # Placeholder


class WorkerMetricsCollector:
    """Collector for worker lifecycle metrics."""
    
    def record_start(self, worker_type: str, symbol: str, timeframe: str):
        """Record worker start."""
        worker_starts.labels(
            worker_type=worker_type,
            symbol=symbol,
            timeframe=timeframe
        ).inc()
    
    def record_stop(self, worker_type: str, symbol: str, timeframe: str, reason: str = "normal"):
        """Record worker stop."""
        worker_stops.labels(
            worker_type=worker_type,
            symbol=symbol,
            timeframe=timeframe,
            reason=reason
        ).inc()
    
    def record_error(
        self,
        worker_type: str,
        symbol: str,
        timeframe: str,
        error_type: str
    ):
        """Record worker error."""
        worker_errors.labels(
            worker_type=worker_type,
            symbol=symbol,
            timeframe=timeframe,
            error_type=error_type
        ).inc()
    
    def record_processing_time(
        self,
        worker_type: str,
        symbol: str,
        timeframe: str,
        duration_seconds: float
    ):
        """Record worker processing time."""
        worker_processing_time.labels(
            worker_type=worker_type,
            symbol=symbol,
            timeframe=timeframe
        ).observe(duration_seconds)


class UpdateBusMetricsCollector:
    """Collector for UpdateBus metrics."""
    
    def record_publish(self, message_type: str):
        """Record a message publication."""
        update_bus_messages.labels(message_type=message_type).inc()
    
    def record_dropped(self, message_type: str, reason: str):
        """Record a dropped message."""
        update_bus_dropped.labels(message_type=message_type, reason=reason).inc()
    
    def update_queue_size(self, size: int):
        """Update queue size gauge."""
        update_bus_queue_size.set(size)


# Global collectors for convenience
_signal_collector: Optional[SignalExecutionCollector] = None
_websocket_collector: Optional[WebSocketMetricsCollector] = None
_api_collector: Optional[APIMetricsCollector] = None
_cache_collector: Optional[CacheMetricsCollector] = None
_worker_collector: Optional[WorkerMetricsCollector] = None
_update_bus_collector: Optional[UpdateBusMetricsCollector] = None


def get_signal_collector() -> SignalExecutionCollector:
    """Get the global signal execution collector."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = SignalExecutionCollector()
    return _signal_collector


def get_websocket_collector() -> WebSocketMetricsCollector:
    """Get the global WebSocket metrics collector."""
    global _websocket_collector
    if _websocket_collector is None:
        _websocket_collector = WebSocketMetricsCollector()
    return _websocket_collector


def get_api_collector() -> APIMetricsCollector:
    """Get the global API metrics collector."""
    global _api_collector
    if _api_collector is None:
        _api_collector = APIMetricsCollector()
    return _api_collector


def get_cache_collector() -> CacheMetricsCollector:
    """Get the global cache metrics collector."""
    global _cache_collector
    if _cache_collector is None:
        _cache_collector = CacheMetricsCollector()
    return _cache_collector


def get_worker_collector() -> WorkerMetricsCollector:
    """Get the global worker metrics collector."""
    global _worker_collector
    if _worker_collector is None:
        _worker_collector = WorkerMetricsCollector()
    return _worker_collector


def get_update_bus_collector() -> UpdateBusMetricsCollector:
    """Get the global UpdateBus metrics collector."""
    global _update_bus_collector
    if _update_bus_collector is None:
        _update_bus_collector = UpdateBusMetricsCollector()
    return _update_bus_collector
