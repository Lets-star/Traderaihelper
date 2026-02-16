"""
Prometheus metrics for trading system observability.

Provides metrics for:
- WebSocket connections, reconnections, latency, messages
- Signal execution: executed count, latency, validation errors
- API calls: requests, latency, rate limits
- Cache: hits, misses, size
- Trading: positions, trades, PnL

Usage:
    from metrics import websocket_latency, signal_execution_latency
    
    # Record metrics
    websocket_latency.observe(0.05)  # 50ms
    signal_execution_latency.observe(0.5)  # 500ms
"""

from __future__ import annotations

import os
from typing import Optional

# Try to import prometheus_client, provide stubs if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
    # Stub classes for when prometheus_client is not installed
    class _StubMetric:
        """Stub metric that does nothing when prometheus_client is unavailable."""
        
        def __init__(self, *args, **kwargs):
            pass
        
        def inc(self, amount: float = 1):
            pass
        
        def dec(self, amount: float = 1):
            pass
        
        def set(self, value: float):
            pass
        
        def observe(self, value: float):
            pass
        
        def time(self):
            class _Timer:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return _Timer()
        
        def labels(self, *args, **kwargs):
            return self
    
    Counter = _StubMetric
    Histogram = _StubMetric
    Gauge = _StubMetric
    Info = _StubMetric
    CollectorRegistry = lambda: None
    
    def generate_latest(*args, **kwargs):
        return b"# Prometheus metrics disabled (prometheus_client not installed)\n"


# Global registry
REGISTRY = CollectorRegistry() if PROMETHEUS_AVAILABLE else None


# ==================== WebSocket Metrics ====================

websocket_connections = Counter(
    "websocket_connections_total",
    "Total WebSocket connection attempts",
    ["symbol", "interval", "status"],
    registry=REGISTRY
)

websocket_reconnections = Counter(
    "websocket_reconnections_total",
    "Total WebSocket reconnection attempts",
    ["symbol", "interval"],
    registry=REGISTRY
)

websocket_latency = Histogram(
    "websocket_latency_seconds",
    "WebSocket connection establishment latency",
    ["symbol", "interval"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

websocket_messages = Counter(
    "websocket_messages_total",
    "Total WebSocket messages received",
    ["symbol", "interval", "type"],
    registry=REGISTRY
)

websocket_active_connections = Gauge(
    "websocket_active_connections",
    "Number of currently active WebSocket connections",
    ["symbol", "interval"],
    registry=REGISTRY
)

websocket_errors = Counter(
    "websocket_errors_total",
    "Total WebSocket errors",
    ["symbol", "interval", "error_type"],
    registry=REGISTRY
)


# ==================== Signal Execution Metrics ====================

signal_executions = Counter(
    "signal_executions_total",
    "Total signal executions",
    ["symbol", "status", "error_type"],
    registry=REGISTRY
)

signal_execution_latency = Histogram(
    "signal_execution_latency_seconds",
    "Signal execution latency (from submission to completion)",
    ["symbol", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

signal_validation_errors = Counter(
    "signal_validation_errors_total",
    "Total signal validation errors",
    ["symbol", "error_field"],
    registry=REGISTRY
)

active_signals = Gauge(
    "active_signals",
    "Number of signals currently being processed",
    ["symbol"],
    registry=REGISTRY
)


# ==================== API Metrics ====================

api_requests = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY
)

api_latency = Histogram(
    "api_latency_seconds",
    "API request latency",
    ["endpoint", "method", "status"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY
)

api_rate_limits = Counter(
    "api_rate_limits_total",
    "Total API rate limit hits",
    ["endpoint", "method"],
    registry=REGISTRY
)

api_errors = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint", "method", "error_code"],
    registry=REGISTRY
)

api_active_requests = Gauge(
    "api_active_requests",
    "Number of currently active API requests",
    ["endpoint", "method"],
    registry=REGISTRY
)


# ==================== Cache Metrics ====================

cache_hits = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["cache_name", "key_type"],
    registry=REGISTRY
)

cache_misses = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["cache_name", "key_type"],
    registry=REGISTRY
)

cache_size = Gauge(
    "cache_size",
    "Current cache size (number of entries)",
    ["cache_name"],
    registry=REGISTRY
)

cache_evictions = Counter(
    "cache_evictions_total",
    "Total cache evictions",
    ["cache_name", "reason"],
    registry=REGISTRY
)

cache_latency = Histogram(
    "cache_latency_seconds",
    "Cache operation latency",
    ["cache_name", "operation"],
    buckets=[0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01],
    registry=REGISTRY
)


# ==================== Trading Metrics ====================

trading_positions = Gauge(
    "trading_positions",
    "Current number of open positions",
    ["symbol", "side"],
    registry=REGISTRY
)

trading_trades = Counter(
    "trading_trades_total",
    "Total trades executed",
    ["symbol", "side", "order_type", "status"],
    registry=REGISTRY
)

trading_pnl = Gauge(
    "trading_pnl",
    "Current unrealized PnL",
    ["symbol"],
    registry=REGISTRY
)

trading_balance = Gauge(
    "trading_balance",
    "Account balance",
    ["account_type", "currency"],
    registry=REGISTRY
)


# ==================== Worker Metrics ====================

worker_starts = Counter(
    "worker_starts_total",
    "Total worker starts",
    ["worker_type", "symbol", "timeframe"],
    registry=REGISTRY
)

worker_stops = Counter(
    "worker_stops_total",
    "Total worker stops",
    ["worker_type", "symbol", "timeframe", "reason"],
    registry=REGISTRY
)

worker_errors = Counter(
    "worker_errors_total",
    "Total worker errors",
    ["worker_type", "symbol", "timeframe", "error_type"],
    registry=REGISTRY
)

worker_processing_time = Histogram(
    "worker_processing_time_seconds",
    "Worker processing time per cycle",
    ["worker_type", "symbol", "timeframe"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)


# ==================== UpdateBus Metrics ====================

update_bus_messages = Counter(
    "update_bus_messages_total",
    "Total messages published to UpdateBus",
    ["message_type"],
    registry=REGISTRY
)

update_bus_dropped = Counter(
    "update_bus_dropped_total",
    "Total dropped messages from UpdateBus",
    ["message_type", "reason"],
    registry=REGISTRY
)

update_bus_queue_size = Gauge(
    "update_bus_queue_size",
    "Current UpdateBus queue size",
    [],
    registry=REGISTRY
)


# ==================== Utility Functions ====================

def get_metrics_text() -> bytes:
    """Get metrics in Prometheus exposition format.
    
    Returns:
        Metrics text in Prometheus format
    """
    return generate_latest(REGISTRY)


def is_metrics_enabled() -> bool:
    """Check if Prometheus metrics are enabled.
    
    Returns:
        True if metrics are available and enabled
    """
    if not PROMETHEUS_AVAILABLE:
        return False
    
    # Can be disabled via environment variable
    return os.getenv("DISABLE_PROMETHEUS_METRICS", "").lower() not in ("1", "true", "yes")


def record_api_call(endpoint: str, method: str, latency: float, status: str, error_code: str = ""):
    """Record an API call with metrics.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        latency: Request latency in seconds
        status: Response status (success/error)
        error_code: Error code if status is error
    """
    api_requests.labels(endpoint=endpoint, method=method, status=status).inc()
    api_latency.labels(endpoint=endpoint, method=method, status=status).observe(latency)
    
    if status == "error" and error_code:
        api_errors.labels(endpoint=endpoint, method=method, error_code=error_code).inc()


def record_cache_operation(cache_name: str, operation: str, hit: Optional[bool] = None):
    """Record a cache operation.
    
    Args:
        cache_name: Name of the cache
        operation: Operation type (get/set/delete)
        hit: Whether it was a cache hit (None for non-lookup operations)
    """
    if hit is True:
        cache_hits.labels(cache_name=cache_name, key_type="default").inc()
    elif hit is False:
        cache_misses.labels(cache_name=cache_name, key_type="default").inc()
    
    # Record latency for all operations
    cache_latency.labels(cache_name=cache_name, operation=operation)


def record_signal_execution(symbol: str, success: bool, latency: float, error_type: str = ""):
    """Record a signal execution.
    
    Args:
        symbol: Trading symbol
        success: Whether execution succeeded
        latency: Execution latency in seconds
        error_type: Type of error if failed
    """
    status = "success" if success else "error"
    signal_executions.labels(symbol=symbol, status=status, error_type=error_type or "none").inc()
    signal_execution_latency.labels(symbol=symbol, status=status).observe(latency)
