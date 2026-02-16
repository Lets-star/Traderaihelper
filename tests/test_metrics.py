"""
Tests for the metrics module.

Tests metric registration, collection, and Prometheus exposition format.
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from metrics import (
    websocket_connections,
    websocket_reconnections,
    websocket_latency,
    websocket_messages,
    signal_executions,
    signal_execution_latency,
    api_requests,
    api_latency,
    cache_hits,
    cache_misses,
    record_api_call,
    record_cache_operation,
    record_signal_execution,
    get_metrics_text,
    is_metrics_enabled,
    PROMETHEUS_AVAILABLE,
)

from metrics.collectors import (
    SignalExecutionCollector,
    WebSocketMetricsCollector,
    APIMetricsCollector,
    CacheMetricsCollector,
    WorkerMetricsCollector,
    UpdateBusMetricsCollector,
)


class TestMetricRegistration:
    """Test that metrics are properly registered."""
    
    def test_websocket_metrics_exist(self):
        """Test WebSocket metrics are defined."""
        # These should not raise AttributeError
        assert websocket_connections is not None
        assert websocket_reconnections is not None
        assert websocket_latency is not None
        assert websocket_messages is not None
    
    def test_signal_execution_metrics_exist(self):
        """Test signal execution metrics are defined."""
        assert signal_executions is not None
        assert signal_execution_latency is not None
    
    def test_api_metrics_exist(self):
        """Test API metrics are defined."""
        assert api_requests is not None
        assert api_latency is not None
    
    def test_cache_metrics_exist(self):
        """Test cache metrics are defined."""
        assert cache_hits is not None
        assert cache_misses is not None


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus-client not installed")
class TestMetricLabels:
    """Test metric labels work correctly."""
    
    def test_websocket_connection_labels(self):
        """Test WebSocket connection metric with labels."""
        # Should not raise
        websocket_connections.labels(symbol="BTCUSDT", interval="1h", status="success").inc()
    
    def test_websocket_latency_labels(self):
        """Test WebSocket latency metric with labels."""
        websocket_latency.labels(symbol="BTCUSDT", interval="1h").observe(0.05)
    
    def test_signal_execution_labels(self):
        """Test signal execution metric with labels."""
        signal_executions.labels(symbol="BTCUSDT", status="success", error_type="none").inc()
    
    def test_api_request_labels(self):
        """Test API request metric with labels."""
        api_requests.labels(endpoint="/v5/order", method="POST", status="success").inc()


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus-client not installed")
class TestMetricHelpers:
    """Test metric helper functions."""
    
    def test_record_api_call(self):
        """Test record_api_call helper."""
        record_api_call("/v5/order", "POST", 0.1, "success")
        # Should not raise
    
    def test_record_api_call_with_error(self):
        """Test record_api_call with error."""
        record_api_call("/v5/order", "POST", 0.1, "error", "RATE_LIMIT")
    
    def test_record_cache_operation_hit(self):
        """Test record_cache_operation with hit."""
        record_cache_operation("candle_cache", "get", hit=True)
    
    def test_record_cache_operation_miss(self):
        """Test record_cache_operation with miss."""
        record_cache_operation("candle_cache", "get", hit=False)
    
    def test_record_cache_operation_no_hit(self):
        """Test record_cache_operation without hit parameter."""
        record_cache_operation("candle_cache", "set", hit=None)
    
    def test_record_signal_execution_success(self):
        """Test record_signal_execution with success."""
        record_signal_execution("BTCUSDT", True, 0.5)
    
    def test_record_signal_execution_error(self):
        """Test record_signal_execution with error."""
        record_signal_execution("BTCUSDT", False, 0.5, "validation")
    
    def test_get_metrics_text(self):
        """Test get_metrics_text returns bytes."""
        text = get_metrics_text()
        assert isinstance(text, bytes)
    
    def test_is_metrics_enabled_default(self):
        """Test metrics enabled by default."""
        # Clear any environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert is_metrics_enabled() is True
    
    def test_is_metrics_disabled_via_env(self):
        """Test metrics disabled via environment variable."""
        with patch.dict(os.environ, {"DISABLE_PROMETHEUS_METRICS": "1"}):
            assert is_metrics_enabled() is False
    
    def test_is_metrics_disabled_via_true(self):
        """Test metrics disabled with 'true' value."""
        with patch.dict(os.environ, {"DISABLE_PROMETHEUS_METRICS": "true"}):
            assert is_metrics_enabled() is False


class TestSignalExecutionCollector:
    """Test SignalExecutionCollector."""
    
    def test_record_execution(self):
        """Test recording a signal execution."""
        collector = SignalExecutionCollector()
        collector.record("sig_001", "BTCUSDT", "filled", 100.0)
        
        stats = collector.get_stats(window_seconds=60)
        assert stats["total"] == 1
        assert stats["successful"] == 1
    
    def test_record_multiple_executions(self):
        """Test recording multiple executions."""
        collector = SignalExecutionCollector()
        collector.record("sig_001", "BTCUSDT", "filled", 100.0)
        collector.record("sig_002", "BTCUSDT", "error", 200.0, "Network timeout")
        collector.record("sig_003", "ETHUSDT", "filled", 150.0)
        
        stats = collector.get_stats(window_seconds=60)
        assert stats["total"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
    
    def test_record_validation_error(self):
        """Test recording validation error."""
        collector = SignalExecutionCollector()
        collector.record_validation_error("BTCUSDT", "symbol")
        # Should not raise
    
    def test_stats_empty_history(self):
        """Test stats with empty history."""
        collector = SignalExecutionCollector()
        stats = collector.get_stats()
        
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency_ms"] == 0.0
    
    def test_stats_latency_percentiles(self):
        """Test latency percentile calculation."""
        collector = SignalExecutionCollector()
        
        # Add executions with varying latencies
        for i in range(100):
            collector.record(f"sig_{i}", "BTCUSDT", "filled", float(i * 10))
        
        stats = collector.get_stats(window_seconds=60)
        assert stats["p50_latency_ms"] > 0
        assert stats["p95_latency_ms"] > stats["p50_latency_ms"]
        assert stats["p99_latency_ms"] >= stats["p95_latency_ms"]
    
    def test_history_limit(self):
        """Test history size limit."""
        collector = SignalExecutionCollector(max_history=10)
        
        for i in range(20):
            collector.record(f"sig_{i}", "BTCUSDT", "filled", 100.0)
        
        stats = collector.get_stats(window_seconds=60)
        # Should only have the most recent 10
        assert stats["total"] == 10


class TestWebSocketMetricsCollector:
    """Test WebSocketMetricsCollector."""
    
    def test_record_connect(self):
        """Test recording connection."""
        collector = WebSocketMetricsCollector()
        collector.record_connect("BTCUSDT", "1h", True, 50.0)
        
        stats = collector.get_connection_stats()
        assert stats["connects"] == 1
        assert stats["active_connections"] == 1
    
    def test_record_connect_failure(self):
        """Test recording failed connection."""
        collector = WebSocketMetricsCollector()
        collector.record_connect("BTCUSDT", "1h", False, 0)
        
        stats = collector.get_connection_stats()
        # Failed connects are tracked but don't count as active
        assert stats["active_connections"] == 0
    
    def test_record_disconnect(self):
        """Test recording disconnection."""
        collector = WebSocketMetricsCollector()
        collector.record_connect("BTCUSDT", "1h", True, 50.0)
        collector.record_disconnect("BTCUSDT", "1h", "network_error")
        
        stats = collector.get_connection_stats()
        assert stats["disconnects"] == 1
        assert stats["active_connections"] == 0
    
    def test_record_reconnect(self):
        """Test recording reconnection."""
        collector = WebSocketMetricsCollector()
        collector.record_reconnect("BTCUSDT", "1h", 1)
        
        stats = collector.get_connection_stats()
        assert stats["reconnects"] == 1
    
    def test_record_error(self):
        """Test recording error."""
        collector = WebSocketMetricsCollector()
        collector.record_error("BTCUSDT", "1h", "connection_reset")
        
        stats = collector.get_connection_stats()
        assert stats["errors"] == 1
    
    def test_record_message(self):
        """Test recording message."""
        collector = WebSocketMetricsCollector()
        collector.record_message("BTCUSDT", "1h", "closed")
        collector.record_message("BTCUSDT", "1h", "forming")
    
    def test_multiple_connections(self):
        """Test multiple active connections."""
        collector = WebSocketMetricsCollector()
        collector.record_connect("BTCUSDT", "1h", True, 50.0)
        collector.record_connect("ETHUSDT", "1h", True, 50.0)
        
        assert collector.get_active_connections() == 2


class TestAPIMetricsCollector:
    """Test APIMetricsCollector."""
    
    def test_record_request(self):
        """Test recording API request."""
        collector = APIMetricsCollector()
        collector.record_request("/v5/order", "POST", "success", 150.0)
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["success_rate"] == 100.0
    
    def test_record_request_error(self):
        """Test recording failed request."""
        collector = APIMetricsCollector()
        collector.record_request("/v5/order", "POST", "error", 200.0, "RATE_LIMIT")
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful"] == 0
    
    def test_record_rate_limit(self):
        """Test recording rate limit."""
        collector = APIMetricsCollector()
        collector.record_rate_limit("/v5/order", "POST")
        
        stats = collector.get_stats()
        assert stats["rate_limit_hits"] == 1
    
    def test_stats_with_latencies(self):
        """Test stats include latency info."""
        collector = APIMetricsCollector()
        collector.record_request("/v5/order", "POST", "success", 100.0)
        collector.record_request("/v5/order", "POST", "success", 200.0)
        
        stats = collector.get_stats()
        assert stats["avg_latency_ms"] == 150.0
        assert stats["max_latency_ms"] == 200.0


class TestCacheMetricsCollector:
    """Test CacheMetricsCollector."""
    
    def test_record_hit(self):
        """Test recording cache hit."""
        collector = CacheMetricsCollector()
        collector.record_hit("candle_cache")
    
    def test_record_miss(self):
        """Test recording cache miss."""
        collector = CacheMetricsCollector()
        collector.record_miss("candle_cache")
    
    def test_update_size(self):
        """Test updating cache size."""
        collector = CacheMetricsCollector()
        collector.update_size("candle_cache", 100)
    
    def test_record_eviction(self):
        """Test recording eviction."""
        collector = CacheMetricsCollector()
        collector.record_eviction("candle_cache", "size_limit")


class TestWorkerMetricsCollector:
    """Test WorkerMetricsCollector."""
    
    def test_record_start(self):
        """Test recording worker start."""
        collector = WorkerMetricsCollector()
        collector.record_start("chart_worker", "BTCUSDT", "1h")
    
    def test_record_stop(self):
        """Test recording worker stop."""
        collector = WorkerMetricsCollector()
        collector.record_stop("chart_worker", "BTCUSDT", "1h", "normal")
    
    def test_record_error(self):
        """Test recording worker error."""
        collector = WorkerMetricsCollector()
        collector.record_error("chart_worker", "BTCUSDT", "1h", "websocket_error")
    
    def test_record_processing_time(self):
        """Test recording processing time."""
        collector = WorkerMetricsCollector()
        collector.record_processing_time("chart_worker", "BTCUSDT", "1h", 0.05)


class TestUpdateBusMetricsCollector:
    """Test UpdateBusMetricsCollector."""
    
    def test_record_publish(self):
        """Test recording message publish."""
        collector = UpdateBusMetricsCollector()
        collector.record_publish("chart_update")
    
    def test_record_dropped(self):
        """Test recording dropped message."""
        collector = UpdateBusMetricsCollector()
        collector.record_dropped("chart_update", "queue_full")
    
    def test_update_queue_size(self):
        """Test updating queue size."""
        collector = UpdateBusMetricsCollector()
        collector.update_queue_size(10)
