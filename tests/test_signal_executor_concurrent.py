"""
Concurrent execution tests for SignalExecutor.

Tests ThreadPoolExecutor behavior, thread-safe CSV logging,
concurrent signal execution, race conditions in statistics tracking,
UpdateBus integration under load, and executor shutdown/cleanup.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest

from signal_executor import SignalExecutor


class TestSignalExecutorThreadPool:
    """Test ThreadPoolExecutor behavior."""
    
    def test_executor_creation(self):
        """Test that ThreadPoolExecutor is created with correct max workers."""
        executor = SignalExecutor()
        assert executor._executor._max_workers == SignalExecutor.MAX_WORKER_THREADS
    
    def test_executor_max_workers_limit(self):
        """Test that max workers is respected."""
        executor = SignalExecutor()
        assert executor.MAX_WORKER_THREADS == 3
    
    def test_multiple_executions_concurrent(self):
        """Test that multiple signals can be executed concurrently."""
        executor = SignalExecutor()
        executor.enabled = True
        executor.dry_run = True
        
        signals = [
            {
                "signal_id": f"sig_{i}",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "quantity": 0.001,
            }
            for i in range(10)
        ]
        
        start_time = time.time()
        
        # Submit all signals
        for signal in signals:
            executor.execute_signal(signal)
        
        # Wait for completion
        executor._executor.shutdown(wait=True)
        
        elapsed = time.time() - start_time
        
        # Should complete in less than sequential time (10 * 0.1s = 1s)
        # With 3 workers, should take roughly 0.4s
        assert elapsed < 0.8


class TestSignalExecutorThreadSafety:
    """Test thread safety of SignalExecutor."""
    
    def test_statistics_thread_safety(self):
        """Test that statistics tracking is thread-safe."""
        executor = SignalExecutor()
        executor.enabled = True
        executor.dry_run = True
        
        errors = []
        
        def execute_and_track(signal):
            try:
                executor._execute_signal_sync(signal)
            except Exception as e:
                errors.append(e)
        
        signals = [
            {
                "signal_id": f"sig_{i}",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "quantity": 0.001,
            }
            for i in range(50)
        ]
        
        threads = []
        for signal in signals:
            t = threading.Thread(target=execute_and_track, args=(signal,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=10)
        
        # No errors should occur
        assert len(errors) == 0
        
        # Statistics should match
        stats = executor.get_statistics()
        assert stats["total_executions"] == 50
    
    def test_concurrent_statistics_update(self):
        """Test concurrent updates to statistics counters."""
        executor = SignalExecutor()
        
        def update_stats():
            with executor._lock:
                executor._total_executions += 1
                executor._successful_executions += 1
        
        threads = [threading.Thread(target=update_stats) for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = executor.get_statistics()
        assert stats["total_executions"] == 100
        assert stats["successful_executions"] == 100


class TestSignalExecutorCSVLogging:
    """Test thread-safe CSV logging."""
    
    def test_csv_logging_thread_safety(self, tmp_path):
        """Test that CSV logging is thread-safe under concurrent writes."""
        log_file = str(tmp_path / "test_trades.csv")
        
        executor = SignalExecutor()
        executor.LOG_FILE = log_file
        executor._ensure_log_file()
        
        def log_trade(i):
            executor._log_trade({
                "signal_id": f"sig_{i}",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "qty": "0.001",
                "entry_price": "50000",
                "status": "filled",
                "response_code": "0",
                "latency_ms": "100",
                "error_msg": "",
                "validation_errors": "",
                "thread_id": str(threading.current_thread().ident),
            })
        
        threads = [threading.Thread(target=log_trade, args=(i,)) for i in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify file was written correctly
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Header + 20 trades
        assert len(lines) == 21
    
    def test_csv_logging_under_load(self, tmp_path):
        """Test CSV logging performance under high load."""
        log_file = str(tmp_path / "load_test_trades.csv")
        
        executor = SignalExecutor()
        executor.LOG_FILE = log_file
        executor._ensure_log_file()
        
        start_time = time.time()
        
        def log_multiple_trades(thread_id):
            for i in range(10):
                executor._log_trade({
                    "signal_id": f"thread{thread_id}_sig_{i}",
                    "symbol": "BTCUSDT",
                    "direction": "LONG",
                    "qty": "0.001",
                    "entry_price": "50000",
                    "status": "filled",
                    "response_code": "0",
                    "latency_ms": "100",
                    "error_msg": "",
                    "validation_errors": "",
                    "thread_id": str(threading.current_thread().ident),
                })
        
        threads = [threading.Thread(target=log_multiple_trades, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 5.0
        
        # Verify all trades logged
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Header + 100 trades (10 threads * 10 trades each)
        assert len(lines) == 101


class TestSignalExecutorUpdateBus:
    """Test UpdateBus integration under load."""
    
    def test_update_bus_publishes_under_load(self):
        """Test UpdateBus publishes correctly under concurrent load."""
        bus = MagicMock()
        bus.publish = MagicMock(return_value=True)
        
        executor = SignalExecutor(update_bus=bus)
        executor.enabled = True
        executor.dry_run = True
        
        signals = [
            {
                "signal_id": f"sig_{i}",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "quantity": 0.001,
            }
            for i in range(20)
        ]
        
        for signal in signals:
            executor.execute_signal(signal)
        
        # Wait for completion
        executor._executor.shutdown(wait=True)
        
        # Should have published updates (at least status updates)
        assert bus.publish.call_count > 0
    
    def test_update_bus_handles_errors_gracefully(self):
        """Test UpdateBus errors don't crash execution."""
        bus = MagicMock()
        bus.publish = MagicMock(side_effect=Exception("Bus error"))
        
        executor = SignalExecutor(update_bus=bus)
        executor.enabled = True
        executor.dry_run = True
        
        signal = {
            "signal_id": "sig_001",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry_price": 50000.0,
            "quantity": 0.001,
        }
        
        # Should not raise
        executor.execute_signal(signal)
        executor._executor.shutdown(wait=True)


class TestSignalExecutorCleanup:
    """Test executor shutdown and cleanup."""
    
    def test_cleanup_shuts_down_executor(self):
        """Test that cleanup properly shuts down the executor."""
        executor = SignalExecutor()
        
        # Submit some tasks
        executor.enabled = True
        executor.dry_run = True
        
        for i in range(5):
            executor.execute_signal({
                "signal_id": f"sig_{i}",
                "symbol": "BTCUSDT",
                "direction": "LONG",
                "entry_price": 50000.0,
                "quantity": 0.001,
            })
        
        # Cleanup
        executor.cleanup()
        
        # Executor should be shut down
        assert executor._executor._shutdown
    
    def test_cleanup_can_be_called_multiple_times(self):
        """Test that cleanup can be safely called multiple times."""
        executor = SignalExecutor()
        
        executor.cleanup()
        
        # Second cleanup should not raise
        executor.cleanup()
    
    def test_destructor_calls_cleanup(self):
        """Test that destructor attempts cleanup."""
        executor = SignalExecutor()
        
        # Delete should attempt cleanup without error
        del executor
        
        # If we get here, no exception was raised
        assert True


class TestSignalExecutorValidationConcurrency:
    """Test validation under concurrent execution."""
    
    def test_concurrent_validation(self):
        """Test that validation works correctly under concurrent access."""
        executor = SignalExecutor()
        
        signals = [
            {"signal_id": "", "symbol": "", "direction": "INVALID"},  # Invalid
            {"signal_id": "sig_1", "symbol": "BTCUSDT", "direction": "LONG", "entry_price": 50000},  # Valid
            {"signal_id": "sig_2", "symbol": "ETHUSDT", "direction": "SHORT", "entry_price": 3000},  # Valid
        ]
        
        results = []
        
        def validate(signal):
            errors = executor._validate_signal(signal)
            results.append(len(errors))
        
        threads = [threading.Thread(target=validate, args=(s,)) for s in signals]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have validation results for all
        assert len(results) == 3
        assert sum(results) > 0  # At least one signal had errors


class TestSignalExecutorRaceConditions:
    """Test for race conditions in SignalExecutor."""
    
    def test_race_condition_in_configure(self):
        """Test for race condition during configuration."""
        executor = SignalExecutor()
        
        def configure_1():
            executor.configure(
                enabled=True,
                api_key="key1",
                api_secret="secret1",
                testnet=True
            )
        
        def configure_2():
            executor.configure(
                enabled=True,
                api_key="key2",
                api_secret="secret2",
                testnet=False
            )
        
        threads = [
            threading.Thread(target=configure_1),
            threading.Thread(target=configure_2),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should end up with one of the configurations
        assert executor.api_key in ["key1", "key2"]
    
    def test_race_condition_in_statistics_reset(self):
        """Test for race condition between get and reset statistics."""
        executor = SignalExecutor()
        executor._total_executions = 100
        
        results = []
        
        def get_stats():
            results.append(executor.get_statistics())
        
        def reset_stats():
            executor.reset_statistics()
        
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=get_stats))
            threads.append(threading.Thread(target=reset_stats))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have results without errors
        assert len(results) == 10


class TestSignalExecutorWithMockClient:
    """Test with mocked ByBitClient."""
    
    def test_concurrent_api_calls(self):
        """Test concurrent API calls through executor."""
        executor = SignalExecutor()
        executor.enabled = True
        executor.api_key = "test_key"
        executor.api_secret = "test_secret"
        executor.testnet = True
        
        with patch('signal_executor.ByBitClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.validate_credentials.return_value = True
            mock_client.set_leverage.return_value = {"retCode": 0}
            mock_client.place_order.return_value = {"retCode": 0}
            
            signals = [
                {
                    "signal_id": f"sig_{i}",
                    "symbol": "BTCUSDT",
                    "direction": "LONG",
                    "entry_price": 50000.0,
                    "quantity": 0.001,
                }
                for i in range(10)
            ]
            
            for signal in signals:
                executor.execute_signal(signal)
            
            # Wait for completion
            executor._executor.shutdown(wait=True)
            
            # Should have made API calls
            assert mock_client.place_order.call_count == 10
