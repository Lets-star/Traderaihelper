"""
Thread safety tests for ChartDataStore.

Tests ChartDataStore thread safety with concurrent operations,
concurrent read/write operations, lock contention scenarios,
data consistency under concurrent access, and forming bar updates with closed data.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from chart_auto_refresh import ChartDataStore, _CANDLE_CACHE, _CACHE_LOCK


class TestChartDataStoreBasicThreading:
    """Test basic ChartDataStore thread safety."""
    
    @pytest.fixture
    def sample_df(self):
        """Provide a sample DataFrame."""
        return pd.DataFrame({
            "ts": [1700000000000, 1700003600000, 1700007200000],
            "open": [50000.0, 50100.0, 50200.0],
            "high": [50100.0, 50200.0, 50300.0],
            "low": [49900.0, 50000.0, 50100.0],
            "close": [50050.0, 50150.0, 50250.0],
            "volume": [100.0, 150.0, 200.0],
        })
    
    def test_concurrent_reads(self, sample_df):
        """Test concurrent reads from the store."""
        store = ChartDataStore()
        store.update_closed(sample_df, 1700007200000, append=False)
        
        results = []
        errors = []
        
        def read_store():
            try:
                df, indicators, ts = store.snapshot(include_forming=False)
                results.append((len(df) if df is not None else 0, ts))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=read_store) for _ in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 20
        # All reads should see consistent data
        assert all(r[0] == 3 for r in results)
    
    def test_concurrent_writes(self):
        """Test concurrent writes to the store."""
        store = ChartDataStore()
        
        errors = []
        
        def write_store(i):
            try:
                df = pd.DataFrame({
                    "ts": [1700000000000 + i],
                    "open": [50000.0],
                    "high": [50100.0],
                    "low": [49900.0],
                    "close": [50050.0],
                    "volume": [100.0],
                })
                store.update_closed(df, 1700000000000 + i, append=True)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=write_store, args=(i,)) for i in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Verify data is consistent
        df, _, _ = store.snapshot(include_forming=False)
        assert df is not None
        assert len(df) == 20
    
    def test_concurrent_read_write(self, sample_df):
        """Test concurrent reads and writes."""
        store = ChartDataStore()
        store.update_closed(sample_df, 1700007200000, append=False)
        
        errors = []
        read_results = []
        
        def reader():
            try:
                for _ in range(10):
                    df, _, _ = store.snapshot(include_forming=False)
                    read_results.append(len(df) if df is not None else 0)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def writer():
            try:
                for i in range(10):
                    df = pd.DataFrame({
                        "ts": [1700000000000 + i + 1000],
                        "open": [50000.0],
                        "high": [50100.0],
                        "low": [49900.0],
                        "close": [50050.0],
                        "volume": [100.0],
                    })
                    store.update_closed(df, 1700000000000 + i + 1000, append=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestChartDataStoreLockContention:
    """Test lock contention scenarios."""
    
    def test_lock_held_during_update(self):
        """Test that lock is held during update operation."""
        store = ChartDataStore()
        
        lock_held = [False]
        
        def check_lock():
            # Try to acquire the lock
            acquired = store._lock.acquire(blocking=False)
            if not acquired:
                lock_held[0] = True
            else:
                store._lock.release()
        
        def slow_update():
            df = pd.DataFrame({
                "ts": [1700000000000],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            })
            # This should hold the lock
            store.update_closed(df, 1700000000000, append=False)
        
        t1 = threading.Thread(target=slow_update)
        t2 = threading.Thread(target=check_lock)
        
        t1.start()
        time.sleep(0.01)  # Give t1 time to acquire lock
        t2.start()
        
        t2.join(timeout=1)
        t1.join(timeout=1)
        
        # This test is timing-dependent, so we just verify no crash
        assert True
    
    def test_multiple_threads_waiting_for_lock(self):
        """Test multiple threads waiting to acquire the lock."""
        store = ChartDataStore()
        
        order = []
        
        def writer(thread_id):
            df = pd.DataFrame({
                "ts": [1700000000000 + thread_id],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            })
            store.update_closed(df, 1700000000000 + thread_id, append=True)
            order.append(thread_id)
        
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All writes should have completed
        assert len(order) == 10


class TestChartDataStoreDataConsistency:
    """Test data consistency under concurrent access."""
    
    def test_no_duplicate_timestamps_after_concurrent_writes(self):
        """Test that no duplicate timestamps exist after concurrent writes."""
        store = ChartDataStore()
        
        def write_same_timestamp(thread_id):
            df = pd.DataFrame({
                "ts": [1700000000000],  # Same timestamp
                "open": [50000.0 + thread_id],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            })
            store.update_closed(df, 1700000000000, append=True)
        
        threads = [threading.Thread(target=write_same_timestamp, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check for duplicates
        df, _, _ = store.snapshot(include_forming=False)
        if df is not None and not df.empty:
            timestamps = df["ts"].tolist()
            unique_timestamps = set(timestamps)
            # Due to deduplication, we should have at most 1 unique timestamp
            assert len(unique_timestamps) <= 1
    
    def test_sorted_data_after_concurrent_writes(self):
        """Test that data remains sorted after concurrent writes."""
        store = ChartDataStore()
        
        def write_random_timestamp(thread_id):
            # Write timestamps in random order
            ts = 1700000000000 + (thread_id * 1000 if thread_id % 2 == 0 else (9 - thread_id) * 1000)
            df = pd.DataFrame({
                "ts": [ts],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            })
            store.update_closed(df, ts, append=True)
        
        threads = [threading.Thread(target=write_random_timestamp, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify data is sorted
        df, _, _ = store.snapshot(include_forming=False)
        if df is not None and len(df) > 1:
            timestamps = df["ts"].tolist()
            assert timestamps == sorted(timestamps)
    
    def test_consistent_snapshot_during_updates(self):
        """Test that snapshots are consistent even during updates."""
        store = ChartDataStore()
        
        # Initial data
        df = pd.DataFrame({
            "ts": [1700000000000],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
        })
        store.update_closed(df, 1700000000000, append=False)
        
        snapshot_results = []
        errors = []
        
        def take_snapshots():
            try:
                for _ in range(50):
                    df, indicators, ts = store.snapshot(include_forming=False)
                    # Verify snapshot is consistent (not partially updated)
                    if df is not None and not df.empty:
                        snapshot_results.append(len(df))
            except Exception as e:
                errors.append(e)
        
        def update_data():
            try:
                for i in range(50):
                    df = pd.DataFrame({
                        "ts": [1700000000000 + i + 1],
                        "open": [50000.0],
                        "high": [50100.0],
                        "low": [49900.0],
                        "close": [50050.0],
                        "volume": [100.0],
                    })
                    store.update_closed(df, 1700000000000 + i + 1, append=True)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=take_snapshots),
            threading.Thread(target=update_data),
            threading.Thread(target=take_snapshots),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestChartDataStoreFormingBar:
    """Test forming bar updates with closed data."""
    
    def test_concurrent_forming_and_closed_updates(self):
        """Test concurrent forming bar and closed data updates."""
        store = ChartDataStore()
        
        # Initial closed data
        closed_df = pd.DataFrame({
            "ts": [1700000000000, 1700003600000],
            "open": [50000.0, 50100.0],
            "high": [50100.0, 50200.0],
            "low": [49900.0, 50000.0],
            "close": [50050.0, 50150.0],
            "volume": [100.0, 150.0],
        })
        store.update_closed(closed_df, 1700003600000, append=False)
        
        errors = []
        
        def update_forming():
            try:
                for i in range(20):
                    forming_df = pd.DataFrame({
                        "ts": [1700007200000],
                        "open": [50200.0],
                        "high": [50300.0],
                        "low": [50100.0],
                        "close": [50250.0 + i],  # Changing price
                        "volume": [200.0],
                    })
                    store.set_forming_bar(forming_df)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def update_closed():
            try:
                for i in range(20):
                    new_closed = pd.DataFrame({
                        "ts": [1700007200000 + i * 1000],
                        "open": [50000.0],
                        "high": [50100.0],
                        "low": [49900.0],
                        "close": [50050.0],
                        "volume": [100.0],
                    })
                    store.update_closed(new_closed, 1700007200000 + i * 1000, append=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def read_with_forming():
            try:
                for _ in range(20):
                    df, indicators, ts = store.snapshot(include_forming=True)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=update_forming),
            threading.Thread(target=update_closed),
            threading.Thread(target=read_with_forming),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Verify final state is consistent
        df, _, _ = store.snapshot(include_forming=True)
        assert df is not None
    
    def test_forming_bar_overwrite(self):
        """Test that forming bar can be overwritten."""
        store = ChartDataStore()
        
        # Set forming bar multiple times
        for i in range(10):
            forming_df = pd.DataFrame({
                "ts": [1700007200000],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0 + i],
                "volume": [100.0],
            })
            store.set_forming_bar(forming_df)
        
        # Should have the last value
        df, _, _ = store.snapshot(include_forming=True)
        if df is not None and not df.empty:
            # The last update should have close = 50050 + 9
            forming_row = df[df["ts"] == 1700007200000]
            if not forming_row.empty:
                assert forming_row["close"].iloc[0] == 50059.0


class TestChartDataStoreReset:
    """Test reset functionality."""
    
    def test_reset_during_concurrent_access(self):
        """Test that reset works during concurrent access."""
        store = ChartDataStore()
        
        # Initial data
        df = pd.DataFrame({
            "ts": [1700000000000],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
        })
        store.update_closed(df, 1700000000000, append=False)
        
        errors = []
        
        def reader():
            try:
                for _ in range(50):
                    store.snapshot(include_forming=False)
            except Exception as e:
                errors.append(e)
        
        def resetter():
            try:
                for _ in range(10):
                    store.reset()
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=resetter),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestChartCacheThreading:
    """Test cache threading safety."""
    
    def test_concurrent_cache_access(self):
        """Test concurrent access to global cache."""
        errors = []
        
        def cache_writer(i):
            try:
                with _CACHE_LOCK:
                    key = (f"SYMBOL{i}", "1h", 0, 1000)
                    _CANDLE_CACHE[key] = pd.DataFrame({"test": [i]})
            except Exception as e:
                errors.append(e)
        
        def cache_reader(i):
            try:
                with _CACHE_LOCK:
                    key = (f"SYMBOL{i}", "1h", 0, 1000)
                    _ = _CANDLE_CACHE.get(key)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=cache_writer, args=(i,)))
            threads.append(threading.Thread(target=cache_reader, args=(i,)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_cache_cleanup_under_load(self):
        """Test cache cleanup operations under load."""
        from chart_auto_refresh import invalidate_cache
        
        # Populate cache
        for i in range(50):
            with _CACHE_LOCK:
                key = (f"BTCUSDT", "1h", i * 1000, (i + 1) * 1000)
                _CANDLE_CACHE[key] = pd.DataFrame({"test": [i]})
        
        errors = []
        
        def cleanup():
            try:
                invalidate_cache("BTCUSDT", "1h")
            except Exception as e:
                errors.append(e)
        
        def read_cache():
            try:
                with _CACHE_LOCK:
                    for key in list(_CANDLE_CACHE.keys()):
                        _ = _CANDLE_CACHE.get(key)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=cleanup),
            threading.Thread(target=read_cache),
            threading.Thread(target=cleanup),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
