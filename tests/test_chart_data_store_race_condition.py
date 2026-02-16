"""Tests for ChartDataStore race condition fix."""

from __future__ import annotations

import threading
import time
import pytest
import pandas as pd
import numpy as np

from chart_auto_refresh import ChartDataStore


class TestChartDataStoreRaceCondition:
    """Test ChartDataStore for thread-safety and race condition fixes."""

    def create_test_df(self, num_rows=10, start_ts=1700000000000):
        """Create a test DataFrame."""
        data = {
            "ts": np.arange(start_ts, start_ts + num_rows * 60000, 60000),
            "open": np.random.rand(num_rows) * 100 + 100,
            "high": np.random.rand(num_rows) * 100 + 105,
            "low": np.random.rand(num_rows) * 100 + 95,
            "close": np.random.rand(num_rows) * 100 + 100,
            "volume": np.random.rand(num_rows) * 1000,
        }
        return pd.DataFrame(data)

    def test_set_forming_bar_thread_safety(self):
        """Test that set_forming_bar is thread-safe."""
        store = ChartDataStore()

        def update_forming_bar():
            for i in range(100):
                df = self.create_test_df(num_rows=5, start_ts=1700000000000 + i * 60000)
                store.set_forming_bar(df)

        def read_snapshot():
            for _ in range(100):
                df, indicators, last_closed = store.snapshot(include_forming=True)
                if df is not None:
                    assert isinstance(df, pd.DataFrame)

        # Create threads
        threads = [
            threading.Thread(target=update_forming_bar),
            threading.Thread(target=read_snapshot),
        ]

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)

        assert not any(t.is_alive() for t in threads)

    def test_update_closed_and_forming_bar_concurrent(self):
        """Test concurrent update_closed and set_forming_bar calls."""
        store = ChartDataStore()

        def update_closed_bars():
            for i in range(50):
                df = self.create_test_df(num_rows=10, start_ts=1700000000000 + i * 600000)
                store.update_closed(df, 1700000000000 + i * 600000, append=True)

        def update_forming_bars():
            for i in range(50):
                df = self.create_test_df(num_rows=1, start_ts=1700000000000 + i * 600000)
                store.set_forming_bar(df)

        # Create threads
        threads = [
            threading.Thread(target=update_closed_bars),
            threading.Thread(target=update_forming_bars),
        ]

        # Start threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=10.0)

        assert not any(t.is_alive() for t in threads)

        # Verify final state is consistent
        df_closed, indicators, last_closed = store.snapshot(include_forming=False)
        assert df_closed is not None
        assert len(df_closed) > 0

    def test_rebuild_with_forming_locked_captures_reference(self):
        """Test that _rebuild_with_forming_locked captures reference under lock."""
        store = ChartDataStore()

        # Initialize with closed data
        closed_df = self.create_test_df(num_rows=10)
        store.update_closed(closed_df, 1700000000000 + 10 * 60000, append=False)

        # Set forming bar
        forming_df = self.create_test_df(num_rows=1, start_ts=1700000000000 + 10 * 60000)
        store.set_forming_bar(forming_df)

        # Verify both datasets are present
        df_with_forming, indicators, last_closed = store.snapshot(include_forming=True)
        assert df_with_forming is not None
        # Should have 11 bars (10 closed + 1 forming)
        assert len(df_with_forming) == 11

    def test_clear_forming_bar_during_updates(self):
        """Test clearing forming bar while updates are happening."""
        store = ChartDataStore()

        # Initialize with data
        closed_df = self.create_test_df(num_rows=10)
        store.update_closed(closed_df, 1700000000000 + 10 * 60000, append=False)

        forming_df = self.create_test_df(num_rows=1, start_ts=1700000000000 + 10 * 60000)
        store.set_forming_bar(forming_df)

        def clear_and_set():
            for _ in range(50):
                store.clear_forming_bar()
                time.sleep(0.001)
                df = self.create_test_df(num_rows=1, start_ts=1700000000000 + 10 * 60000)
                store.set_forming_bar(df)

        def read_snapshots():
            for _ in range(50):
                df_with, _, _ = store.snapshot(include_forming=True)
                df_closed, _, _ = store.snapshot(include_forming=False)

                if df_closed is not None:
                    assert len(df_closed) == 10  # Should always have 10 closed bars

        threads = [
            threading.Thread(target=clear_and_set),
            threading.Thread(target=read_snapshots),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        assert not any(t.is_alive() for t in threads)

    def test_snapshot_consistency_during_rapid_updates(self):
        """Test that snapshots are consistent even during rapid updates."""
        store = ChartDataStore()

        def rapid_updates():
            for i in range(100):
                df = self.create_test_df(num_rows=5, start_ts=1700000000000 + i * 60000)
                store.update_closed(df, 1700000000000 + i * 60000 + 5 * 60000, append=True)
                time.sleep(0.001)

        def consistent_reads():
            last_len = 0
            for _ in range(100):
                df, _, last_closed = store.snapshot(include_forming=False)
                if df is not None:
                    current_len = len(df)
                    # Length should never decrease (append only)
                    assert current_len >= last_len
                    last_len = current_len
                time.sleep(0.001)

        threads = [
            threading.Thread(target=rapid_updates),
            threading.Thread(target=consistent_reads),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)

        assert not any(t.is_alive() for t in threads)

    def test_no_corruption_with_null_forming_bar(self):
        """Test that null forming bar doesn't cause corruption."""
        store = ChartDataStore()

        # Set initial closed data
        closed_df = self.create_test_df(num_rows=10)
        store.update_closed(closed_df, 1700000000000 + 10 * 60000, append=False)

        # Set forming bar to None
        store.set_forming_bar(None)

        # Verify snapshot still works
        df_closed, indicators, last_closed = store.snapshot(include_forming=False)
        assert df_closed is not None
        assert len(df_closed) == 10

        df_with_forming, _, _ = store.snapshot(include_forming=True)
        # Should be same as closed when no forming bar
        assert df_with_forming is None or len(df_with_forming) == 10

    def test_forming_bar_copy_isolation(self):
        """Test that forming bar copies don't affect internal state."""
        store = ChartDataStore()

        # Set forming bar
        forming_df = self.create_test_df(num_rows=1)
        store.set_forming_bar(forming_df)

        # Get snapshot
        df_snapshot, _, _ = store.snapshot(include_forming=True)

        # Modify snapshot (should not affect internal state)
        if df_snapshot is not None:
            df_snapshot.iloc[0, 0] = 999999999

            # Get new snapshot
            df_new, _, _ = store.snapshot(include_forming=True)

            # Should not be affected by previous modification
            if df_new is not None:
                assert df_new.iloc[0, 0] != 999999999
