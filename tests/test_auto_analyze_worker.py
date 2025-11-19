"""Tests for auto-analyze worker functionality."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from indicator_collector.trading_system.auto_analyze_worker import (
    floor_closed_bar,
    TIMEFRAME_TO_MS,
)


class TestFloorClosedBar:
    """Test floor_closed_bar function for various timeframes."""
    
    def test_floor_closed_bar_1m(self):
        """Test 1-minute timeframe."""
        tf_ms = TIMEFRAME_TO_MS["1m"]
        
        # At 12:00:30, current bar started at 12:00:00, last closed is 11:59:00
        now_ms = 1700000030000  # arbitrary timestamp + 30 seconds
        last_closed = floor_closed_bar(now_ms, tf_ms)
        
        # Expected: floor to 1m boundary, then go back 1m
        expected_current_bar = (now_ms // tf_ms) * tf_ms
        expected_last_closed = expected_current_bar - tf_ms
        
        assert last_closed == expected_last_closed
    
    def test_floor_closed_bar_1h(self):
        """Test 1-hour timeframe."""
        tf_ms = TIMEFRAME_TO_MS["1h"]
        
        # At 14:30:00, current bar started at 14:00:00, last closed is 13:00:00
        now_ms = 1700000000000 + (30 * 60 * 1000)  # + 30 minutes
        last_closed = floor_closed_bar(now_ms, tf_ms)
        
        expected_current_bar = (now_ms // tf_ms) * tf_ms
        expected_last_closed = expected_current_bar - tf_ms
        
        assert last_closed == expected_last_closed
    
    def test_floor_closed_bar_near_boundary(self):
        """Test when we're very close to a bar boundary."""
        tf_ms = TIMEFRAME_TO_MS["1h"]
        tol_ms = 60_000  # 60 seconds tolerance
        
        # At 14:00:30 (30 seconds into new bar) - within tolerance
        current_bar_start = 1700000000000
        now_ms = current_bar_start + (tol_ms // 2)  # 30 seconds
        
        last_closed = floor_closed_bar(now_ms, tf_ms, tol_ms)
        
        # Should go back one more bar because we're within tolerance
        expected_last_closed = current_bar_start - tf_ms
        assert last_closed == expected_last_closed
    
    def test_floor_closed_bar_3h(self):
        """Test 3-hour timeframe."""
        tf_ms = TIMEFRAME_TO_MS["3h"]
        
        # At 15:30:00, current 3h bar started at 15:00:00, last closed is 12:00:00
        # 3h bars: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
        base_ms = 1700000000000
        # Align to a 3h boundary
        aligned_base = (base_ms // tf_ms) * tf_ms
        now_ms = aligned_base + (tf_ms // 2)  # halfway through a 3h bar
        
        last_closed = floor_closed_bar(now_ms, tf_ms)
        
        expected_current_bar = (now_ms // tf_ms) * tf_ms
        expected_last_closed = expected_current_bar - tf_ms
        
        assert last_closed == expected_last_closed
    
    def test_floor_closed_bar_1d(self):
        """Test 1-day timeframe."""
        tf_ms = TIMEFRAME_TO_MS["1d"]
        
        # At some point during the day
        base_ms = 1700000000000
        # Align to start of day (00:00:00 UTC)
        aligned_base = (base_ms // tf_ms) * tf_ms
        now_ms = aligned_base + (12 * 60 * 60 * 1000)  # noon
        
        last_closed = floor_closed_bar(now_ms, tf_ms)
        
        expected_current_bar = (now_ms // tf_ms) * tf_ms
        expected_last_closed = expected_current_bar - tf_ms
        
        assert last_closed == expected_last_closed
    
    def test_floor_closed_bar_all_timeframes(self):
        """Test all supported timeframes."""
        now_ms = 1700000000000  # arbitrary timestamp
        
        for tf_name, tf_ms in TIMEFRAME_TO_MS.items():
            last_closed = floor_closed_bar(now_ms, tf_ms)
            
            # Verify last_closed is before now
            assert last_closed < now_ms
            
            # Verify last_closed is aligned to tf_ms
            assert last_closed % tf_ms == 0
            
            # Verify it's exactly one bar before current bar
            current_bar = (now_ms // tf_ms) * tf_ms
            assert last_closed == current_bar - tf_ms
