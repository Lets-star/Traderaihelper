"""Tests for auto-refresh and new-bar detection."""

import pytest
from indicator_collector.trading_system.auto_refresh import (
    get_refresh_interval_ms,
    calculate_next_bar_close_ms,
    has_new_closed_bar,
    get_last_closed_bar_ts,
    format_eta_seconds,
    AutoRefreshState,
)


class TestRefreshIntervals:
    """Test refresh interval mapping."""

    def test_get_refresh_interval_1m(self):
        """1m timeframe should have 5s refresh."""
        assert get_refresh_interval_ms("1m") == 5_000

    def test_get_refresh_interval_5m(self):
        """5m timeframe should have 30s refresh."""
        assert get_refresh_interval_ms("5m") == 30_000

    def test_get_refresh_interval_1h(self):
        """1h timeframe should have 2min refresh."""
        assert get_refresh_interval_ms("1h") == 120_000

    def test_get_refresh_interval_3h(self):
        """3h timeframe should have 5min refresh."""
        assert get_refresh_interval_ms("3h") == 300_000

    def test_get_refresh_interval_default(self):
        """Unknown timeframe should default to 60s."""
        # Using an arbitrary timeframe that is not mapped
        assert get_refresh_interval_ms("2h") == 60_000


class TestNextBarCalculation:
    """Test next bar close calculation."""

    def test_calculate_next_bar_close_1h(self):
        """Test next bar close for 1h timeframe."""
        # Current time: 2024-01-01 10:30:00 UTC (1704106200000 ms)
        current_time_ms = 1704106200000
        
        # Should return next bar close: 2024-01-01 12:00:00 UTC
        next_close = calculate_next_bar_close_ms(current_time_ms, "1h")
        expected = 1704110400000  # 12:00:00
        
        assert next_close == expected

    def test_calculate_next_bar_close_15m(self):
        """Test next bar close for 15m timeframe."""
        # Current time: 2024-01-01 10:07:00 UTC
        current_time_ms = 1704104820000
        
        # Current bar: 10:00-10:15, next bar: 10:15-10:30
        # Should return 10:30
        next_close = calculate_next_bar_close_ms(current_time_ms, "15m")
        expected = 1704105000000  # 10:30:00
        
        assert next_close == expected

    def test_calculate_next_bar_close_1d(self):
        """Test next bar close for 1d timeframe."""
        # Current time: 2024-01-01 12:00:00 UTC
        current_time_ms = 1704110400000
        
        # Should return next day's close: 2024-01-03 00:00:00
        next_close = calculate_next_bar_close_ms(current_time_ms, "1d")
        expected = 1704240000000  # 2024-01-03 00:00:00
        
        assert next_close == expected


class TestNewBarDetection:
    """Test new closed bar detection."""

    def test_has_new_closed_bar_true(self):
        """Should detect when a new bar has closed."""
        # Last closed: 2024-01-01 10:00:00 (1h bar)
        last_closed_ts = 1704103200000
        
        # Current time: 2024-01-01 12:00:00 (one full bar later)
        current_time_ms = 1704110400000
        
        # Should detect new bar (11:00-12:00 is now closed)
        assert has_new_closed_bar(last_closed_ts, current_time_ms, "1h") is True

    def test_has_new_closed_bar_false(self):
        """Should not detect new bar when none has closed."""
        # Last closed: 2024-01-01 10:00:00
        last_closed_ts = 1704103200000
        
        # Current time: 2024-01-01 10:30:00 (mid-bar)
        current_time_ms = 1704105000000
        
        # Should not detect new bar yet
        assert has_new_closed_bar(last_closed_ts, current_time_ms, "1h") is False

    def test_has_new_closed_bar_exact_boundary(self):
        """Should detect new bar at exact close time."""
        # Last closed: 2024-01-01 10:00:00
        last_closed_ts = 1704103200000
        
        # Current time: 2024-01-01 12:00:00 (exact close of next bar)
        current_time_ms = 1704110400000
        
        # Should detect new bar
        assert has_new_closed_bar(last_closed_ts, current_time_ms, "1h") is True


class TestLastClosedBarCalculation:
    """Test last closed bar timestamp calculation."""

    def test_get_last_closed_bar_mid_bar(self):
        """Test getting last closed bar during mid-bar."""
        # Current time: 2024-01-01 10:30:00 (mid 10:00-11:00 bar)
        current_time_ms = 1704105000000
        
        # Last closed bar should be 09:00-10:00 (opens at 09:00)
        last_closed = get_last_closed_bar_ts(current_time_ms, "1h")
        expected = 1704099600000  # 09:00:00
        
        assert last_closed == expected

    def test_get_last_closed_bar_at_boundary(self):
        """Test getting last closed bar at bar boundary."""
        # Current time: 2024-01-01 11:00:00 (exact bar open)
        current_time_ms = 1704106800000
        
        # Last closed bar should be 10:00-11:00 (opens at 10:00)
        last_closed = get_last_closed_bar_ts(current_time_ms, "1h")
        expected = 1704103200000  # 10:00:00
        
        assert last_closed == expected


class TestETAFormatting:
    """Test ETA formatting."""

    def test_format_eta_seconds_only(self):
        """Test formatting seconds only."""
        assert format_eta_seconds(45.0) == "45s"

    def test_format_eta_minutes_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_eta_seconds(125.0) == "2m 5s"

    def test_format_eta_hours_minutes(self):
        """Test formatting hours and minutes."""
        assert format_eta_seconds(3665.0) == "1h 1m 5s"

    def test_format_eta_zero(self):
        """Test formatting zero seconds."""
        assert format_eta_seconds(0.0) == "0s"

    def test_format_eta_negative(self):
        """Test formatting negative time."""
        assert format_eta_seconds(-10.0) == "now"


class TestAutoRefreshState:
    """Test AutoRefreshState class."""

    def test_initialization(self):
        """Test state initialization."""
        state = AutoRefreshState("BTCUSDT", "1h")
        assert state.symbol == "BTCUSDT"
        assert state.timeframe == "1h"
        assert state.enabled is False
        assert state.last_closed_ts is None
        assert state.refresh_interval_ms == 120_000

    def test_update_last_closed(self):
        """Test updating last closed timestamp."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.update_last_closed(1704103200000)
        
        assert state.last_closed_ts == 1704103200000
        assert state.last_fetched_at is not None

    def test_should_refresh_disabled(self):
        """Should not refresh when disabled."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.enabled = False
        state.last_closed_ts = 1704103200000
        
        assert state.should_refresh(1704110400000) is False

    def test_should_refresh_first_time(self):
        """Should refresh on first call."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.enabled = True
        
        assert state.should_refresh() is True

    def test_should_refresh_new_bar_available(self):
        """Should refresh when new bar is available."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.enabled = True
        state.last_closed_ts = 1704103200000  # 10:00:00
        
        # Time is now 12:00:00 (new bar available)
        assert state.should_refresh(1704110400000) is True

    def test_should_refresh_no_new_bar(self):
        """Should not refresh when no new bar."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.enabled = True
        state.last_closed_ts = 1704103200000  # 10:00:00
        
        # Time is now 10:30:00 (mid-bar, no new bar yet)
        assert state.should_refresh(1704105000000) is False

    def test_get_next_update_eta(self):
        """Test getting next update ETA."""
        state = AutoRefreshState("BTCUSDT", "1h")
        state.last_closed_ts = 1704103200000  # 10:00:00
        
        # Current time: 10:30:00 (30 minutes into next bar)
        # Next bar closes at 12:00:00, so 90 minutes remaining
        current_time = 1704105000000
        seconds, formatted = state.get_next_update_eta(current_time)
        
        assert seconds == 5400  # 90 minutes = 5400 seconds
        assert "1h 30m" in formatted

    def test_get_state_key(self):
        """Test state key generation."""
        state = AutoRefreshState("BTCUSDT", "1h")
        key = state.get_state_key()
        
        assert key == "autorefresh_BTCUSDT_1h"


class TestEdgeCases:
    """Test edge cases."""

    def test_multiple_bars_passed(self):
        """Should detect new bar even if multiple bars have passed."""
        # Last closed: 2024-01-01 10:00:00
        last_closed_ts = 1704103200000
        
        # Current time: 2024-01-01 15:00:00 (5 hours later)
        current_time_ms = 1704121200000
        
        # Should detect new bars
        assert has_new_closed_bar(last_closed_ts, current_time_ms, "1h") is True

    def test_different_timeframes(self):
        """Test with different timeframes."""
        # 5m timeframe
        last_closed_ts = 1704103200000  # 10:00:00
        current_time_ms = 1704103800000  # 10:10:00 (2 bars later)
        
        assert has_new_closed_bar(last_closed_ts, current_time_ms, "5m") is True
