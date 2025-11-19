"""Auto-refresh and new-bar detection utilities for timeframe-based analysis."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from ..timeframes import Timeframe


# Refresh intervals in milliseconds for each timeframe
TIMEFRAME_REFRESH_INTERVALS_MS: Dict[str, int] = {
    "1m": 5_000,       # 5 seconds
    "5m": 30_000,      # 30 seconds
    "15m": 60_000,     # 60 seconds
    "1h": 120_000,     # 2 minutes
    "3h": 300_000,     # 5 minutes
    "4h": 300_000,     # 5 minutes
    "1d": 600_000,     # 10 minutes
}


def get_refresh_interval_ms(timeframe: str) -> int:
    """
    Get the refresh interval in milliseconds for the given timeframe.

    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "3h", "4h", "1d")

    Returns:
        Refresh interval in milliseconds
    """
    tf = Timeframe.from_value(timeframe)
    return TIMEFRAME_REFRESH_INTERVALS_MS.get(tf.value, 60_000)


def calculate_next_bar_close_ms(current_time_ms: int, timeframe: str) -> int:
    """
    Calculate the timestamp of the next closed bar for the given timeframe.

    Args:
        current_time_ms: Current time in milliseconds
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")

    Returns:
        Timestamp of the next bar close in milliseconds
    """
    tf = Timeframe.from_value(timeframe)
    interval_ms = tf.to_milliseconds()
    if interval_ms <= 0:
        return current_time_ms + 60_000  # Default to 1 minute

    # Floor to current bar start
    current_bar_start = (current_time_ms // interval_ms) * interval_ms
    # Next bar close is current bar start + 2 * interval
    return current_bar_start + (2 * interval_ms)


def has_new_closed_bar(
    last_closed_ts: int,
    current_server_time_ms: int,
    timeframe: str,
) -> bool:
    """
    Check if a new closed bar is available since the last check.

    Args:
        last_closed_ts: Timestamp of the last closed bar we processed (in ms)
        current_server_time_ms: Current server time in milliseconds
        timeframe: Timeframe string

    Returns:
        True if a new closed bar is available, False otherwise
    """
    tf = Timeframe.from_value(timeframe)
    interval_ms = tf.to_milliseconds()
    if interval_ms <= 0:
        return False

    # Calculate expected next bar open time
    next_bar_open = last_closed_ts + interval_ms

    # Check if we've crossed into the next bar's close time
    next_bar_close = next_bar_open + interval_ms
    return current_server_time_ms >= next_bar_close


def get_last_closed_bar_ts(current_server_time_ms: int, timeframe: str) -> int:
    """
    Get the timestamp of the last closed bar given the current server time.

    Args:
        current_server_time_ms: Current server time in milliseconds
        timeframe: Timeframe string

    Returns:
        Timestamp of the last closed bar in milliseconds
    """
    tf = Timeframe.from_value(timeframe)
    interval_ms = tf.to_milliseconds()
    if interval_ms <= 0:
        return current_server_time_ms

    # Floor to current bar start
    current_bar_start = (current_server_time_ms // interval_ms) * interval_ms
    # Last closed bar is the bar before the current one
    return current_bar_start - interval_ms


def format_eta_seconds(seconds_remaining: float) -> str:
    """
    Format remaining time as a human-readable string.

    Args:
        seconds_remaining: Seconds until next event

    Returns:
        Formatted string (e.g., "5s", "2m 30s", "1h 5m")
    """
    if seconds_remaining < 0:
        return "now"

    hours = int(seconds_remaining // 3600)
    minutes = int((seconds_remaining % 3600) // 60)
    seconds = int(seconds_remaining % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def get_server_time_ms() -> int:
    """Get current server time in milliseconds (UTC)."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class AutoRefreshState:
    """Manage auto-refresh state for a specific symbol and timeframe."""

    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize auto-refresh state.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.enabled = False
        self.last_closed_ts: Optional[int] = None
        self.last_fetched_at: Optional[datetime] = None
        self.refresh_interval_ms = get_refresh_interval_ms(timeframe)

    def update_last_closed(self, last_closed_ts: int) -> None:
        """Update the last closed bar timestamp."""
        self.last_closed_ts = last_closed_ts
        self.last_fetched_at = datetime.now(timezone.utc)

    def should_refresh(self, current_server_time_ms: Optional[int] = None) -> bool:
        """
        Check if a refresh should be triggered.

        Args:
            current_server_time_ms: Optional current server time (uses system time if None)

        Returns:
            True if a refresh should be triggered
        """
        if not self.enabled:
            return False

        if self.last_closed_ts is None:
            return True  # First time, always refresh

        server_time = current_server_time_ms or get_server_time_ms()
        return has_new_closed_bar(self.last_closed_ts, server_time, self.timeframe)

    def get_next_update_eta(self, current_server_time_ms: Optional[int] = None) -> Tuple[int, str]:
        """
        Get the ETA for the next update.

        Args:
            current_server_time_ms: Optional current server time

        Returns:
            Tuple of (seconds_remaining, formatted_string)
        """
        if self.last_closed_ts is None:
            return (0, "now")

        server_time = current_server_time_ms or get_server_time_ms()
        next_close_ms = calculate_next_bar_close_ms(self.last_closed_ts, self.timeframe)
        remaining_ms = max(next_close_ms - server_time, 0)
        remaining_seconds = remaining_ms / 1000.0

        return (int(remaining_seconds), format_eta_seconds(remaining_seconds))

    def get_state_key(self) -> str:
        """Get a unique key for session state storage."""
        return f"autorefresh_{self.symbol}_{self.timeframe}"
