"""Timestamp normalization and validation utilities."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Union


def normalize_timestamp(ts: Union[int, float]) -> int:
    """
    Normalize timestamp to milliseconds (auto-detects seconds vs milliseconds).

    Args:
        ts: Timestamp value (seconds or milliseconds)

    Returns:
        Timestamp in milliseconds

    Raises:
        ValueError: If timestamp is invalid (0, negative, NaN, or out of reasonable range)
    """
    if ts is None or math.isnan(ts):
        raise ValueError(f"Invalid timestamp: {ts} (NaN or None)")

    ts_float = float(ts)

    if ts_float == 0:
        raise ValueError("Invalid timestamp: 0 (timestamps cannot be zero)")

    if ts_float < 0:
        raise ValueError(f"Invalid timestamp: {ts_float} (negative values not allowed)")

    if math.isnan(ts_float):
        raise ValueError(f"Invalid timestamp: {ts_float} (NaN)")

    # Auto-detect: timestamps in seconds are typically < 1e11
    # while milliseconds are >= 1e12 (for dates after 2001)
    # Reasonable range: 2020-01-01 to 2030-01-01
    MIN_TS_SEC = int(datetime(2020, 1, 1).timestamp())  # ~1577836800
    MAX_TS_SEC = int(datetime(2030, 1, 1).timestamp())  # ~1893456000
    MIN_TS_MS = MIN_TS_SEC * 1000  # ~1577836800000
    MAX_TS_MS = MAX_TS_SEC * 1000  # ~1893456000000

    # If value looks like seconds, convert to ms
    if ts_float < 1e11:
        if ts_float < MIN_TS_SEC or ts_float > MAX_TS_SEC:
            raise ValueError(f"Invalid timestamp (seconds): {ts_float} out of reasonable range")
        return int(ts_float * 1000)

    # Otherwise treat as milliseconds
    if ts_float < MIN_TS_MS or ts_float > MAX_TS_MS:
        raise ValueError(f"Invalid timestamp (milliseconds): {ts_float} out of reasonable range")

    return int(ts_float)


def validate_timestamps_monotonic(timestamps: list[int]) -> bool:
    """
    Validate that timestamps are strictly increasing.

    Args:
        timestamps: List of timestamps

    Raises:
        ValueError: If timestamps are not strictly increasing
    """
    if len(timestamps) < 2:
        return True

    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            raise ValueError(
                f"Non-monotonic timestamps at index {i}: "
                f"{timestamps[i-1]} >= {timestamps[i]}"
            )

    return True


def validate_no_future_timestamps(timestamps: list[int]) -> bool:
    """
    Validate that no timestamps are in the future.

    Args:
        timestamps: List of timestamps (in milliseconds)

    Raises:
        ValueError: If any timestamp is in the future
    """
    if not timestamps:
        return True

    current_ms = int(datetime.utcnow().timestamp() * 1000)
    # Allow 1 minute tolerance for real-world data
    future_tolerance_ms = 60 * 1000

    for ts in timestamps:
        if ts > current_ms + future_tolerance_ms:
            raise ValueError(
                f"Future timestamp detected: {ts} "
                f"(current: {current_ms}, tolerance: {future_tolerance_ms}ms)"
            )

    return True
