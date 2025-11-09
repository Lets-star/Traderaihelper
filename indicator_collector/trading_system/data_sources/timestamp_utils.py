"""Timestamp normalization and validation utilities."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Union

import pandas as pd

from ...timeframes import Timeframe


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


def ensure_utc_datetime(value: datetime) -> datetime:
    """Ensure a datetime value is timezone-aware in UTC."""
    if not isinstance(value, datetime):
        raise TypeError("Datetime value expected")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def datetime_to_milliseconds(value: datetime) -> int:
    """Convert a datetime to UTC milliseconds since epoch."""
    utc_value = ensure_utc_datetime(value)
    return int(round(utc_value.timestamp() * 1000))


def floor_to_interval(ts_ms: int, interval_ms: int) -> int:
    """Floor a timestamp in milliseconds to the nearest interval boundary."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (int(ts_ms) // interval_ms) * interval_ms


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


def validate_no_future_timestamps(
    timestamps: list[int],
    tolerance_ms: int = 60 * 1000,
    reference_ms: Optional[int] = None,
) -> bool:
    """
    Validate that no timestamps are in the future.

    Args:
        timestamps: List of timestamps (in milliseconds)
        tolerance_ms: Allowed future tolerance in milliseconds
        reference_ms: Optional reference timestamp (ms) to compare against.
            Defaults to current UTC time when not provided.

    Raises:
        ValueError: If any timestamp is in the future beyond the tolerance
    """
    if not timestamps:
        return True

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    if reference_ms is None:
        current_ms = int(datetime.utcnow().timestamp() * 1000)
    else:
        current_ms = int(reference_ms)

    for ts in timestamps:
        if ts > current_ms + tolerance_ms:
            raise ValueError(
                f"Future timestamp detected: {ts} "
                f"(reference: {current_ms}, tolerance: {tolerance_ms}ms)"
            )

    return True


def get_last_closed_candle_ts(
    candles_df: "pd.DataFrame",
    timeframe: Union[str, Timeframe],
    tolerance_ms: int = 60 * 1000,
) -> int:
    """
    Determine the last fully closed candle timestamp for the given timeframe.

    Args:
        candles_df: DataFrame containing candle data with a 'ts' column representing open times.
        timeframe: Candle timeframe (e.g. "1h", "3h", "15m").
        tolerance_ms: Allowed future tolerance when validating the closed timestamp.

    Returns:
        Timestamp in milliseconds representing the last closed candle.

    Raises:
        ValueError: If no candles are available or all candles are in the future.
    """
    if candles_df is None or candles_df.empty:
        raise ValueError("Cannot determine last closed candle timestamp: no candles provided")

    if "ts" not in candles_df.columns:
        raise ValueError("Candles dataframe must contain 'ts' column")

    df = candles_df.sort_values("ts")
    tf = Timeframe.from_value(timeframe)
    interval_ms = tf.to_milliseconds()

    last_close_ts: Optional[int] = None
    last_error: Optional[Exception] = None

    for raw_ts in reversed(df["ts"].tolist()):
        normalized_open = normalize_timestamp(raw_ts)
        bucket_index = normalized_open // interval_ms
        close_ts = (bucket_index + 1) * interval_ms
        last_close_ts = close_ts

        try:
            validate_no_future_timestamps([close_ts], tolerance_ms=tolerance_ms)
        except ValueError as exc:
            last_error = exc
            continue

        return close_ts

    if last_close_ts is None:
        raise ValueError(f"No candles provided for timeframe {tf.value}")

    message = (
        f"No closed candles available for timeframe {tf.value}; "
        f"latest candle close ts={last_close_ts}"
    )
    if last_error:
        raise ValueError(message) from last_error
    raise ValueError(message)
