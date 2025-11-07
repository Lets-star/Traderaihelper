"""Binance API integration for real historical OHLCV data."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from ...timeframes import Timeframe
from .interfaces import HistoricalDataSource
from .timestamp_utils import (
    normalize_timestamp,
    validate_no_future_timestamps,
    validate_timestamps_monotonic,
)

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect rate limits
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # Exponential backoff multiplier


class BinanceKlinesSource(HistoricalDataSource):
    """Load historical OHLCV candles from Binance API."""

    # Mapping of internal timeframes to Binance intervals
    TIMEFRAME_TO_BINANCE_INTERVAL: Dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "3h": "1h",  # 3h will be aggregated from 1h
    }

    # Max candles per Binance API request
    MAX_CANDLES_PER_REQUEST = 1000

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit_delay: float = BINANCE_RATE_LIMIT_DELAY,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize Binance data source.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

    def load_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV candles from Binance.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe for candles
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
            All timestamps are in UTC milliseconds.

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        # Normalize inputs
        tf = Timeframe.from_value(timeframe)
        symbol = symbol.upper().strip()

        # Determine if 3h aggregation is needed
        is_3h = tf.value == "3h"
        source_timeframe = "1h" if is_3h else tf.value
        binance_interval = self.TIMEFRAME_TO_BINANCE_INTERVAL[source_timeframe]

        try:
            # Fetch raw candles
            raw_candles = self._fetch_candles_paginated(
                symbol, binance_interval, start, end
            )

            if not raw_candles:
                raise ValueError(
                    f"No data available for {symbol} {timeframe} from {start} to {end}"
                )

            # Convert to DataFrame
            df = self._candles_to_dataframe(raw_candles)

            # Apply 3h aggregation if needed
            if is_3h:
                df = self._aggregate_to_3h(df)

            # Validate and normalize
            df = self._validate_and_normalize(df, tf)

            return df

        except Exception as e:
            logger.error(f"Failed to load candles from Binance: {e}")
            raise ValueError(f"Failed to load {symbol} {timeframe} data: {e}") from e

    def _fetch_candles_paginated(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[list]:
        """
        Fetch candles with pagination to handle large date ranges.

        Args:
            symbol: Trading symbol
            interval: Binance interval (1m, 5m, 15m, 1h, 4h, 1d)
            start: Start datetime
            end: End datetime

        Returns:
            List of candle data (each is a list from Binance API)

        Raises:
            ValueError: If API requests fail after retries
        """
        all_candles = []
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        current_start_ms = start_ms

        while current_start_ms < end_ms:
            try:
                # Fetch batch of candles
                candles = self._fetch_klines_batch(symbol, interval, current_start_ms)

                if not candles:
                    break  # No more data available

                all_candles.extend(candles)

                # Update start for next batch
                last_candle_time = candles[-1][0]
                current_start_ms = last_candle_time + 1  # Start after last candle

                # Respect rate limits
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching batch starting at {current_start_ms}: {e}")
                raise

        return all_candles

    def _fetch_klines_batch(self, symbol: str, interval: str, start_ms: int) -> list[list]:
        """
        Fetch a single batch of klines with retry logic.

        Args:
            symbol: Trading symbol
            interval: Binance interval
            start_ms: Start time in milliseconds

        Returns:
            List of candle data from Binance API

        Raises:
            ValueError: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                url = (
                    f"{BINANCE_BASE_URL}?"
                    f"symbol={symbol}&"
                    f"interval={interval}&"
                    f"startTime={start_ms}&"
                    f"limit={self.MAX_CANDLES_PER_REQUEST}"
                )

                with urlopen(url) as response:
                    raw_data = response.read()

                data = json.loads(raw_data)
                return data

            except HTTPError as e:
                if e.code == 429:  # Rate limited
                    backoff_time = RETRY_BACKOFF ** attempt
                    logger.warning(
                        f"Rate limited on attempt {attempt + 1}, backing off {backoff_time}s"
                    )
                    time.sleep(backoff_time)
                else:
                    raise ValueError(f"HTTP error {e.code}: {e.reason}") from e

            except URLError as e:
                if attempt < self.max_retries - 1:
                    backoff_time = RETRY_BACKOFF ** attempt
                    logger.warning(
                        f"Network error on attempt {attempt + 1}, backing off {backoff_time}s"
                    )
                    time.sleep(backoff_time)
                else:
                    raise ValueError(f"Network error: {e.reason}") from e

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to decode Binance response: {e}") from e

        raise ValueError(f"Failed to fetch klines after {self.max_retries} retries")

    def _candles_to_dataframe(self, candles: list[list]) -> pd.DataFrame:
        """
        Convert Binance candle data to DataFrame.

        Args:
            candles: List of candle data from Binance API
                Each candle is: [openTime, open, high, low, close, volume, ...]

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
        """
        df_data = {
            "ts": [int(c[0]) for c in candles],
            "open": [float(c[1]) for c in candles],
            "high": [float(c[2]) for c in candles],
            "low": [float(c[3]) for c in candles],
            "close": [float(c[4]) for c in candles],
            "volume": [float(c[5]) for c in candles],  # Volume at index 5
        }

        df = pd.DataFrame(df_data)
        return df

    def _aggregate_to_3h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1h candles to 3h candles.

        Args:
            df: DataFrame with 1h candles

        Returns:
            DataFrame with 3h candles (aligned to 00:00, 03:00, 06:00, etc. UTC)
        """
        if df.empty:
            return df

        # Convert timestamp to datetime for grouping
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

        # Calculate 3h bucket start time (aligned to 00:00, 03:00, 06:00, 09:00, ...)
        # Get the hour from UTC datetime
        df["hour"] = df["datetime"].dt.hour
        # Calculate which 3h bucket this hour belongs to (0-2->0, 3-5->3, 6-8->6, ...)
        df["bucket_hour"] = (df["hour"] // 3) * 3
        # Create bucket date (date at start of 3h period)
        df["bucket_date"] = df["datetime"].dt.normalize()
        # Create bucket start time
        df["bucket_start"] = (
            df["bucket_date"] + pd.to_timedelta(df["bucket_hour"], unit="h")
        )
        df["bucket_start_ms"] = (df["bucket_start"].astype(int) // 1e6).astype(int)

        # Group by 3h bucket and aggregate
        aggregated = df.groupby("bucket_start_ms", as_index=False).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Use bucket start time as ts
        aggregated["ts"] = aggregated["bucket_start_ms"]

        # Return only the required columns
        result = aggregated[["ts", "open", "high", "low", "close", "volume"]].copy()

        return result

    def _validate_and_normalize(self, df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """
        Validate and normalize the candle DataFrame.

        Args:
            df: DataFrame with candle data
            timeframe: Expected timeframe

        Returns:
            Validated and normalized DataFrame

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Empty dataframe")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Validate columns exist
        required_cols = ["ts", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected {required_cols}, got {list(df.columns)}")

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values
        if df[required_cols].isna().any().any():
            raise ValueError("NaN values found in candle data")

        # Normalize timestamps
        try:
            df["ts"] = df["ts"].apply(normalize_timestamp)
        except Exception as e:
            raise ValueError(f"Failed to normalize timestamps: {e}") from e

        # Validate monotonicity
        try:
            validate_timestamps_monotonic(df["ts"].tolist())
        except Exception as e:
            raise ValueError(f"Timestamps not monotonic: {e}") from e

        # Validate no future timestamps
        try:
            validate_no_future_timestamps(df["ts"].tolist())
        except Exception as e:
            raise ValueError(f"Future timestamps detected: {e}") from e

        # Validate no zero prices (before OHLC relationships)
        if (df[["open", "high", "low", "close"]] == 0).any().any():
            raise ValueError("Zero prices detected in OHLC data")

        # Validate positive volume
        if (df["volume"] < 0).any():
            raise ValueError("Negative volume detected")

        # Validate OHLC relationships
        if not (df["low"] <= df["open"]).all() or not (df["open"] <= df["high"]).all():
            raise ValueError("OHLC data violates low <= open <= high")

        if not (df["low"] <= df["close"]).all() or not (df["close"] <= df["high"]).all():
            raise ValueError("OHLC data violates low <= close <= high")

        # Reset index and return
        df = df.reset_index(drop=True)
        return df
