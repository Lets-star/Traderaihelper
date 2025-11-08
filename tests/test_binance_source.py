"""Tests for Binance historical data source."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import pandas as pd

from indicator_collector.trading_system.data_sources import (
    BinanceKlinesSource,
    normalize_timestamp,
    validate_timestamps_monotonic,
    validate_no_future_timestamps,
)
from indicator_collector.timeframes import Timeframe


class TestTimestampUtils:
    """Test timestamp normalization and validation utilities."""

    def test_normalize_timestamp_milliseconds(self):
        """Test normalizing milliseconds timestamp."""
        ts_ms = 1704067200000  # 2024-01-01 00:00:00 UTC
        result = normalize_timestamp(ts_ms)
        assert result == ts_ms
        assert isinstance(result, int)

    def test_normalize_timestamp_seconds(self):
        """Test normalizing seconds timestamp to milliseconds."""
        ts_sec = 1704067200  # 2024-01-01 00:00:00 UTC
        result = normalize_timestamp(ts_sec)
        assert result == ts_sec * 1000
        assert isinstance(result, int)

    def test_normalize_timestamp_zero_raises(self):
        """Test that zero timestamp raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timestamp: 0"):
            normalize_timestamp(0)

    def test_normalize_timestamp_negative_raises(self):
        """Test that negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            normalize_timestamp(-1000)

    def test_normalize_timestamp_nan_raises(self):
        """Test that NaN timestamp raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            normalize_timestamp(float("nan"))

    def test_normalize_timestamp_out_of_range_raises(self):
        """Test that timestamps outside reasonable range raise ValueError."""
        # Way too old (before 2020)
        with pytest.raises(ValueError, match="out of reasonable range"):
            normalize_timestamp(1000000000)  # Very old timestamp

    def test_validate_timestamps_monotonic(self):
        """Test validating monotonic timestamps."""
        timestamps = [1704067200000, 1704067300000, 1704067400000]
        assert validate_timestamps_monotonic(timestamps) is True

    def test_validate_timestamps_monotonic_fails(self):
        """Test that non-monotonic timestamps raise ValueError."""
        timestamps = [1704067200000, 1704067100000, 1704067400000]  # Descending
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_timestamps_monotonic(timestamps)

    def test_validate_timestamps_monotonic_duplicates(self):
        """Test that duplicate timestamps raise ValueError."""
        timestamps = [1704067200000, 1704067200000, 1704067400000]
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_timestamps_monotonic(timestamps)

    def test_validate_no_future_timestamps(self):
        """Test validating no future timestamps."""
        # Current time - 1 hour
        past_ts = int((datetime.utcnow().timestamp() - 3600) * 1000)
        assert validate_no_future_timestamps([past_ts]) is True

    def test_validate_future_timestamps_raises(self):
        """Test that future timestamps raise ValueError."""
        # Current time + 2 minutes (outside 1 min tolerance)
        future_ts = int((datetime.utcnow().timestamp() + 120) * 1000)
        with pytest.raises(ValueError, match="Future timestamp"):
            validate_no_future_timestamps([future_ts])


class TestBinanceKlinesSourceDataConversion:
    """Test data conversion in BinanceKlinesSource."""

    def test_candles_to_dataframe(self):
        """Test converting Binance candle data to DataFrame."""
        source = BinanceKlinesSource()

        base_time = 1704067200000  # 2024-01-01
        candles_data = [
            [base_time, "50000.0", "50100.0", "49900.0", "50050.0", "100.0", base_time + 3600000, "10000.0"],
            [base_time + 3600000, "50050.0", "50150.0", "49950.0", "50100.0", "110.0", base_time + 7200000, "11000.0"],
        ]

        df = source._candles_to_dataframe(candles_data)

        assert len(df) == 2
        assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
        assert df["ts"].iloc[0] == base_time
        assert df["open"].iloc[0] == 50000.0
        assert df["volume"].iloc[0] == 100.0  # Index 5 = volume

    def test_aggregate_to_3h(self):
        """Test aggregating 1h candles to 3h."""
        source = BinanceKlinesSource()

        # Create 6 1h candles starting at midnight UTC
        base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
        hour_ms = 60 * 60 * 1000

        df_data = {
            "ts": [base_time + i * hour_ms for i in range(6)],
            "open": [50000.0 + i * 10 for i in range(6)],
            "high": [50050.0 + i * 10 for i in range(6)],
            "low": [49950.0 + i * 10 for i in range(6)],
            "close": [50025.0 + i * 10 for i in range(6)],
            "volume": [100.0 + i * 5 for i in range(6)],
        }
        df = pd.DataFrame(df_data)

        aggregated = source._aggregate_to_3h(df)

        # Should have 2 3h candles (0-2h and 3-5h)
        assert len(aggregated) == 2

        # First 3h candle (0:00-3:00)
        assert aggregated.iloc[0]["ts"] == base_time
        assert aggregated.iloc[0]["open"] == 50000.0  # First candle's open
        assert aggregated.iloc[0]["close"] == 50045.0  # Last of first 3 candles' close
        assert aggregated.iloc[0]["high"] == 50070.0  # Max high
        assert aggregated.iloc[0]["low"] == 49950.0  # Min low
        assert aggregated.iloc[0]["volume"] == 315.0  # Sum of first 3 volumes: 100+105+110

    def test_aggregate_to_3h_alignment(self):
        """Test that 3h aggregation is aligned to UTC hour boundaries."""
        source = BinanceKlinesSource()

        # Create 3h period starting at 00:00
        base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
        hour_ms = 60 * 60 * 1000

        df_data = {
            "ts": [base_time + i * hour_ms for i in range(3)],
            "open": [100.0] * 3,
            "high": [100.0] * 3,
            "low": [100.0] * 3,
            "close": [100.0] * 3,
            "volume": [100.0] * 3,
        }
        df = pd.DataFrame(df_data)

        aggregated = source._aggregate_to_3h(df)

        assert len(aggregated) == 1
        # Should be aligned to start of UTC day
        dt = pd.to_datetime(aggregated.iloc[0]["ts"], unit="ms", utc=True)
        assert dt.hour in [0, 3, 6, 9, 12, 15, 18, 21]


class TestBinanceKlinesSourceValidation:
    """Test data validation in BinanceKlinesSource."""

    def test_validate_and_normalize_valid_data(self):
        """Test validation passes for valid data."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        df_data = {
            "ts": [base_time, base_time + 3600000],
            "open": [50000.0, 50050.0],
            "high": [50100.0, 50150.0],
            "low": [49900.0, 49950.0],
            "close": [50050.0, 50100.0],
            "volume": [100.0, 110.0],
        }
        df = pd.DataFrame(df_data)

        result = source._validate_and_normalize(df, Timeframe.H1)

        assert len(result) == 2
        assert not result.isna().any().any()

    def test_validate_and_normalize_missing_columns_raises(self):
        """Test validation fails for missing columns."""
        source = BinanceKlinesSource()

        df = pd.DataFrame({
            "ts": [1704067200000],
            "open": [50000.0],
            # Missing other OHLC columns
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            source._validate_and_normalize(df, Timeframe.H1)

    def test_validate_and_normalize_nan_raises(self):
        """Test validation fails for NaN values."""
        source = BinanceKlinesSource()

        df = pd.DataFrame({
            "ts": [1704067200000, float("nan")],
            "open": [50000.0, 50050.0],
            "high": [50100.0, 50150.0],
            "low": [49900.0, 49950.0],
            "close": [50050.0, 50100.0],
            "volume": [100.0, 110.0],
        })

        with pytest.raises(ValueError, match="NaN"):
            source._validate_and_normalize(df, Timeframe.H1)

    def test_validate_and_normalize_zero_price_raises(self):
        """Test validation fails for zero prices."""
        source = BinanceKlinesSource()

        df = pd.DataFrame({
            "ts": [1704067200000],
            "open": [0.0],  # Zero price
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
        })

        with pytest.raises(ValueError, match="Zero prices"):
            source._validate_and_normalize(df, Timeframe.H1)

    def test_validate_and_normalize_negative_volume_raises(self):
        """Test validation fails for negative volume."""
        source = BinanceKlinesSource()

        df = pd.DataFrame({
            "ts": [1704067200000],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [-100.0],  # Negative volume
        })

        with pytest.raises(ValueError, match="Negative volume"):
            source._validate_and_normalize(df, Timeframe.H1)

    def test_validate_and_normalize_ohlc_violation_raises(self):
        """Test validation fails for OHLC relationship violations."""
        source = BinanceKlinesSource()

        df = pd.DataFrame({
            "ts": [1704067200000],
            "open": [50000.0],
            "high": [49900.0],  # High < Open (violation)
            "low": [49800.0],
            "close": [50050.0],
            "volume": [100.0],
        })

        with pytest.raises(ValueError, match="OHLC.*violates"):
            source._validate_and_normalize(df, Timeframe.H1)


class TestBinanceKlinesSourceFetching:
    """Test fetching logic in BinanceKlinesSource."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.urlopen")
    def test_fetch_klines_batch_success(self, mock_urlopen):
        """Test successful klines batch fetch."""
        source = BinanceKlinesSource()

        # Mock response
        response_data = [
            [1704067200000, "50000", "50100", "49900", "50050", "100", 1704070800000, "10000"],
            [1704070800000, "50050", "50150", "49950", "50100", "110", 1704074400000, "11000"],
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)

        assert len(result) == 2
        assert result[0][0] == 1704067200000

    @patch("indicator_collector.trading_system.data_sources.binance_source.urlopen")
    @patch("indicator_collector.trading_system.data_sources.binance_source.time.sleep")
    def test_fetch_klines_batch_rate_limit_retry(self, mock_sleep, mock_urlopen):
        """Test retry on rate limit (429)."""
        source = BinanceKlinesSource(max_retries=2)

        from urllib.error import HTTPError

        # Fail first, then succeed
        mock_urlopen.side_effect = [
            HTTPError("", 429, "Too Many Requests", {}, None),
            MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"[[]]")))),
        ]

        result = source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)

        assert result == [[]]
        assert mock_sleep.called  # Should have backed off

    @patch("indicator_collector.trading_system.data_sources.binance_source.urlopen")
    def test_fetch_klines_batch_max_retries_exceeded(self, mock_urlopen):
        """Test failure after max retries."""
        source = BinanceKlinesSource(max_retries=1)

        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("Connection failed")

        with pytest.raises(ValueError, match="Failed to fetch klines"):
            source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)


class TestBinanceKlinesSourceIntegration:
    """Integration tests for BinanceKlinesSource."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_klines_batch")
    def test_load_candles_1h(self, mock_fetch):
        """Test loading 1h candles."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        candles_data = [
            [base_time + i * 3600000, "50000", "50100", "49900", "50050", "100", base_time + (i + 1) * 3600000, "10000"]
            for i in range(24)  # 24 hours of data
        ]

        mock_fetch.return_value = candles_data

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)

        df = source.load_candles("BTCUSDT", "1h", start, end)

        assert len(df) >= 24  # May have more from pagination
        assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
        assert not df.isna().any().any()
        meta = df.attrs.get("meta")
        assert meta is not None
        assert meta["source"] == "binance"
        assert meta["symbol"] == "BTCUSDT"
        assert meta["timeframe"] == "1h"
        assert meta["real_data"] is True
        assert meta["source_timeframe"] == "1h"

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_klines_batch")
    def test_load_candles_3h(self, mock_fetch):
        """Test loading 3h candles (aggregated from 1h)."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        # Return 72 hours of 1h candles (should aggregate to 24 3h candles)
        candles_data = [
            [base_time + i * 3600000, "50000", "50100", "49900", "50050", "100", base_time + (i + 1) * 3600000, "10000"]
            for i in range(72)
        ]

        mock_fetch.return_value = candles_data

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 4, 0, 0, 0)

        df = source.load_candles("BTCUSDT", "3h", start, end)

        # 72 1h candles should aggregate to 24 3h candles
        assert len(df) >= 24  # May have more from pagination
        assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
        meta = df.attrs.get("meta")
        assert meta is not None
        assert meta["source"] == "binance"
        assert meta["symbol"] == "BTCUSDT"
        assert meta["timeframe"] == "3h"
        assert meta["real_data"] is True
        assert meta["source_timeframe"] == "1h"
        aggregation = meta.get("aggregation")
        assert aggregation is not None
        assert aggregation["from_timeframe"] == "1h"
        assert aggregation["factor"] == 3

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_klines_batch")
    def test_load_candles_empty_raises(self, mock_fetch):
        """Test that empty response raises ValueError."""
        source = BinanceKlinesSource()

        mock_fetch.return_value = []

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 2, 0, 0, 0)

        with pytest.raises(ValueError, match="No data available"):
            source.load_candles("BTCUSDT", "1h", start, end)

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_klines_batch")
    def test_load_candles_pagination(self, mock_fetch):
        """Test pagination for large date ranges."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        hour_ms = 3600000

        # Return data in chunks (simulating pagination)
        def fetch_side_effect(symbol, interval, start_ms):
            # Return 1000 candles per request
            count = 1000
            return [
                [start_ms + i * hour_ms, "50000", "50100", "49900", "50050", "100", start_ms + (i + 1) * hour_ms, "10000"]
                for i in range(count)
            ]

        mock_fetch.side_effect = fetch_side_effect

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 2, 1, 0, 0, 0)  # 1 month = ~720 hours

        df = source.load_candles("BTCUSDT", "1h", start, end)

        # Should have fetched multiple pages
        assert len(df) > 0
        assert mock_fetch.call_count > 1
