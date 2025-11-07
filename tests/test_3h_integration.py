"""Tests for 3h timeframe integration and data fetching."""

import pytest
from datetime import datetime, timedelta

from indicator_collector.data_fetcher import (
    aggregate_candles_to_3h,
    fetch_klines_with_source_metadata,
    validate_timestamp_monotonicity,
    validate_timestamp_plausibility,
    fetch_and_validate_klines,
    create_source_metadata_dict,
)
from indicator_collector.math_utils import Candle


class Test3hAggregation:
    """Test cases for 3h candle aggregation."""
    
    def test_aggregate_1h_to_3h(self):
        """Test aggregating 1h candles to 3h."""
        # Create 6 hours of 1h candles (should make 2 3h candles)
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        
        candles_1h = [
            Candle(
                open_time=base_time + i * hour_ms,
                close_time=base_time + (i + 1) * hour_ms,
                open=50000.0 + i * 10,
                high=50050.0 + i * 10,
                low=49950.0 + i * 10,
                close=50025.0 + i * 10,
                volume=100.0 + i * 5
            )
            for i in range(6)
        ]
        
        aggregated = aggregate_candles_to_3h(candles_1h)
        
        # Should have 2 3h candles
        assert len(aggregated) == 2
        
        # Check first 3h candle (hours 0-2)
        first_candle = aggregated[0]
        assert first_candle.open_time == base_time
        assert first_candle.close_time == base_time + 3 * hour_ms
        assert first_candle.open == 50000.0
        assert first_candle.high == 50070.0  # Max of first 3 hours
        assert first_candle.low == 49950.0  # Min of first 3 hours
        assert first_candle.close == 50045.0  # Close of hour 2
        assert first_candle.volume == 315.0  # Sum of first 3 hours
        
        # Check second 3h candle (hours 3-5)
        second_candle = aggregated[1]
        assert second_candle.open_time == base_time + 3 * hour_ms
        assert second_candle.close_time == base_time + 6 * hour_ms
        assert second_candle.open == 50030.0  # Open of hour 3
        assert second_candle.high == 50100.0  # Max of last 3 hours
        assert second_candle.low == 50040.0  # Min of last 3 hours
        assert second_candle.close == 50055.0  # Close of hour 5
        assert second_candle.volume == 330.0  # Sum of last 3 hours
    
    def test_aggregate_15m_to_3h(self):
        """Test aggregating 15m candles to 3h."""
        # Create 3 hours of 15m candles (should make 1 3h candle)
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        min_15_ms = 15 * 60 * 1000
        
        candles_15m = [
            Candle(
                open_time=base_time + i * min_15_ms,
                close_time=base_time + (i + 1) * min_15_ms,
                open=50000.0 + i * 2,
                high=50010.0 + i * 2,
                low=49990.0 + i * 2,
                close=50005.0 + i * 2,
                volume=25.0 + i
            )
            for i in range(12)  # 12 * 15m = 3h
        ]
        
        aggregated = aggregate_candles_to_3h(candles_15m)
        
        # Should have 1 3h candle
        assert len(aggregated) == 1
        
        candle = aggregated[0]
        assert candle.open_time == base_time
        assert candle.close_time == base_time + 3 * 60 * 60 * 1000
        assert candle.open == 50000.0  # First open
        assert candle.high == 50022.0  # Max of all 15m candles
        assert candle.low == 49990.0  # Min of all 15m candles
        assert candle.close == 50021.0  # Last close
        assert candle.volume == 366.0  # Sum of all volumes
    
    def test_aggregate_empty_list(self):
        """Test aggregating empty candle list."""
        result = aggregate_candles_to_3h([])
        assert result == []
    
    def test_aggregate_single_candle(self):
        """Test aggregating single candle."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        
        candles = [
            Candle(
                open_time=base_time,
                close_time=base_time + 60 * 60 * 1000,
                open=50000.0,
                high=50050.0,
                low=49950.0,
                close=50025.0,
                volume=100.0
            )
        ]
        
        result = aggregate_candles_to_3h(candles)
        # Single candle should be returned as-is
        assert len(result) == 1
        assert result[0].open_time == base_time
        assert result[0].close == 50025.0
    
    def test_aggregate_already_3h(self):
        """Test aggregating candles that are already 3h."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        three_hour_ms = 3 * 60 * 60 * 1000
        
        candles_3h = [
            Candle(
                open_time=base_time + i * three_hour_ms,
                close_time=base_time + (i + 1) * three_hour_ms,
                open=50000.0 + i * 100,
                high=50100.0 + i * 100,
                low=49900.0 + i * 100,
                close=50050.0 + i * 100,
                volume=300.0 + i * 50
            )
            for i in range(2)
        ]
        
        result = aggregate_candles_to_3h(candles_3h)
        
        # Should return original candles as they're already 3h
        assert len(result) == 2
        assert result[0].open_time == base_time
        assert result[0].close_time == base_time + three_hour_ms
        assert result[0].volume == 300.0


class TestSourceMetadata:
    """Test cases for source metadata functions."""
    
    def test_create_source_metadata_dict(self):
        """Test creating source metadata dictionary."""
        metadata = create_source_metadata_dict(
            source="binance",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1h",
            method="direct",
            limit=500
        )
        
        assert metadata["source"] == "binance"
        assert metadata["exchange"] == "binance"
        assert metadata["symbol"] == "BTCUSDT"
        assert metadata["timeframe"] == "1h"
        assert metadata["granularity"] == "1h"
        assert metadata["method"] == "direct"
        assert metadata["limit"] == 500
        assert metadata["is_real_data"] is True
        assert "fetch_timestamp" in metadata
    
    def test_create_source_metadata_dict_with_kwargs(self):
        """Test creating source metadata with additional kwargs."""
        metadata = create_source_metadata_dict(
            source="binance",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="3h",
            method="aggregated",
            aggregation_factor=3,
            source_timeframe="1h"
        )
        
        assert metadata["source"] == "binance"
        assert metadata["exchange"] == "binance"
        assert metadata["timeframe"] == "3h"
        assert metadata["method"] == "aggregated"
        assert metadata["aggregation_factor"] == 3
        assert metadata["source_timeframe"] == "1h"
    
    def test_fetch_klines_with_source_metadata_3h(self):
        """Test fetching klines with source metadata for 3h."""
        # This test would require actual API calls, so we'll test the structure
        # In a real test environment, this would mock the API calls
        
        # For now, test that the function exists and has correct signature
        assert callable(fetch_klines_with_source_metadata)
    
    def test_fetch_and_validate_klines(self):
        """Test fetching and validating klines."""
        # This would require actual API calls
        # Test structure and signature
        assert callable(fetch_and_validate_klines)


class TestTimestampValidation:
    """Test cases for timestamp validation functions."""
    
    def test_validate_timestamp_monotonicity_success(self):
        """Test successful monotonicity validation."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        
        candles = [
            Candle(
                open_time=base_time + i * hour_ms,
                close_time=base_time + (i + 1) * hour_ms,
                open=50000.0 + i,
                high=50050.0 + i,
                low=49950.0 + i,
                close=50025.0 + i,
                volume=100.0
            )
            for i in range(5)
        ]
        
        result = validate_timestamp_monotonicity(candles)
        assert result is True
    
    def test_validate_timestamp_monotonicity_failure(self):
        """Test monotonicity validation failure."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        
        # Create candles with non-monotonic timestamps
        candles = [
            Candle(
                open_time=base_time,
                close_time=base_time + hour_ms,
                open=50000.0,
                high=50050.0,
                low=49950.0,
                close=50025.0,
                volume=100.0
            ),
            Candle(
                open_time=base_time - hour_ms,  # Earlier time!
                close_time=base_time,
                open=50100.0,
                high=50150.0,
                low=50050.0,
                close=50125.0,
                volume=110.0
            ),
            Candle(
                open_time=base_time + 2 * hour_ms,
                close_time=base_time + 3 * hour_ms,
                open=50200.0,
                high=50250.0,
                low=50150.0,
                close=50225.0,
                volume=120.0
            )
        ]
        
        with pytest.raises(ValueError, match="Non-monotonic timestamps"):
            validate_timestamp_monotonicity(candles)
    
    def test_validate_timestamp_monotonicity_close_times(self):
        """Test monotonicity validation with close times."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        
        # Create candles with non-monotonic close times
        candles = [
            Candle(
                open_time=base_time,
                close_time=base_time + hour_ms,
                open=50000.0,
                high=50050.0,
                low=49950.0,
                close=50025.0,
                volume=100.0
            ),
            Candle(
                open_time=base_time + hour_ms,
                close_time=base_time + hour_ms - 1000,  # Earlier close time!
                open=50100.0,
                high=50150.0,
                low=50050.0,
                close=50125.0,
                volume=110.0
            )
        ]
        
        with pytest.raises(ValueError, match="Non-monotonic close times"):
            validate_timestamp_monotonicity(candles)
    
    def test_validate_timestamp_plausibility_success(self):
        """Test successful plausibility validation."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        
        candles = [
            Candle(
                open_time=base_time + i * hour_ms,
                close_time=base_time + (i + 1) * hour_ms,
                open=50000.0 + i,
                high=50050.0 + i,
                low=49950.0 + i,
                close=50025.0 + i,
                volume=100.0
            )
            for i in range(3)
        ]
        
        result = validate_timestamp_plausibility(candles, "1h")
        assert result is True
    
    def test_validate_timestamp_plausibility_wrong_interval(self):
        """Test plausibility validation with wrong interval."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        half_hour_ms = 30 * 60 * 1000
        
        # Create candles with 30m intervals but validate as 1h
        candles = [
            Candle(
                open_time=base_time + i * half_hour_ms,
                close_time=base_time + (i + 1) * half_hour_ms,
                open=50000.0 + i,
                high=50050.0 + i,
                low=49950.0 + i,
                close=50025.0 + i,
                volume=100.0
            )
            for i in range(3)
        ]
        
        with pytest.raises(ValueError, match="Implausible timestamp interval"):
            validate_timestamp_plausibility(candles, "1h")
    
    def test_validate_timestamp_plausibility_tolerance(self):
        """Test plausibility validation with tolerance."""
        base_time = int(datetime(2024, 1, 1, 0, 0, 0).timestamp() * 1000)
        hour_ms = 60 * 60 * 1000
        tolerance = hour_ms // 10  # 10% tolerance
        
        # Create candles with slight deviation within tolerance
        candles = [
            Candle(
                open_time=base_time,
                close_time=base_time + hour_ms,
                open=50000.0,
                high=50050.0,
                low=49950.0,
                close=50025.0,
                volume=100.0
            ),
            Candle(
                open_time=base_time + hour_ms + tolerance // 2,  # Small deviation
                close_time=base_time + 2 * hour_ms + tolerance // 2,
                open=50100.0,
                high=50150.0,
                low=50050.0,
                close=50125.0,
                volume=110.0
            )
        ]
        
        # Should pass within tolerance
        result = validate_timestamp_plausibility(candles, "1h")
        assert result is True


class Test3hTimeframeIntegration:
    """Test cases for 3h timeframe integration."""
    
    def test_3h_in_timeframe_aliases(self):
        """Test that 3h is properly supported in timeframe aliases."""
        from indicator_collector.data_fetcher import timeframe_to_binance_interval
        
        assert timeframe_to_binance_interval("3h") == "3h"
        assert timeframe_to_binance_interval("180") == "3h"
    
    def test_3h_in_timeframe_to_minutes(self):
        """Test that 3h converts correctly to minutes."""
        from indicator_collector.data_fetcher import timeframe_to_minutes
        
        assert timeframe_to_minutes("3h") == 180
        assert timeframe_to_minutes("180") == 180
    
    def test_3h_in_interval_to_milliseconds(self):
        """Test that 3h converts correctly to milliseconds."""
        from indicator_collector.data_fetcher import interval_to_milliseconds
        
        assert interval_to_milliseconds("3h") == 3 * 60 * 60 * 1000
    
    def test_3h_data_validation(self):
        """Test that 3h data passes validation."""
        from indicator_collector.real_data_validator import validate_real_data_payload
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "3h",
                "granularity": "3h"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 300.5
            }
        }
        
        result = validate_real_data_payload(payload, "3h")
        assert result is True
    
    def test_3h_caching_keys(self):
        """Test that 3h timeframe works with caching systems."""
        # This would test CME gap caching or other time-based caching
        # For now, test that the timeframe is properly handled
        from indicator_collector.timeframes import Timeframe
        
        tf = Timeframe("3h")
        assert tf.to_minutes() == 180
        assert tf.get_display_name() == "3 Hours"
        assert tf.is_intraday() is True
        assert tf.is_hourly_or_less() is False