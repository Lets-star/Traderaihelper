"""Tests for timeframe management functionality."""

import pytest

from indicator_collector.timeframes import (
    Timeframe,
    TimeframeParameters,
    timeframe_params,
    validate_timeframe,
    get_timeframe_info,
    get_aggregation_source_timeframes,
    get_aggregation_factor,
)


class TestTimeframeEnum:
    """Test cases for Timeframe enum."""
    
    def test_all_timeframes(self):
        """Test getting all supported timeframes."""
        all_tfs = Timeframe.all_timeframes()
        expected = ["1m", "3m", "5m", "15m", "30m", "1h", "3h", "4h", "1d"]
        
        assert len(all_tfs) == len(expected)
        for tf in expected:
            assert tf in all_tfs
    
    def test_common_timeframes(self):
        """Test getting commonly used timeframes."""
        common_tfs = Timeframe.common_timeframes()
        expected = ["5m", "15m", "30m", "1h", "3h", "4h", "1d"]
        
        assert len(common_tfs) == len(expected)
        for tf in expected:
            assert tf in common_tfs
    
    def test_to_minutes(self):
        """Test converting timeframes to minutes."""
        assert Timeframe.MINUTE_1.to_minutes() == 1
        assert Timeframe.MINUTE_3.to_minutes() == 3
        assert Timeframe.MINUTE_5.to_minutes() == 5
        assert Timeframe.MINUTE_15.to_minutes() == 15
        assert Timeframe.MINUTE_30.to_minutes() == 30
        assert Timeframe.HOUR_1.to_minutes() == 60
        assert Timeframe.HOUR_3.to_minutes() == 180
        assert Timeframe.HOUR_4.to_minutes() == 240
        assert Timeframe.DAY_1.to_minutes() == 1440
    
    def test_to_milliseconds(self):
        """Test converting timeframes to milliseconds."""
        assert Timeframe.MINUTE_1.to_milliseconds() == 60 * 1000
        assert Timeframe.HOUR_1.to_milliseconds() == 60 * 60 * 1000
        assert Timeframe.HOUR_3.to_milliseconds() == 3 * 60 * 60 * 1000
        assert Timeframe.DAY_1.to_milliseconds() == 24 * 60 * 60 * 1000
    
    def test_is_intraday(self):
        """Test intraday timeframe detection."""
        assert Timeframe.MINUTE_1.is_intraday() is True
        assert Timeframe.MINUTE_15.is_intraday() is True
        assert Timeframe.HOUR_1.is_intraday() is True
        assert Timeframe.HOUR_3.is_intraday() is True
        assert Timeframe.DAY_1.is_intraday() is False
    
    def test_is_hourly_or_less(self):
        """Test hourly-or-less timeframe detection."""
        assert Timeframe.MINUTE_1.is_hourly_or_less() is True
        assert Timeframe.MINUTE_15.is_hourly_or_less() is True
        assert Timeframe.HOUR_1.is_hourly_or_less() is True
        assert Timeframe.HOUR_3.is_hourly_or_less() is False
        assert Timeframe.DAY_1.is_hourly_or_less() is False
    
    def test_get_display_name(self):
        """Test getting display names."""
        assert Timeframe.MINUTE_1.get_display_name() == "1 Minute"
        assert Timeframe.MINUTE_3.get_display_name() == "3 Minutes"
        assert Timeframe.HOUR_1.get_display_name() == "1 Hour"
        assert Timeframe.HOUR_3.get_display_name() == "3 Hours"
        assert Timeframe.DAY_1.get_display_name() == "1 Day"


class TestTimeframeParameters:
    """Test cases for TimeframeParameters class."""
    
    def test_get_parameters(self):
        """Test getting parameters for timeframe."""
        params = timeframe_params.get_parameters("1h")
        
        assert isinstance(params, dict)
        assert "rsi_period" in params
        assert "macd_fast" in params
        assert "atr_period" in params
        assert params["rsi_period"] == 14
        assert params["macd_fast"] == 12
    
    def test_get_parameters_invalid_timeframe(self):
        """Test getting parameters for invalid timeframe."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_params.get_parameters("invalid")
    
    def test_get_parameter(self):
        """Test getting specific parameter."""
        rsi_period = timeframe_params.get_parameter("1h", "rsi_period")
        assert rsi_period == 14
        
        macd_fast = timeframe_params.get_parameter("1h", "macd_fast")
        assert macd_fast == 12
    
    def test_get_parameter_invalid_name(self):
        """Test getting invalid parameter name."""
        with pytest.raises(ValueError, match="Parameter 'invalid_param' not found"):
            timeframe_params.get_parameter("1h", "invalid_param")
    
    def test_set_parameter(self):
        """Test setting parameter."""
        # Original value
        original_rsi = timeframe_params.get_parameter("1h", "rsi_period")
        assert original_rsi == 14
        
        # Set new value
        timeframe_params.set_parameter("1h", "rsi_period", 20)
        new_rsi = timeframe_params.get_parameter("1h", "rsi_period")
        assert new_rsi == 20
        
        # Reset for other tests
        timeframe_params.set_parameter("1h", "rsi_period", 14)
    
    def test_get_rsi_period(self):
        """Test getting RSI period."""
        assert timeframe_params.get_rsi_period("1m") == 14
        assert timeframe_params.get_rsi_period("1h") == 14
        assert timeframe_params.get_rsi_period("3h") == 14
        assert timeframe_params.get_rsi_period("1d") == 14
    
    def test_get_macd_parameters(self):
        """Test getting MACD parameters."""
        fast, slow, signal = timeframe_params.get_macd_parameters("1h")
        assert fast == 12
        assert slow == 26
        assert signal == 9
        
        fast, slow, signal = timeframe_params.get_macd_parameters("3h")
        assert fast == 12
        assert slow == 26
        assert signal == 9
    
    def test_get_atr_period(self):
        """Test getting ATR period."""
        assert timeframe_params.get_atr_period("1m") == 14
        assert timeframe_params.get_atr_period("1h") == 14
        assert timeframe_params.get_atr_period("3h") == 14
        assert timeframe_params.get_atr_period("1d") == 14
    
    def test_get_sma_periods(self):
        """Test getting SMA periods."""
        fast, slow = timeframe_params.get_sma_periods("1h")
        assert fast == 9
        assert slow == 21
        
        fast, slow = timeframe_params.get_sma_periods("3h")
        assert fast == 8
        assert slow == 21
    
    def test_get_bollinger_parameters(self):
        """Test getting Bollinger Band parameters."""
        period, std = timeframe_params.get_bollinger_parameters("1h")
        assert period == 20
        assert std == 2
        
        period, std = timeframe_params.get_bollinger_parameters("3h")
        assert period == 20
        assert std == 2
    
    def test_get_volume_ma_period(self):
        """Test getting volume MA period."""
        assert timeframe_params.get_volume_ma_period("1h") == 20
        assert timeframe_params.get_volume_ma_period("3h") == 16
    
    def test_get_vwap_period(self):
        """Test getting VWAP period."""
        assert timeframe_params.get_vwap_period("1h") == 24
        assert timeframe_params.get_vwap_period("3h") == 8
    
    def test_get_orderbook_depth(self):
        """Test getting orderbook depth."""
        assert timeframe_params.get_orderbook_depth("1m") == 20
        assert timeframe_params.get_orderbook_depth("1h") == 20
        assert timeframe_params.get_orderbook_depth("3h") == 15
        assert timeframe_params.get_orderbook_depth("1d") == 10
    
    def test_get_data_point_limits(self):
        """Test getting data point limits."""
        min_points, max_points = timeframe_params.get_data_point_limits("1m")
        assert min_points == 100
        assert max_points == 1000
        
        min_points, max_points = timeframe_params.get_data_point_limits("3h")
        assert min_points == 20
        assert max_points == 200
        
        min_points, max_points = timeframe_params.get_data_point_limits("1d")
        assert min_points == 15
        assert max_points == 150


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_validate_timeframe(self):
        """Test timeframe validation."""
        assert validate_timeframe("1m") is True
        assert validate_timeframe("3h") is True
        assert validate_timeframe("1d") is True
        assert validate_timeframe("invalid") is False
        assert validate_timeframe("") is False
    
    def test_get_timeframe_info(self):
        """Test getting comprehensive timeframe information."""
        info = get_timeframe_info("1h")
        
        assert info["timeframe"] == "1h"
        assert info["display_name"] == "1 Hour"
        assert info["minutes"] == 60
        assert info["milliseconds"] == 60 * 60 * 1000
        assert info["is_intraday"] is True
        assert info["is_hourly_or_less"] is True
        assert isinstance(info["parameters"], dict)
        
        # Test 3h specifically
        info_3h = get_timeframe_info("3h")
        assert info_3h["timeframe"] == "3h"
        assert info_3h["display_name"] == "3 Hours"
        assert info_3h["minutes"] == 180
        assert info_3h["is_intraday"] is True
        assert info_3h["is_hourly_or_less"] is False
    
    def test_get_timeframe_info_invalid(self):
        """Test getting info for invalid timeframe."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            get_timeframe_info("invalid")
    
    def test_get_aggregation_source_timeframes(self):
        """Test getting aggregation source timeframes."""
        # 3h should have aggregation sources
        sources_3h = get_aggregation_source_timeframes("3h")
        assert "1h" in sources_3h
        assert "15m" in sources_3h
        
        # 1m should have aggregation sources
        sources_1m = get_aggregation_source_timeframes("1m")
        assert "1m" in sources_1m
        
        # Non-aggregatable timeframe should return empty
        sources_1h = get_aggregation_source_timeframes("1h")
        assert len(sources_1h) == 0
    
    def test_get_aggregation_factor(self):
        """Test getting aggregation factor."""
        # 1m to 3m
        factor = get_aggregation_factor("1m", "3m")
        assert factor == 3
        
        # 15m to 3h
        factor = get_aggregation_factor("15m", "3h")
        assert factor == 12
        
        # 1h to 3h
        factor = get_aggregation_factor("1h", "3h")
        assert factor == 3
        
        # Invalid aggregation
        with pytest.raises(ValueError, match="Cannot aggregate"):
            get_aggregation_factor("1h", "15m")
        
        with pytest.raises(ValueError, match="Cannot aggregate"):
            get_aggregation_factor("3h", "1m")


class TestTimeframeSpecificParameters:
    """Test that 3h timeframe has appropriate parameters."""
    
    def test_3h_specific_parameters(self):
        """Test that 3h has appropriate parameter values."""
        params = timeframe_params.get_parameters("3h")
        
        # Should have adjusted parameters for longer timeframe
        assert params["sma_fast"] == 8  # Lower than 1h (9)
        assert params["volume_ma_period"] == 16  # Lower than 1h (20)
        assert params["vwap_period"] == 8  # Much lower than 1h (24)
        assert params["orderbook_depth"] == 15  # Lower than 1h (20)
        assert params["min_data_points"] == 20  # Lower than 1h (25)
        assert params["max_data_points"] == 200  # Lower than 1h (250)
        
        # Should keep core parameters the same
        assert params["rsi_period"] == 14
        assert params["macd_fast"] == 12
        assert params["macd_slow"] == 26
        assert params["atr_period"] == 14
    
    def test_3h_vs_1h_parameters(self):
        """Test that 3h parameters differ appropriately from 1h."""
        params_3h = timeframe_params.get_parameters("3h")
        params_1h = timeframe_params.get_parameters("1h")
        
        # 3h should have more conservative parameters
        assert params_3h["sma_fast"] < params_1h["sma_fast"]
        assert params_3h["volume_ma_period"] < params_1h["volume_ma_period"]
        assert params_3h["vwap_period"] < params_1h["vwap_period"]
        assert params_3h["orderbook_depth"] < params_1h["orderbook_depth"]
        assert params_3h["min_data_points"] < params_1h["min_data_points"]
        assert params_3h["max_data_points"] < params_1h["max_data_points"]