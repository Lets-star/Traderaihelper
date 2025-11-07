"""Timeframe management and utilities for the trading system.

This module provides centralized timeframe handling including the new 3h timeframe
support and timeframe-specific parameter management.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple


class Timeframe(str, Enum):
    """Supported trading timeframes."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_3 = "3h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    
    @classmethod
    def all_timeframes(cls) -> List[str]:
        """Get all supported timeframes as strings."""
        return [tf.value for tf in cls]
    
    @classmethod
    def common_timeframes(cls) -> List[str]:
        """Get commonly used timeframes."""
        return [cls.MINUTE_5.value, cls.MINUTE_15.value, cls.MINUTE_30.value,
                cls.HOUR_1.value, cls.HOUR_3.value, cls.HOUR_4.value, cls.DAY_1.value]
    
    def to_minutes(self) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            Timeframe.MINUTE_1: 1,
            Timeframe.MINUTE_3: 3,
            Timeframe.MINUTE_5: 5,
            Timeframe.MINUTE_15: 15,
            Timeframe.MINUTE_30: 30,
            Timeframe.HOUR_1: 60,
            Timeframe.HOUR_3: 180,
            Timeframe.HOUR_4: 240,
            Timeframe.DAY_1: 1440,
        }
        return mapping[self]
    
    def to_milliseconds(self) -> int:
        """Convert timeframe to milliseconds."""
        return self.to_minutes() * 60 * 1000
    
    def is_intraday(self) -> bool:
        """Check if timeframe is intraday (less than 1 day)."""
        return self.to_minutes() < 1440
    
    def is_hourly_or_less(self) -> bool:
        """Check if timeframe is hourly or shorter."""
        return self.to_minutes() <= 60
    
    def get_display_name(self) -> str:
        """Get human-readable display name."""
        mapping = {
            Timeframe.MINUTE_1: "1 Minute",
            Timeframe.MINUTE_3: "3 Minutes",
            Timeframe.MINUTE_5: "5 Minutes",
            Timeframe.MINUTE_15: "15 Minutes",
            Timeframe.MINUTE_30: "30 Minutes",
            Timeframe.HOUR_1: "1 Hour",
            Timeframe.HOUR_3: "3 Hours",
            Timeframe.HOUR_4: "4 Hours",
            Timeframe.DAY_1: "1 Day",
        }
        return mapping[self]


class TimeframeParameters:
    """Manages timeframe-specific analysis parameters."""
    
    def __init__(self):
        self._parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> Dict[str, Dict[str, any]]:
        """Initialize default parameters for each timeframe."""
        return {
            Timeframe.MINUTE_1.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 390,  # One trading day in minutes
                "orderbook_depth": 20,
                "min_data_points": 100,
                "max_data_points": 1000,
            },
            Timeframe.MINUTE_3.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 130,  # One trading day in 3m intervals
                "orderbook_depth": 20,
                "min_data_points": 80,
                "max_data_points": 800,
            },
            Timeframe.MINUTE_5.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 78,  # One trading day in 5m intervals
                "orderbook_depth": 20,
                "min_data_points": 50,
                "max_data_points": 500,
            },
            Timeframe.MINUTE_15.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 26,  # One trading day in 15m intervals
                "orderbook_depth": 20,
                "min_data_points": 40,
                "max_data_points": 400,
            },
            Timeframe.MINUTE_30.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 13,  # One trading day in 30m intervals
                "orderbook_depth": 20,
                "min_data_points": 30,
                "max_data_points": 300,
            },
            Timeframe.HOUR_1.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 9,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 24,  # One day in hours
                "orderbook_depth": 20,
                "min_data_points": 25,
                "max_data_points": 250,
            },
            Timeframe.HOUR_3.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 8,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 16,
                "vwap_period": 8,  # One day in 3h intervals
                "orderbook_depth": 15,
                "min_data_points": 20,
                "max_data_points": 200,
            },
            Timeframe.HOUR_4.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 8,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 16,
                "vwap_period": 6,  # One day in 4h intervals
                "orderbook_depth": 15,
                "min_data_points": 18,
                "max_data_points": 180,
            },
            Timeframe.DAY_1.value: {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "sma_fast": 8,
                "sma_slow": 21,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "volume_ma_period": 20,
                "vwap_period": 1,  # One day
                "orderbook_depth": 10,
                "min_data_points": 15,
                "max_data_points": 150,
            },
        }
    
    def get_parameters(self, timeframe: str) -> Dict[str, any]:
        """Get parameters for a specific timeframe."""
        if timeframe not in self._parameters:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self._parameters[timeframe].copy()
    
    def get_parameter(self, timeframe: str, parameter_name: str) -> any:
        """Get a specific parameter for a timeframe."""
        params = self.get_parameters(timeframe)
        if parameter_name not in params:
            raise ValueError(f"Parameter '{parameter_name}' not found for timeframe '{timeframe}'")
        return params[parameter_name]
    
    def set_parameter(self, timeframe: str, parameter_name: str, value: any) -> None:
        """Set a parameter for a specific timeframe."""
        if timeframe not in self._parameters:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self._parameters[timeframe][parameter_name] = value
    
    def get_rsi_period(self, timeframe: str) -> int:
        """Get RSI period for timeframe."""
        return self.get_parameter(timeframe, "rsi_period")
    
    def get_macd_parameters(self, timeframe: str) -> Tuple[int, int, int]:
        """Get MACD parameters (fast, slow, signal) for timeframe."""
        fast = self.get_parameter(timeframe, "macd_fast")
        slow = self.get_parameter(timeframe, "macd_slow")
        signal = self.get_parameter(timeframe, "macd_signal")
        return fast, slow, signal
    
    def get_atr_period(self, timeframe: str) -> int:
        """Get ATR period for timeframe."""
        return self.get_parameter(timeframe, "atr_period")
    
    def get_sma_periods(self, timeframe: str) -> Tuple[int, int]:
        """Get SMA periods (fast, slow) for timeframe."""
        fast = self.get_parameter(timeframe, "sma_fast")
        slow = self.get_parameter(timeframe, "sma_slow")
        return fast, slow
    
    def get_bollinger_parameters(self, timeframe: str) -> Tuple[int, float]:
        """Get Bollinger Band parameters (period, std) for timeframe."""
        period = self.get_parameter(timeframe, "bollinger_period")
        std = self.get_parameter(timeframe, "bollinger_std")
        return period, std
    
    def get_volume_ma_period(self, timeframe: str) -> int:
        """Get volume moving average period for timeframe."""
        return self.get_parameter(timeframe, "volume_ma_period")
    
    def get_vwap_period(self, timeframe: str) -> int:
        """Get VWAP period for timeframe."""
        return self.get_parameter(timeframe, "vwap_period")
    
    def get_orderbook_depth(self, timeframe: str) -> int:
        """Get orderbook depth for timeframe."""
        return self.get_parameter(timeframe, "orderbook_depth")
    
    def get_data_point_limits(self, timeframe: str) -> Tuple[int, int]:
        """Get min/max data points for timeframe."""
        min_points = self.get_parameter(timeframe, "min_data_points")
        max_points = self.get_parameter(timeframe, "max_data_points")
        return min_points, max_points


# Global instance for easy access
timeframe_params = TimeframeParameters()


def validate_timeframe(timeframe: str) -> bool:
    """Validate if timeframe is supported."""
    try:
        Timeframe(timeframe)
        return True
    except ValueError:
        return False


def get_timeframe_info(timeframe: str) -> Dict[str, any]:
    """Get comprehensive information about a timeframe."""
    if not validate_timeframe(timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    tf = Timeframe(timeframe)
    params = timeframe_params.get_parameters(timeframe)
    
    return {
        "timeframe": timeframe,
        "display_name": tf.get_display_name(),
        "minutes": tf.to_minutes(),
        "milliseconds": tf.to_milliseconds(),
        "is_intraday": tf.is_intraday(),
        "is_hourly_or_less": tf.is_hourly_or_less(),
        "parameters": params,
    }


def get_aggregation_source_timeframes(target_timeframe: str) -> List[str]:
    """Get source timeframes that can be aggregated to create target timeframe."""
    aggregation_map = {
        Timeframe.MINUTE_3.value: [Timeframe.MINUTE_1.value],
        Timeframe.HOUR_3.value: [Timeframe.MINUTE_15.value, Timeframe.HOUR_1.value],
    }
    
    return aggregation_map.get(target_timeframe, [])


def get_aggregation_factor(source_timeframe: str, target_timeframe: str) -> int:
    """Get the factor needed to aggregate source to target timeframe."""
    source_minutes = Timeframe(source_timeframe).to_minutes()
    target_minutes = Timeframe(target_timeframe).to_minutes()
    
    if target_minutes % source_minutes != 0:
        raise ValueError(f"Cannot aggregate {source_timeframe} to {target_timeframe}")
    
    return target_minutes // source_minutes