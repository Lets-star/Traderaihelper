"""Timeframe management and utilities for the trading system.

This module provides centralized timeframe handling including the new 3h timeframe
support and timeframe-specific parameter management.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple


class Timeframe(str, Enum):
    """Supported trading timeframes."""
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    H1 = '1h'
    H3 = '3h'
    H4 = '4h'
    D1 = '1d'
    
    @classmethod
    def from_value(cls, value: str | 'Timeframe'):
        """Convert various timeframe formats to Timeframe enum with aliases support."""
        if isinstance(value, Timeframe): 
            return value
        v = str(value).strip().lower()
        aliases = {
            '1m': ['1m'], 
            '5m': ['5m'], 
            '15m': ['15m'], 
            '1h': ['1h', '60m'], 
            '3h': ['3h', '180m'], 
            '4h': ['4h', '240m'], 
            '1d': ['1d', '24h']
        }
        for k, vs in aliases.items():
            if v in vs: 
                return Timeframe(k)
        raise ValueError(f"Unsupported timeframe: {value}")
    
    @classmethod
    def is_supported(cls, value) -> bool:
        """Check if a timeframe value is supported."""
        try: 
            cls.from_value(value); 
            return True
        except Exception: 
            return False
    
    @classmethod
    def to_minutes(cls, value) -> int:
        """Convert timeframe value to minutes."""
        tf = cls.from_value(value)
        return {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '3h': 180, '4h': 240, '1d': 1440}[tf.value]
    
    @classmethod
    def validate_timeframe(cls, value):
        """Validate timeframe and return normalized Timeframe enum."""
        return cls.from_value(value)
    
    @classmethod
    def all_timeframes(cls) -> List[str]:
        """Get all supported timeframes as strings."""
        return [tf.value for tf in cls]
    
    @classmethod
    def common_timeframes(cls) -> List[str]:
        """Get commonly used timeframes."""
        return [cls.M5.value, cls.M15.value, cls.H1.value, cls.H3.value, cls.H4.value, cls.D1.value]
    
    def to_minutes_instance(self) -> int:
        """Convert timeframe enum to minutes."""
        return {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '3h': 180, '4h': 240, '1d': 1440}[self.value]
    
    def to_milliseconds(self) -> int:
        """Convert timeframe to milliseconds."""
        return self.to_minutes_instance() * 60 * 1000
    
    def is_intraday(self) -> bool:
        """Check if timeframe is intraday (less than 1 day)."""
        return self.to_minutes_instance() < 1440
    
    def is_hourly_or_less(self) -> bool:
        """Check if timeframe is hourly or shorter."""
        return self.to_minutes_instance() <= 60
    
    def get_display_name(self) -> str:
        """Get human-readable display name."""
        mapping = {
            Timeframe.M1: "1 Minute",
            Timeframe.M5: "5 Minutes", 
            Timeframe.M15: "15 Minutes",
            Timeframe.H1: "1 Hour",
            Timeframe.H3: "3 Hours",
            Timeframe.H4: "4 Hours",
            Timeframe.D1: "1 Day",
        }
        return mapping[self]


class TimeframeParameters:
    """Manages timeframe-specific analysis parameters."""
    
    def __init__(self):
        self._parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> Dict[str, Dict[str, any]]:
        """Initialize default parameters for each timeframe."""
        return {
            Timeframe.M1.value: {
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
            Timeframe.M5.value: {
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
            Timeframe.M15.value: {
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
            Timeframe.H1.value: {
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
            Timeframe.H3.value: {
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
            Timeframe.H4.value: {
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
            Timeframe.D1.value: {
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
    return Timeframe.is_supported(timeframe)


def get_timeframe_info(timeframe: str) -> Dict[str, any]:
    """Get comprehensive information about a timeframe."""
    if not validate_timeframe(timeframe):
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    tf = Timeframe.from_value(timeframe)
    params = timeframe_params.get_parameters(tf.value)
    
    return {
        "timeframe": tf.value,
        "display_name": tf.get_display_name(),
        "minutes": tf.to_minutes_instance(),
        "milliseconds": tf.to_milliseconds(),
        "is_intraday": tf.is_intraday(),
        "is_hourly_or_less": tf.is_hourly_or_less(),
        "parameters": params,
    }


def get_aggregation_source_timeframes(target_timeframe: str) -> List[str]:
    """Get source timeframes that can be aggregated to create target timeframe."""
    aggregation_map = {
        '3m': ['1m'],  # Note: 3m not in main enum but kept for compatibility
        Timeframe.H3.value: [Timeframe.M15.value, Timeframe.H1.value],
    }
    
    return aggregation_map.get(target_timeframe, [])


def get_aggregation_factor(source_timeframe: str, target_timeframe: str) -> int:
    """Get the factor needed to aggregate source to target timeframe."""
    source_minutes = Timeframe.to_minutes(source_timeframe)
    target_minutes = Timeframe.to_minutes(target_timeframe)
    
    if target_minutes % source_minutes != 0:
        raise ValueError(f"Cannot aggregate {source_timeframe} to {target_timeframe}")
    
    return target_minutes // source_minutes