"""
Tests for type definitions module.

This module tests:
- Protocol implementations
- TypedDict validation
- TypeGuard functions
- Enum values
- Generic types
"""

import pytest
from typing import Dict, Any, List

from trader_types import (
    # TypedDict
    SignalPayload,
    KlineData,
    ExecutionResult,
    # Enums
    Timeframe,
    SignalDirection,
    SignalStrength,
    FactorCategory,
    ExecutionStatus,
    OrderSide,
    OrderType,
    # Type guards
    is_valid_signal,
    is_kline_data,
    is_execution_result,
    is_position_data,
    is_streamlit_component,
    is_valid_symbol,
    is_valid_leverage,
    is_valid_confidence,
    # Generics
    UpdateBus,
    Result,
    DataStore,
)


class TestTypeGuards:
    """Tests for TypeGuard functions."""

    def test_is_valid_signal_valid(self):
        """Test valid signal detection."""
        signal = {
            "signal_id": "test-123",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry_price": 50000.0,
        }
        assert is_valid_signal(signal) is True

    def test_is_valid_signal_invalid(self):
        """Test invalid signal detection."""
        # Missing required field
        signal = {"symbol": "BTCUSDT", "direction": "LONG"}
        assert is_valid_signal(signal) is False

        # Wrong direction
        signal = {
            "signal_id": "test-123",
            "symbol": "BTCUSDT",
            "direction": "UP",  # Invalid
            "entry_price": 50000.0,
        }
        assert is_valid_signal(signal) is False

        # Not a dict
        assert is_valid_signal("not a dict") is False

    def test_is_kline_data_valid(self):
        """Test valid kline data detection."""
        kline = {
            "ts": 1700000000000,
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
        }
        assert is_kline_data(kline) is True

    def test_is_kline_data_invalid(self):
        """Test invalid kline data detection."""
        # Missing field
        kline = {"ts": 1700000000000, "open": 50000.0}
        assert is_kline_data(kline) is False

        # Wrong type
        kline = {
            "ts": "not a number",
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
        }
        assert is_kline_data(kline) is False

    def test_is_execution_result_valid(self):
        """Test valid execution result detection."""
        result = {"status": "filled"}
        assert is_execution_result(result) is True

    def test_is_execution_result_invalid(self):
        """Test invalid execution result detection."""
        # Missing status
        result = {"error": "something went wrong"}
        assert is_execution_result(result) is False

        # Wrong type for status
        result = {"status": 123}
        assert is_execution_result(result) is False

    def test_is_valid_symbol(self):
        """Test symbol validation."""
        assert is_valid_symbol("BTCUSDT") is True
        assert is_valid_symbol("ETHUSDT") is True
        assert is_valid_symbol("BTC") is True  # Minimum 3 chars
        assert is_valid_symbol("BT") is False  # Too short
        assert is_valid_symbol(123) is False  # Not a string
        assert is_valid_symbol("BTC-USDT") is False  # Invalid chars

    def test_is_valid_leverage(self):
        """Test leverage validation."""
        assert is_valid_leverage(5) is True
        assert is_valid_leverage(5.5) is True
        assert is_valid_leverage(1) is True
        assert is_valid_leverage(125) is True
        assert is_valid_leverage(0) is False
        assert is_valid_leverage(126) is False
        assert is_valid_leverage(-5) is False
        assert is_valid_leverage("invalid") is False

    def test_is_valid_confidence(self):
        """Test confidence validation."""
        assert is_valid_confidence(0.5) is True
        assert is_valid_confidence(0) is True
        assert is_valid_confidence(1) is True
        assert is_valid_confidence(-0.1) is False
        assert is_valid_confidence(1.1) is False
        assert is_valid_confidence("invalid") is False


class TestEnums:
    """Tests for Enum types."""

    def test_timeframe_values(self):
        """Test Timeframe enum values."""
        assert Timeframe.MINUTE_1.value == "1m"
        assert Timeframe.HOUR_1.value == "1h"
        assert Timeframe.DAY_1.value == "1d"

    def test_timeframe_from_string(self):
        """Test creating Timeframe from string."""
        tf = Timeframe.from_string("1h")
        assert tf == Timeframe.HOUR_1

        with pytest.raises(ValueError):
            Timeframe.from_string("invalid")

    def test_timeframe_milliseconds(self):
        """Test Timeframe millisecond conversion."""
        assert Timeframe.MINUTE_1.milliseconds == 60_000
        assert Timeframe.HOUR_1.milliseconds == 3_600_000
        assert Timeframe.DAY_1.milliseconds == 86_400_000

    def test_signal_direction(self):
        """Test SignalDirection enum."""
        assert SignalDirection.LONG.value == "LONG"
        assert SignalDirection.SHORT.value == "SHORT"
        assert SignalDirection.LONG.order_side == "Buy"
        assert SignalDirection.SHORT.order_side == "Sell"
        assert SignalDirection.LONG.opposite == SignalDirection.SHORT

    def test_signal_strength_from_confidence(self):
        """Test SignalStrength from confidence."""
        assert SignalStrength.from_confidence(0.9) == SignalStrength.STRONG
        assert SignalStrength.from_confidence(0.7) == SignalStrength.MODERATE
        assert SignalStrength.from_confidence(0.5) == SignalStrength.WEAK
        assert SignalStrength.from_confidence(0.3) == SignalStrength.HOLD

    def test_order_side_from_direction(self):
        """Test OrderSide from SignalDirection."""
        assert OrderSide.from_direction(SignalDirection.LONG) == OrderSide.BUY
        assert OrderSide.from_direction(SignalDirection.SHORT) == OrderSide.SELL

    def test_factor_category_default_weights(self):
        """Test FactorCategory default weights."""
        weights = FactorCategory.get_default_weights()
        assert "technical" in weights
        assert "sentiment" in weights
        assert sum(weights.values()) == 1.0


class TestGenericTypes:
    """Tests for Generic types."""

    def test_update_bus_generic(self):
        """Test UpdateBus with different types."""
        # Integer bus
        bus: UpdateBus[int] = UpdateBus()
        assert bus.publish(42) is True
        assert bus.drain() == [42]

        # String bus
        str_bus: UpdateBus[str] = UpdateBus()
        str_bus.publish("hello")
        assert str_bus.drain() == ["hello"]

    def test_update_bus_overflow(self):
        """Test UpdateBus overflow handling."""
        bus: UpdateBus[int] = UpdateBus(max_size=2)
        assert bus.publish(1) is True
        assert bus.publish(2) is True
        assert bus.publish(3) is False  # Queue full
        assert bus.get_dropped_count() == 1

    def test_result_ok(self):
        """Test Result.ok factory."""
        result = Result.ok(42)
        assert result.is_ok is True
        assert result.is_err is False
        assert result.unwrap() == 42
        assert result.unwrap_or(0) == 42

    def test_result_err(self):
        """Test Result.err factory."""
        error = ValueError("test error")
        result = Result.err(error)
        assert result.is_ok is False
        assert result.is_err is True
        assert result.unwrap_or(0) == 0
        assert result.error == error

    def test_result_unwrap_raises(self):
        """Test Result.unwrap raises on error."""
        result = Result.err(ValueError("test"))
        with pytest.raises(RuntimeError):
            result.unwrap()

    def test_data_store(self):
        """Test DataStore."""
        store: DataStore[str, int] = DataStore()
        store.set("key1", 100)
        assert store.get("key1") == 100
        assert store.get("nonexistent") is None
        assert store.get("nonexistent", 0) == 0
        assert store.contains("key1") is True
        assert store.contains("nonexistent") is False

    def test_data_store_max_size(self):
        """Test DataStore max size eviction."""
        store: DataStore[str, int] = DataStore(max_size=2)
        store.set("key1", 1)
        store.set("key2", 2)
        store.set("key3", 3)  # Should evict key1
        assert store.contains("key1") is False
        assert store.contains("key2") is True
        assert store.contains("key3") is True

    def test_data_store_get_or_compute(self):
        """Test DataStore get_or_compute."""
        store: DataStore[str, int] = DataStore()
        computed = []

        def compute() -> int:
            computed.append(1)
            return 42

        # First access - compute
        value = store.get_or_compute("key", compute)
        assert value == 42
        assert len(computed) == 1

        # Second access - cached
        value = store.get_or_compute("key", compute)
        assert value == 42
        assert len(computed) == 1  # Not recomputed


class TestTypedDict:
    """Tests for TypedDict types."""

    def test_signal_payload_structure(self):
        """Test SignalPayload structure."""
        signal: SignalPayload = {
            "signal_id": "test-123",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry_price": 50000.0,
        }
        assert signal["signal_id"] == "test-123"
        assert signal["symbol"] == "BTCUSDT"

    def test_kline_data_structure(self):
        """Test KlineData structure."""
        kline: KlineData = {
            "ts": 1700000000000,
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.5,
        }
        assert kline["ts"] == 1700000000000
        assert kline["close"] == 50500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
