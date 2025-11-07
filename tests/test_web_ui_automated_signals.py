"""Tests for updated web UI automated signals functionality."""

import json
import pytest
from unittest.mock import patch, MagicMock

from indicator_collector.trading_system import (
    load_and_process_payload_dict,
    validate_and_normalize_payload,
    indicator_defaults_for,
    ParameterSet,
)


class TestWebUIAutomatedSignals:
    """Test cases for web UI automated signals integration."""
    
    def test_validate_and_normalize_payload_success(self):
        """Test successful payload validation in web UI context."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,  # 2022-01-01
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5,
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 150.0
            }
        }
        
        result = validate_and_normalize_payload(payload, "1h")
        
        assert isinstance(result, dict)
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["timeframe"] == "1h"
    
    def test_validate_and_normalize_payload_with_3h(self):
        """Test payload validation with 3h timeframe."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "3h",
                "granularity": "3h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 300.5,
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 450.0
            }
        }
        
        result = validate_and_normalize_payload(payload, "3h")
        
        assert isinstance(result, dict)
        assert result["metadata"]["timeframe"] == "3h"
    
    def test_indicator_defaults_for_timeframes(self):
        """Indicator defaults should differ across key timeframes."""
        defaults_1h = indicator_defaults_for("1h")
        defaults_3h = indicator_defaults_for("3h")
        defaults_1d = indicator_defaults_for("1d")

        assert defaults_1h["atr"]["mult"] != defaults_3h["atr"]["mult"]
        assert defaults_1d["rsi"]["period"] >= defaults_1h["rsi"]["period"]
        assert "macd" in defaults_1h

    def test_parameter_set_uses_indicator_defaults(self):
        """ParameterSet should accept indicator defaults from helper."""
        indicator_params = indicator_defaults_for("3h")
        params = ParameterSet(
            weights={
                "technical": 0.4,
                "volume": 0.3,
                "sentiment": 0.2,
                "market_structure": 0.1,
            },
            indicator_params=indicator_params,
            timeframe="3h",
        )

        assert params.timeframe == "3h"
        assert params.indicator_params["atr"]["mult"] == indicator_params["atr"]["mult"]
        assert "bollinger" in params.indicator_params
    
    def test_load_and_process_payload_dict_basic(self):
        """Test basic payload processing for web UI."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5,
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 150.0,
                "trend_strength": 65.0,
                "market_sentiment": 60.0
            },
            "indicators": {
                "trend_strength": 65.0,
                "pattern_score": 70.0,
                "market_sentiment": 60.0
            },
            "multi_timeframe": {
                "trend_strength": {
                    "15m": 60.0,
                    "1h": 65.0,
                    "4h": 70.0
                },
                "direction": {
                    "15m": "bullish",
                    "1h": "bullish",
                    "4h": "bullish"
                }
            }
        }
        
        result = load_and_process_payload_dict(payload, "1h", validate_real_data=False)
        
        # Should return processed signal as dictionary
        assert isinstance(result, dict)
        assert "signal_type" in result
        assert "confidence" in result
        assert "timestamp" in result
        assert "symbol" in result
        assert "timeframe" in result
        assert "factors" in result
        assert "position_plan" in result
        assert "explanation" in result
        assert "metadata" in result
    
    def test_load_and_process_payload_dict_with_factors(self):
        """Test payload processing with factor analysis."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5,
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 150.0
            },
            "indicators": {
                "trend_strength": 65.0,
                "pattern_score": 70.0,
                "market_sentiment": 60.0
            }
        }
        
        result = load_and_process_payload_dict(payload, "1h", validate_real_data=False)
        
        # Should have factors
        assert "factors" in result
        factors = result["factors"]
        assert isinstance(factors, list)
        
        # Should have position plan
        assert "position_plan" in result
        position_plan = result["position_plan"]
        if position_plan:
            assert "entry_price" in position_plan or position_plan is None
    
    def test_load_and_process_payload_dict_with_3h(self):
        """Test payload processing with 3h timeframe."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "3h",
                "granularity": "3h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 300.5,  # Higher volume for 3h
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 450.0  # Higher ATR for 3h
            },
            "indicators": {
                "trend_strength": 65.0,
                "pattern_score": 70.0,
                "market_sentiment": 60.0
            }
        }
        
        result = load_and_process_payload_dict(payload, "3h", validate_real_data=False)
        
        assert result["timeframe"] == "3h"
        assert "factors" in result
        assert "position_plan" in result
    
    def test_load_and_process_payload_dict_json_string(self):
        """Test processing JSON string payload."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        json_str = json.dumps(payload)
        result = load_and_process_payload_dict(json_str, "1h", validate_real_data=False)
        
        assert isinstance(result, dict)
        assert result["symbol"] == "BTCUSDT"
        assert result["timeframe"] == "1h"
    
    def test_load_and_process_payload_dict_validation_enabled(self):
        """Test payload processing with validation enabled."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should work with validation enabled for real data
        result = load_and_process_payload_dict(payload, "1h", validate_real_data=True)
        
        assert isinstance(result, dict)
        assert "metadata" in result
        assert result["metadata"].get("real_data_validated") is True
    
    def test_load_and_process_payload_dict_synthetic_rejection(self):
        """Test that synthetic data is rejected when validation enabled."""
        payload = {
            "metadata": {
                "source": "demo_api",  # Synthetic source
                "exchange": "testnet",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should fail validation with synthetic data
        with pytest.raises(Exception):  # Should raise DataValidationError
            load_and_process_payload_dict(payload, "1h", validate_real_data=True)
    
    def test_payload_processing_metadata_enrichment(self):
        """Test that processed payload has enriched metadata."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = load_and_process_payload_dict(payload, "1h", validate_real_data=False)
        
        # Should have processing metadata
        assert "metadata" in result
        metadata = result["metadata"]
        
        assert "payload_processor" in metadata
        assert "timeframe_used" in metadata
        assert "real_data_validated" in metadata
        assert "processing_timestamp" in metadata
        assert "source_data_quality" in metadata
        
        assert metadata["timeframe_used"] == "1h"
        assert metadata["real_data_validated"] is False  # Validation disabled
    
    def test_payload_processing_timeframe_parameters(self):
        """Test that timeframe parameters are applied during processing."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "3h",
                "granularity": "3h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 300.5
            }
        }
        
        result = load_and_process_payload_dict(payload, "3h", validate_real_data=False)
        
        # Should have 3h-specific parameters applied
        assert "metadata" in result
        metadata = result["metadata"]
        
        # Check if timeframe parameters were applied (this depends on implementation)
        # The exact structure depends on how the payload processor applies parameters
        assert metadata.get("timeframe_used") == "3h"
    
    def test_payload_processing_error_handling(self):
        """Test error handling in payload processing."""
        # Invalid payload that should cause processing errors
        invalid_payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "invalid",  # Invalid timeframe
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should handle errors gracefully
        with pytest.raises(Exception):
            load_and_process_payload_dict(invalid_payload, "invalid", validate_real_data=False)
    
    def test_payload_processing_backward_compatibility(self):
        """Test that payload processing maintains backward compatibility."""
        # Payload in old format without some new fields
        old_format_payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "symbol": "BTCUSDT"
                # Missing timeframe, granularity
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should still process with provided timeframe
        result = load_and_process_payload_dict(old_format_payload, "1h", validate_real_data=False)
        
        assert isinstance(result, dict)
        assert result["timeframe"] == "1h"
        assert "metadata" in result


class TestWebUITimeframeSupport:
    """Test web UI timeframe support including 3h."""
    
    def test_timeframes_list_includes_3h(self):
        """Test that web UI timeframes include 3h."""
        # This would test the actual web UI TIMEFRAMES list
        # For now, test that 3h is a valid timeframe
        from indicator_collector.timeframes import Timeframe
        
        assert "3h" in Timeframe.all_timeframes()
        assert Timeframe.validate_timeframe("3h")
    
    def test_3h_timeframe_selection(self):
        """Test that 3h timeframe can be selected and processed."""
        from indicator_collector.timeframes import get_timeframe_info
        
        info = get_timeframe_info("3h")
        
        assert info["timeframe"] == "3h"
        assert info["display_name"] == "3 Hours"
        assert info["minutes"] == 180
        assert info["is_intraday"] is True
        assert info["is_hourly_or_less"] is False
    
    def test_timeframe_change_triggers_reanalysis(self):
        """Test that timeframe change triggers re-analysis."""
        # This would test the actual web UI behavior
        # For now, test that different timeframes produce different results
        
        payload_1h = {
            "metadata": {"timeframe": "1h", "source": "binance"},
            "latest": {"timestamp": 1640995200000, "close": 50050.0, "volume": 100.5}
        }
        
        payload_3h = {
            "metadata": {"timeframe": "3h", "source": "binance"},
            "latest": {"timestamp": 1640995200000, "close": 50050.0, "volume": 300.5}
        }
        
        result_1h = load_and_process_payload_dict(payload_1h, "1h", validate_real_data=False)
        result_3h = load_and_process_payload_dict(payload_3h, "3h", validate_real_data=False)
        
        # Should have different timeframe metadata
        assert result_1h["timeframe"] == "1h"
        assert result_3h["timeframe"] == "3h"
        
        # Should have different processing metadata
        assert result_1h["metadata"]["timeframe_used"] == "1h"
        assert result_3h["metadata"]["timeframe_used"] == "3h"


class TestWebUIAutoConsumption:
    """Test web UI automatic consumption of full JSON payload."""
    
    def test_auto_consumption_without_manual_mapping(self):
        """Test that payload is consumed automatically without manual field mapping."""
        # Complex payload with nested data
        complex_payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5,
                "rsi": 55.0,
                "macd": 10.5,
                "atr": 150.0,
                "trend_strength": 65.0,
                "pattern_score": 70.0,
                "market_sentiment": 60.0
            },
            "indicators": {
                "trend_strength": 65.0,
                "pattern_score": 70.0,
                "market_sentiment": 60.0,
                "volume_confirmed": True,
                "confluence_score": 75.0
            },
            "multi_timeframe": {
                "trend_strength": {"15m": 60.0, "1h": 65.0, "4h": 70.0},
                "direction": {"15m": "bullish", "1h": "bullish", "4h": "bullish"}
            },
            "advanced": {
                "volume_analysis": {"vpvr": {"poc": 50025.0}},
                "market_structure": {"trend": "bullish"}
            }
        }
        
        result = load_and_process_payload_dict(complex_payload, "1h", validate_real_data=False)
        
        # Should automatically consume all nested data
        assert isinstance(result, dict)
        assert "signal_type" in result
        assert "factors" in result
        assert "position_plan" in result
        assert "explanation" in result
        
        # Should preserve metadata
        assert "metadata" in result
        assert result["metadata"]["symbol"] == "BTCUSDT"
    
    def test_auto_consumption_with_historical_signals(self):
        """Test auto-consumption with historical signals data."""
        payload_with_history = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            },
            "historical_signals": [
                {"timestamp": 1640991600000, "signal_type": "BUY", "outcome": "profit"},
                {"timestamp": 1640988000000, "signal_type": "SELL", "outcome": "loss"},
                {"timestamp": 1640984400000, "signal_type": "BUY", "outcome": "profit"}
            ]
        }
        
        result = load_and_process_payload_dict(payload_with_history, "1h", validate_real_data=False)
        
        # Should include optimization stats when historical data available
        assert "optimization_stats" in result
        optimization_stats = result["optimization_stats"]
        if optimization_stats:
            assert "total_signals" in optimization_stats or optimization_stats is None
    
    def test_auto_consumption_error_recovery(self):
        """Test error recovery in auto-consumption."""
        # Payload with some problematic data
        problematic_payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": 1640995200000,
                "timeframe": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": 1640995200000,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            },
            "indicators": {
                # Some indicators might be missing or invalid
                "trend_strength": "invalid",  # String instead of number
                "pattern_score": None
            }
        }
        
        # Should handle problematic data gracefully
        result = load_and_process_payload_dict(problematic_payload, "1h", validate_real_data=False)
        
        # Should still produce a result, possibly with warnings
        assert isinstance(result, dict)
        assert "metadata" in result