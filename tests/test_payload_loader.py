"""Tests for payload loading and processing functionality."""

import json
import pytest
from datetime import datetime, timedelta

from indicator_collector.trading_system.payload_loader import (
    PayloadProcessor,
    load_full_payload,
    load_and_process_payload_dict,
    validate_and_normalize_payload,
    extract_trading_context,
    payload_processor,
)
from indicator_collector.real_data_validator import DataValidationError


class TestPayloadProcessor:
    """Test cases for PayloadProcessor class."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = PayloadProcessor()
        assert processor.validator is not None
        assert processor.signal_generator is not None
        assert processor.position_manager is not None
        assert processor.statistics_optimizer is not None
    
    def test_load_full_payload_success(self):
        """Test successful payload loading and processing."""
        processor = PayloadProcessor()
        
        # Valid payload
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
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
        
        result = processor.load_full_payload(payload_dict, "1h", validate_real_data=True)
        
        # Should return TradingSignalPayload
        assert hasattr(result, 'signal_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'timeframe')
        assert hasattr(result, 'factors')
        assert hasattr(result, 'position_plan')
        assert hasattr(result, 'explanation')
        assert hasattr(result, 'metadata')
    
    def test_load_full_payload_json_string(self):
        """Test loading payload from JSON string."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        json_str = json.dumps(payload_dict)
        result = processor.load_full_payload(json_str, "1h", validate_real_data=True)
        
        assert hasattr(result, 'signal_type')
        assert result.timeframe == "1h"
    
    def test_load_full_payload_invalid_json(self):
        """Test loading invalid JSON string."""
        processor = PayloadProcessor()
        
        invalid_json = '{"metadata": {"source": "binance"'  # Missing closing braces
        
        with pytest.raises(json.JSONDecodeError):
            processor.load_full_payload(invalid_json, "1h")
    
    def test_load_full_payload_missing_timeframe(self):
        """Test loading payload without timeframe."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should fail because timeframe is missing
        with pytest.raises(ValueError, match="Timeframe not found"):
            processor.load_full_payload(payload_dict, None, validate_real_data=True)
    
    def test_load_full_payload_invalid_timeframe(self):
        """Test loading payload with invalid timeframe."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "invalid",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            processor.load_full_payload(payload_dict, "invalid", validate_real_data=True)
    
    def test_load_full_payload_synthetic_data(self):
        """Test loading payload with synthetic data."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "demo_api",
                "exchange": "testnet",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(DataValidationError, match="Synthetic data detected"):
            processor.load_full_payload(payload_dict, "1h", validate_real_data=True)
    
    def test_load_full_payload_no_validation(self):
        """Test loading payload without real data validation."""
        processor = PayloadProcessor()
        
        # Synthetic payload that should pass without validation
        payload_dict = {
            "metadata": {
                "source": "demo_api",
                "exchange": "testnet",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        # Should succeed when validation is disabled
        result = processor.load_full_payload(payload_dict, "1h", validate_real_data=False)
        assert hasattr(result, 'signal_type')
        assert result.timeframe == "1h"
    
    def test_process_payload_to_dict(self):
        """Test processing payload and returning as dictionary."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = processor.process_payload_to_dict(payload_dict, "1h", validate_real_data=True)
        
        # Should return dictionary
        assert isinstance(result, dict)
        assert "signal_type" in result
        assert "confidence" in result
        assert "timestamp" in result
        assert "timeframe" in result
        assert "metadata" in result
    
    def test_apply_timeframe_parameters(self):
        """Test applying timeframe parameters to context."""
        processor = PayloadProcessor()
        
        # Mock context
        from indicator_collector.trading_system.interfaces import AnalyzerContext
        context = AnalyzerContext(
            symbol="BTCUSDT",
            timeframe="1h",
            timestamp=int(datetime.now().timestamp() * 1000),
            current_price=50000.0,
            ohlcv={"open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 100.5},
            indicators={},
            metadata={},
            extras={}
        )
        
        processor._apply_timeframe_parameters(context, "3h")
        
        # Should have timeframe parameters in extras
        assert "timeframe_parameters" in context.extras
        assert context.extras["timeframe_parameters"]["sma_fast"] == 8  # 3h specific
        assert context.extras["timeframe_parameters"]["vwap_period"] == 8  # 3h specific
        
        # Should have timeframe info in metadata
        assert context.metadata["timeframe"] == "3h"
        assert context.metadata["timeframe_minutes"] == 180
        assert context.metadata["timeframe_display"] == "3 Hours"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_load_full_payload_convenience(self):
        """Test convenience function for loading full payload."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = load_full_payload(payload_dict, "1h", validate_real_data=True)
        
        assert hasattr(result, 'signal_type')
        assert result.timeframe == "1h"
    
    def test_load_and_process_payload_dict_convenience(self):
        """Test convenience function for processing payload as dictionary."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = load_and_process_payload_dict(payload_dict, "1h", validate_real_data=True)
        
        assert isinstance(result, dict)
        assert "signal_type" in result
        assert "timeframe" in result
    
    def test_validate_and_normalize_payload_success(self):
        """Test successful payload validation and normalization."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = validate_and_normalize_payload(payload_dict, "1h")
        
        assert isinstance(result, dict)
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["timeframe"] == "1h"
    
    def test_validate_and_normalize_payload_invalid_timeframe(self):
        """Test validation with invalid timeframe."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "invalid",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            validate_and_normalize_payload(payload_dict, None)
    
    def test_validate_and_normalize_payload_json_string(self):
        """Test validation with JSON string."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        json_str = json.dumps(payload_dict)
        result = validate_and_normalize_payload(json_str, "1h")
        
        assert isinstance(result, dict)
        assert result["metadata"]["source"] == "binance"
    
    def test_extract_trading_context(self):
        """Test extracting trading context from payload."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            },
            "indicators": {
                "trend_strength": 65.0,
                "rsi": 55.0,
                "macd": 10.5
            }
        }
        
        context = extract_trading_context(payload_dict)
        
        assert context.symbol == "BTCUSDT"
        assert context.timeframe == "1h"
        assert context.current_price == 50050.0
        assert context.indicators["trend_strength"] == 65.0
        assert context.indicators["rsi"] == 55.0


class TestGlobalProcessor:
    """Test cases for global processor instance."""
    
    def test_global_processor_instance(self):
        """Test that global processor instance is available."""
        assert payload_processor is not None
        assert isinstance(payload_processor, PayloadProcessor)
    
    def test_global_processor_load_payload(self):
        """Test using global processor to load payload."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "1h",
                "granularity": "1h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = payload_processor.load_full_payload(payload_dict, "1h", validate_real_data=False)
        
        assert hasattr(result, 'signal_type')
        assert result.timeframe == "1h"


class TestPayloadProcessorWith3hTimeframe:
    """Test payload processor specifically with 3h timeframe."""
    
    def test_3h_timeframe_processing(self):
        """Test processing payload with 3h timeframe."""
        processor = PayloadProcessor()
        
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "timeframe": "3h",
                "granularity": "3h",
                "symbol": "BTCUSDT"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = processor.load_full_payload(payload_dict, "3h", validate_real_data=True)
        
        assert result.timeframe == "3h"
        
        # Should have 3h-specific parameters applied
        timeframe_params = result.metadata.get("timeframe_parameters", {})
        if timeframe_params:
            assert timeframe_params["sma_fast"] == 8  # 3h specific
            assert timeframe_params["vwap_period"] == 8  # 3h specific
    
    def test_3h_aggregation_source_detection(self):
        """Test that 3h aggregation sources are detected correctly."""
        from indicator_collector.timeframes import get_aggregation_source_timeframes
        
        sources = get_aggregation_source_timeframes("3h")
        assert "1h" in sources
        assert "15m" in sources