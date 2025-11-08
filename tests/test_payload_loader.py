"""Tests for payload loading and processing functionality."""

import json
import pytest
from datetime import datetime, timedelta, timezone

from indicator_collector.trading_system.payload_loader import (
    PayloadProcessor,
    load_full_payload,
    load_and_process_payload_dict,
    validate_and_normalize_payload,
    extract_trading_context,
    payload_processor,
)
from indicator_collector.real_data_validator import DataValidationError
from indicator_collector.indicator_metrics import (
    summary_to_payload,
    SimulationSummary,
    MarketSnapshot,
    PnLStats,
    SuccessStats,
    SignalRecord,
)


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

    def test_summary_to_payload_adds_metadata_timestamp(self):
        """Ensure summary_to_payload populates metadata timestamp from latest snapshot."""
        timestamp = 1640995200000
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            close=50000.0,
            open=49950.0,
            high=50100.0,
            low=49800.0,
            volume=120.0,
            trend_strength=65.0,
            pattern_score=70.0,
            sentiment=60.0,
            structure_state="bullish",
            structure_event=None,
            volume_confirmed=True,
            volume_ratio=1.2,
            confluence_score=7.5,
            signal="BUY",
            volume_confidence=0.8,
            confluence_bias="bullish",
            confluence_bullish=7.0,
            confluence_bearish=3.0,
            rsi=55.0,
            macd=1.2,
            macd_signal=0.9,
            macd_histogram=0.3,
            bollinger_upper=50500.0,
            bollinger_middle=50000.0,
            bollinger_lower=49500.0,
            atr=150.0,
            atr_channels={"atr_trend_3x": 150.0},
            vwap=50020.0,
            sma_fast=49980.0,
            sma_slow=49850.0,
            rsi_divergence=None,
            macd_divergence=None,
        )
        summary = SimulationSummary(
            snapshots=[snapshot],
            signals=[],
            pnl=PnLStats(),
            success=SuccessStats(),
            active_fvg_zones=[],
            active_ob_zones=[],
            last_structure_levels={},
            multi_timeframe_trend={"1h": 65.0},
            multi_timeframe_direction={"1h": "bullish"},
            market_sentiment=60.0,
            pattern_prediction=55.0,
            multi_symbol=None,
        )
        payload = summary_to_payload(summary, "BINANCE:BTCUSDT", "1h", 200, "test-token")

        expected_iso = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()

        assert payload["metadata"]["timestamp"] == timestamp
        assert payload["metadata"]["timestamp_iso"] == expected_iso
        assert payload["latest"]["timestamp"] == timestamp
        assert payload["latest"]["time_iso"] == expected_iso

    def test_meta_timestamp_from_last_closed_candle(self):
        """Future snapshots should not drive the payload timestamp."""
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        one_hour_ms = 60 * 60 * 1000
        closed_ts = (now_ms // one_hour_ms) * one_hour_ms
        if closed_ts >= now_ms - 60000:
            closed_ts -= one_hour_ms
        future_ts = closed_ts + one_hour_ms

        closed_snapshot = MarketSnapshot(
            timestamp=closed_ts,
            close=50500.0,
            open=50400.0,
            high=50600.0,
            low=50300.0,
            volume=150.0,
            trend_strength=70.0,
            pattern_score=75.0,
            sentiment=65.0,
            structure_state="bullish",
            structure_event=None,
            volume_confirmed=True,
            volume_ratio=1.3,
            confluence_score=8.0,
            signal="BUY",
            volume_confidence=0.9,
            confluence_bias="bullish",
            confluence_bullish=8.5,
            confluence_bearish=2.0,
            rsi=60.0,
            macd=1.5,
            macd_signal=1.1,
            macd_histogram=0.4,
            bollinger_upper=51000.0,
            bollinger_middle=50500.0,
            bollinger_lower=50000.0,
            atr=160.0,
            atr_channels={"atr_trend_3x": 160.0},
            vwap=50520.0,
            sma_fast=50480.0,
            sma_slow=50350.0,
            rsi_divergence=None,
            macd_divergence=None,
        )

        future_snapshot = MarketSnapshot(
            timestamp=future_ts,
            close=52000.0,
            open=51900.0,
            high=52100.0,
            low=51800.0,
            volume=140.0,
            trend_strength=80.0,
            pattern_score=85.0,
            sentiment=75.0,
            structure_state="bullish",
            structure_event=None,
            volume_confirmed=True,
            volume_ratio=1.4,
            confluence_score=9.0,
            signal="BUY",
            volume_confidence=0.95,
            confluence_bias="bullish",
            confluence_bullish=9.0,
            confluence_bearish=1.5,
            rsi=65.0,
            macd=1.8,
            macd_signal=1.3,
            macd_histogram=0.5,
            bollinger_upper=52500.0,
            bollinger_middle=52000.0,
            bollinger_lower=51500.0,
            atr=170.0,
            atr_channels={"atr_trend_3x": 170.0},
            vwap=52020.0,
            sma_fast=51980.0,
            sma_slow=51850.0,
            rsi_divergence=None,
            macd_divergence=None,
        )

        closed_signal = SignalRecord(
            bar_index=5,
            timestamp=closed_ts,
            signal_type="bullish",
            price=50500.0,
            strength=0.85,
        )
        future_signal = SignalRecord(
            bar_index=6,
            timestamp=future_ts,
            signal_type="bullish",
            price=52000.0,
            strength=0.9,
        )

        summary = SimulationSummary(
            snapshots=[closed_snapshot, future_snapshot],
            signals=[closed_signal, future_signal],
            pnl=PnLStats(),
            success=SuccessStats(),
            active_fvg_zones=[],
            active_ob_zones=[],
            last_structure_levels={},
            multi_timeframe_trend={"1h": 70.0},
            multi_timeframe_direction={"1h": "bullish"},
            market_sentiment=65.0,
            pattern_prediction=60.0,
            multi_symbol=None,
        )

        payload = summary_to_payload(summary, "BINANCE:BTCUSDT", "1h", 200, "test-token")

        expected_iso = datetime.fromtimestamp(closed_ts / 1000, tz=timezone.utc).isoformat()

        assert payload["metadata"]["timestamp"] == closed_ts
        assert payload["latest"]["timestamp"] == closed_ts
        assert payload["latest"]["time_iso"] == expected_iso
        assert payload["latest"]["close"] == closed_snapshot.close
        assert all(signal["timestamp"] <= closed_ts for signal in payload["signals"])
        assert payload["metadata"]["timestamp"] <= int(datetime.now(tz=timezone.utc).timestamp() * 1000) + 60000

    def test_meta_timestamp_omitted_when_no_candles(self):
        """Raise a clear error when no snapshots are available."""
        summary = SimulationSummary(
            snapshots=[],
            signals=[],
            pnl=PnLStats(),
            success=SuccessStats(),
            active_fvg_zones=[],
            active_ob_zones=[],
            last_structure_levels={},
            multi_timeframe_trend={},
            multi_timeframe_direction={},
            market_sentiment=0.0,
            pattern_prediction=0.0,
            multi_symbol=None,
        )

        with pytest.raises(ValueError, match="No closed candles available for timeframe 1h"):
            summary_to_payload(summary, "BINANCE:BTCUSDT", "1h", 200, "test-token")


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