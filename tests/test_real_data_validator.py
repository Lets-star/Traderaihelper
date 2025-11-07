"""Tests for real data validation functionality."""

import json
import pytest
from datetime import datetime, timedelta

from indicator_collector.real_data_validator import (
    DataValidationError,
    RealDataValidator,
    DataSource,
    validate_real_data_payload,
    load_and_validate_json_payload,
)


class TestRealDataValidator:
    """Test cases for RealDataValidator class."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = RealDataValidator()
        assert validator.validation_errors == []
    
    def test_validate_payload_sources_success(self):
        """Test successful payload source validation."""
        validator = RealDataValidator()
        
        # Valid real data payload
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
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
        
        result = validator.validate_payload_sources(payload)
        assert result is True
    
    def test_validate_payload_sources_missing_metadata(self):
        """Test payload validation with missing metadata."""
        validator = RealDataValidator()
        
        payload = {
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(DataValidationError, match="Missing metadata section"):
            validator.validate_payload_sources(payload)
    
    def test_validate_payload_sources_synthetic_source(self):
        """Test payload validation with synthetic source."""
        validator = RealDataValidator()
        
        payload = {
            "metadata": {
                "source": "demo_api",
                "exchange": "testnet",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
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
            validator.validate_payload_sources(payload)
    
    def test_validate_payload_sources_invalid_timestamp(self):
        """Test payload validation with invalid timestamp."""
        validator = RealDataValidator()
        
        # Future timestamp
        future_time = (datetime.now() + timedelta(days=365)).timestamp() * 1000
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": future_time,
                "granularity": "1m"
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
        
        with pytest.raises(DataValidationError, match="Invalid timestamp"):
            validator.validate_payload_sources(payload)
    
    def test_ensure_no_synthetic_flags_success(self):
        """Test synthetic flag detection with clean data."""
        validator = RealDataValidator()
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
            },
            "data": [
                {"price": 50000.0, "volume": 100.5},
                {"price": 50100.0, "volume": 98.2}
            ]
        }
        
        result = validator.ensure_no_synthetic_flags(payload)
        assert result is True
    
    def test_ensure_no_synthetic_flags_detection(self):
        """Test synthetic flag detection with synthetic markers."""
        validator = RealDataValidator()
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
            },
            "data": [
                {"price": 50000.0, "volume": 100.5, "type": "mock_data"},
                {"price": 50100.0, "volume": 98.2}
            ]
        }
        
        with pytest.raises(DataValidationError, match="Synthetic data markers detected"):
            validator.ensure_no_synthetic_flags(payload)
    
    def test_validate_time_continuity_success(self):
        """Test successful time continuity validation."""
        validator = RealDataValidator()
        
        current_time = datetime.now().timestamp() * 1000
        recent_time = current_time - 60000  # 1 minute ago
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": recent_time,
                "granularity": "1m"
            },
            "latest": {
                "timestamp": current_time,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        result = validator.validate_time_continuity(payload, "1m")
        assert result is True
    
    def test_validate_time_continuity_future_timestamp(self):
        """Test time continuity validation with future timestamp."""
        validator = RealDataValidator()
        
        future_time = (datetime.now() + timedelta(minutes=5)).timestamp() * 1000
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": future_time,
                "granularity": "1m"
            },
            "latest": {
                "timestamp": future_time,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(DataValidationError, match="timestamp is in the future"):
            validator.validate_time_continuity(payload, "1m")
    
    def test_validate_time_continuity_stale_data(self):
        """Test time continuity validation with stale data."""
        validator = RealDataValidator()
        
        stale_time = (datetime.now() - timedelta(days=2)).timestamp() * 1000
        
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": stale_time,
                "granularity": "1m"
            },
            "latest": {
                "timestamp": stale_time,
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        }
        
        with pytest.raises(DataValidationError, match="Data is too old"):
            validator.validate_time_continuity(payload, "1m")
    
    def test_validate_ohlcv_data_success(self):
        """Test OHLCV data validation with valid data."""
        validator = RealDataValidator()
        
        ohlcv = {
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.5
        }
        
        # Should not raise any errors
        validator._validate_ohlcv_data(ohlcv)
        assert len(validator.validation_errors) == 0
    
    def test_validate_ohlcv_data_invalid_relationships(self):
        """Test OHLCV data validation with invalid OHLC relationships."""
        validator = RealDataValidator()
        
        # Invalid: low > high
        ohlcv = {
            "open": 50000.0,
            "high": 49900.0,  # Lower than low
            "low": 50100.0,  # Higher than high
            "close": 50050.0,
            "volume": 100.5
        }
        
        validator._validate_ohlcv_data(ohlcv)
        assert len(validator.validation_errors) > 0
        assert any("OHLC relationship violation" in error for error in validator.validation_errors)
    
    def test_validate_ohlcv_data_zero_prices(self):
        """Test OHLCV data validation with zero prices."""
        validator = RealDataValidator()
        
        ohlcv = {
            "open": 0.0,  # Zero price
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.5
        }
        
        validator._validate_ohlcv_data(ohlcv)
        assert len(validator.validation_errors) > 0
        assert any("Zero price detected" in error for error in validator.validation_errors)
    
    def test_contains_synthetic_markers(self):
        """Test synthetic marker detection."""
        validator = RealDataValidator()
        
        # Should detect synthetic markers
        assert validator._contains_synthetic_markers("demo_data") is True
        assert validator._contains_synthetic_markers("mock_api") is True
        assert validator._contains_synthetic_markers("test_source") is True
        assert validator._contains_synthetic_markers("simulated") is True
        
        # Should not detect synthetic markers
        assert validator._contains_synthetic_markers("binance") is False
        assert validator._contains_synthetic_markers("coinbase") is False
        assert validator._contains_synthetic_markers("real_data") is False
    
    def test_is_valid_timestamp(self):
        """Test timestamp validation."""
        validator = RealDataValidator()
        
        # Valid timestamps
        current_timestamp = datetime.now().timestamp() * 1000
        assert validator._is_valid_timestamp(current_timestamp) is True
        
        # Invalid timestamps
        old_timestamp = datetime(2019, 1, 1).timestamp() * 1000
        assert validator._is_valid_timestamp(old_timestamp) is False
        
        future_timestamp = datetime(2031, 1, 1).timestamp() * 1000
        assert validator._is_valid_timestamp(future_timestamp) is False
        
        # Invalid type
        assert validator._is_valid_timestamp("invalid") is False
        assert validator._is_valid_timestamp(None) is False


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_validate_real_data_payload_success(self):
        """Test successful payload validation."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
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
        
        result = validate_real_data_payload(payload, "1m")
        assert result is True
    
    def test_validate_real_data_payload_failure(self):
        """Test payload validation with synthetic data."""
        payload = {
            "metadata": {
                "source": "demo_api",
                "exchange": "testnet",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
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
        
        with pytest.raises(DataValidationError):
            validate_real_data_payload(payload, "1m")
    
    def test_load_and_validate_json_payload_string(self):
        """Test loading and validating JSON string."""
        json_str = json.dumps({
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        })
        
        result = load_and_validate_json_payload(json_str, "1m")
        assert isinstance(result, dict)
        assert result["metadata"]["source"] == "binance"
    
    def test_load_and_validate_json_payload_dict(self):
        """Test loading and validating dictionary."""
        payload_dict = {
            "metadata": {
                "source": "binance",
                "exchange": "binance",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
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
        
        result = load_and_validate_json_payload(payload_dict, "1m")
        assert isinstance(result, dict)
        assert result["metadata"]["source"] == "binance"
    
    def test_load_and_validate_json_payload_invalid_json(self):
        """Test loading invalid JSON string."""
        invalid_json = '{"metadata": {"source": "binance"'  # Missing closing braces
        
        with pytest.raises(json.JSONDecodeError):
            load_and_validate_json_payload(invalid_json, "1m")
    
    def test_load_and_validate_json_payload_validation_failure(self):
        """Test loading JSON that fails validation."""
        json_str = json.dumps({
            "metadata": {
                "source": "demo_api",  # Synthetic source
                "exchange": "testnet",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "granularity": "1m"
            },
            "latest": {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 100.5
            }
        })
        
        with pytest.raises(DataValidationError):
            load_and_validate_json_payload(json_str, "1m")


class TestDataSourceEnum:
    """Test cases for DataSource enum."""
    
    def test_data_source_values(self):
        """Test DataSource enum values."""
        assert DataSource.BINANCE.value == "binance"
        assert DataSource.COINBASE.value == "coinbase"
        assert DataSource.KRAKEN.value == "kraken"
        assert DataSource.BITFINEX.value == "bitfinex"
        assert DataSource.UNKNOWN.value == "unknown"
    
    def test_data_source_comparison(self):
        """Test DataSource enum comparison."""
        assert DataSource.BINANCE == "binance"
        assert DataSource.BINANCE != "coinbase"
        assert DataSource.BINANCE != DataSource.COINBASE