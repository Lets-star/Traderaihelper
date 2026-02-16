"""
pytest configuration and fixtures for trading system tests.

Provides common fixtures and configuration for all tests.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_bybit_credentials():
    """Provide mock ByBit API credentials."""
    return {
        "api_key": "test_api_key_123456789",
        "api_secret": "test_api_secret_123456789",
        "testnet": True,
    }


@pytest.fixture
def sample_signal():
    """Provide a sample valid signal."""
    return {
        "signal_id": "test_signal_001",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 50000.0,
        "take_profit": 55000.0,
        "stop_loss": 48000.0,
        "leverage": 5,
        "quantity": 0.001,
        "signal": "BUY",
        "entries": [50000.0],
        "take_profits": {"tp1": 55000.0},
    }


@pytest.fixture
def invalid_signal():
    """Provide an invalid signal for validation tests."""
    return {
        "signal_id": "",
        "symbol": "",
        "direction": "INVALID",
        "entry_price": -1,
        "leverage": 200,
    }


@pytest.fixture
def sample_candles_df():
    """Provide a sample DataFrame with candle data."""
    data = {
        "ts": [1700000000000 + i * 3600000 for i in range(100)],
        "open": [50000.0 + i * 10 for i in range(100)],
        "high": [50100.0 + i * 10 for i in range(100)],
        "low": [49900.0 + i * 10 for i in range(100)],
        "close": [50050.0 + i * 10 for i in range(100)],
        "volume": [100.0 + i for i in range(100)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_update_bus():
    """Provide a mock UpdateBus."""
    bus = MagicMock()
    bus.publish = MagicMock(return_value=True)
    bus.drain = MagicMock(return_value=[])
    bus.has_updates = MagicMock(return_value=False)
    bus.size = MagicMock(return_value=0)
    return bus


@pytest.fixture
def mock_websocket_app():
    """Provide a mock WebSocketApp."""
    ws = MagicMock()
    ws.sock = MagicMock()
    ws.sock.connected = True
    ws.run_forever = MagicMock()
    ws.close = MagicMock()
    return ws


@pytest.fixture
def disable_metrics():
    """Disable Prometheus metrics during tests."""
    with patch.dict(os.environ, {"DISABLE_PROMETHEUS_METRICS": "1"}):
        yield


@pytest.fixture(autouse=True)
def reset_metrics_state():
    """Reset metrics collectors before each test."""
    # Import here to avoid circular imports
    try:
        from metrics.collectors import (
            _signal_collector,
            _websocket_collector,
            _api_collector,
            _cache_collector,
            _worker_collector,
            _update_bus_collector,
        )
        
        # Reset global collectors
        globals().update({
            "_signal_collector": None,
            "_websocket_collector": None,
            "_api_collector": None,
            "_cache_collector": None,
            "_worker_collector": None,
            "_update_bus_collector": None,
        })
    except ImportError:
        pass
    
    yield


@pytest.fixture
def threaded_test_timeout():
    """Helper fixture for timing out threaded tests."""
    class TimeoutHelper:
        def __init__(self):
            self.timed_out = False
        
        def run_with_timeout(self, func, args=(), kwargs=None, timeout=5.0):
            """Run a function with a timeout."""
            kwargs = kwargs or {}
            result = [None]
            exception = [None]
            self.timed_out = False
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                self.timed_out = True
                raise TimeoutError(f"Test timed out after {timeout} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
    
    return TimeoutHelper()


@pytest.fixture
def mock_time():
    """Mock time for deterministic tests."""
    with patch("time.time") as mock:
        mock.return_value = 1700000000.0
        yield mock


@pytest.fixture
def mock_sleep():
    """Mock sleep to speed up tests."""
    with patch("time.sleep") as mock:
        yield mock


@pytest.fixture
def mock_datetime_now():
    """Mock datetime.utcnow for deterministic tests."""
    from datetime import datetime, timezone
    with patch("datetime.datetime") as mock:
        mock.utcnow.return_value = datetime(2023, 11, 14, 12, 0, 0, tzinfo=timezone.utc)
        yield mock


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "network: marks tests that make network requests")
    config.addinivalue_line("markers", "slow: marks tests that are slow (>5s)")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "concurrent: marks tests for concurrent execution")
    config.addinivalue_line("markers", "websocket: marks WebSocket-related tests")
    config.addinivalue_line("markers", "race_condition: marks race condition tests")


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return log messages."""
    caplog.set_level("DEBUG")
    return caplog
