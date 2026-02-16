"""
Integration tests for AutomatedSignalsWorker.

Tests WebSocket integration with mocked client, signal flow end-to-end,
config updates propagation, error handling in _refresh_signals,
SignalExecutor integration, and worker start/stop lifecycle.
"""

from __future__ import annotations

import datetime
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Skip if dependencies not available
pytestmark = [
    pytest.mark.integration,
]


class MockWebSocketClient:
    """Mock WebSocket client for testing."""
    
    def __init__(self, symbol: str, interval: str, **kwargs):
        self.symbol = symbol
        self.interval = interval
        self.on_closed_bar = kwargs.get('on_closed_bar')
        self.on_forming_bar = kwargs.get('on_forming_bar')
        self._running = False
        self._callbacks = kwargs
    
    def start(self):
        self._running = True
    
    def stop(self):
        self._running = False
    
    def is_connected(self):
        return self._running
    
    def simulate_closed_bar(self, kline: Dict):
        """Simulate receiving a closed bar."""
        if self.on_closed_bar:
            self.on_closed_bar(kline)


class TestAutomatedSignalsWorkerLifecycle:
    """Test worker lifecycle: start, run, stop."""
    
    @pytest.fixture
    def worker_dependencies(self):
        """Provide mocked dependencies for worker."""
        update_bus = MagicMock()
        update_bus.publish = MagicMock(return_value=True)
        
        signal_config = {
            "weights": {
                "technical": 0.25,
                "sentiment": 0.15,
                "multitimeframe": 0.10,
                "volume": 0.20,
                "market_structure": 0.15,
                "composite": 0.0,
            },
            "min_confirmations": 3,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "min_confidence": 0.6,
        }
        
        indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9},
        }
        
        signal_params = {}
        
        return {
            "update_bus": update_bus,
            "signal_config": signal_config,
            "indicator_params": indicator_params,
            "signal_params": signal_params,
        }
    
    def test_worker_initialization(self, worker_dependencies):
        """Test worker initializes correctly."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=worker_dependencies["update_bus"],
            signal_config_payload=worker_dependencies["signal_config"],
            indicator_params=worker_dependencies["indicator_params"],
            signal_params=worker_dependencies["signal_params"],
        )
        
        assert worker.symbol == "BTCUSDT"
        assert worker.timeframe == "1h"
        assert worker.ws_client is None
        assert worker.df.empty
    
    def test_worker_start_creates_websocket(self, worker_dependencies):
        """Test that start creates WebSocket client."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        with patch('automated_signals_worker.BinanceWebSocketClient', MockWebSocketClient):
            with patch.object(AutomatedSignalsWorker, '_refresh_signals'):
                with patch('automated_signals_worker.get_binance_server_time_ms', return_value=1700000000000):
                    with patch.object(MockWebSocketClient, 'start'):
                        worker = AutomatedSignalsWorker(
                            symbol="BTCUSDT",
                            timeframe="1h",
                            update_bus=worker_dependencies["update_bus"],
                            signal_config_payload=worker_dependencies["signal_config"],
                            indicator_params=worker_dependencies["indicator_params"],
                            signal_params=worker_dependencies["signal_params"],
                        )
                        
                        # Mock data source
                        worker.data_source = MagicMock()
                        worker.data_source.load_candles.return_value = pd.DataFrame({
                            "ts": [1700000000000],
                            "open": [50000.0],
                            "high": [50100.0],
                            "low": [49900.0],
                            "close": [50050.0],
                            "volume": [100.0],
                        })
                        
                        worker.start()
                        
                        assert worker.ws_client is not None
                        assert isinstance(worker.ws_client, MockWebSocketClient)
    
    def test_worker_stop_cleans_up(self, worker_dependencies):
        """Test that stop cleans up resources."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        with patch('automated_signals_worker.BinanceWebSocketClient', MockWebSocketClient):
            with patch.object(AutomatedSignalsWorker, '_refresh_signals'):
                with patch('automated_signals_worker.get_binance_server_time_ms', return_value=1700000000000):
                    worker = AutomatedSignalsWorker(
                        symbol="BTCUSDT",
                        timeframe="1h",
                        update_bus=worker_dependencies["update_bus"],
                        signal_config_payload=worker_dependencies["signal_config"],
                        indicator_params=worker_dependencies["indicator_params"],
                        signal_params=worker_dependencies["signal_params"],
                    )
                    
                    # Mock data source
                    worker.data_source = MagicMock()
                    worker.data_source.load_candles.return_value = pd.DataFrame({
                        "ts": [1700000000000],
                        "open": [50000.0],
                        "high": [50100.0],
                        "low": [49900.0],
                        "close": [50050.0],
                        "volume": [100.0],
                    })
                    
                    worker.start()
                    worker.stop()
                    
                    assert worker.ws_client is None


class TestAutomatedSignalsWorkerWebSocketIntegration:
    """Test WebSocket integration."""
    
    @pytest.fixture
    def worker_with_mock_ws(self):
        """Create worker with mocked WebSocket."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        update_bus.publish = MagicMock(return_value=True)
        
        signal_config = {
            "weights": {
                "technical": 0.25,
                "sentiment": 0.15,
                "multitimeframe": 0.10,
                "volume": 0.20,
                "market_structure": 0.15,
                "composite": 0.0,
            },
            "min_confirmations": 3,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "min_confidence": 0.6,
        }
        
        indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9},
        }
        
        signal_params = {}
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload=signal_config,
            indicator_params=indicator_params,
            signal_params=signal_params,
        )
        
        # Set up initial data
        worker.df = pd.DataFrame({
            "ts": range(1700000000000, 1700003600000, 3600000),
            "open": [50000.0] * 10,
            "high": [50100.0] * 10,
            "low": [49900.0] * 10,
            "close": [50050.0] * 10,
            "volume": [100.0] * 10,
        })
        
        return worker, update_bus
    
    def test_closed_bar_callback_appends_data(self, worker_with_mock_ws):
        """Test that closed bar callback appends data to DataFrame."""
        worker, _ = worker_with_mock_ws
        
        initial_len = len(worker.df)
        
        # Simulate closed bar
        kline = {
            "ts": 1700003600000,
            "open": 50050.0,
            "high": 50150.0,
            "low": 49950.0,
            "close": 50100.0,
            "volume": 150.0,
        }
        
        with patch.object(worker, '_refresh_signals'):
            worker._on_closed_kline(kline)
        
        assert len(worker.df) == initial_len + 1
        assert worker.df["ts"].iloc[-1] == 1700003600000
    
    def test_closed_bar_triggers_signal_refresh(self, worker_with_mock_ws):
        """Test that closed bar triggers signal refresh."""
        worker, _ = worker_with_mock_ws
        
        with patch.object(worker, '_refresh_signals') as mock_refresh:
            kline = {
                "ts": 1700003600000,
                "open": 50050.0,
                "high": 50150.0,
                "low": 49950.0,
                "close": 50100.0,
                "volume": 150.0,
            }
            
            worker._on_closed_kline(kline)
            
            mock_refresh.assert_called_once()
    
    def test_dataframe_trimmed_after_1000_bars(self, worker_with_mock_ws):
        """Test that DataFrame is trimmed to 1000 bars."""
        worker, _ = worker_with_mock_ws
        
        # Create 1500 bars
        worker.df = pd.DataFrame({
            "ts": range(1700000000000, 1700000000000 + 1500 * 3600000, 3600000),
            "open": [50000.0] * 1500,
            "high": [50100.0] * 1500,
            "low": [49900.0] * 1500,
            "close": [50050.0] * 1500,
            "volume": [100.0] * 1500,
        })
        
        with patch.object(worker, '_refresh_signals'):
            kline = {
                "ts": 1700000000000 + 1500 * 3600000,
                "open": 50050.0,
                "high": 50150.0,
                "low": 49950.0,
                "close": 50100.0,
                "volume": 150.0,
            }
            
            worker._on_closed_kline(kline)
        
        assert len(worker.df) <= 1000


class TestAutomatedSignalsWorkerErrorHandling:
    """Test error handling."""
    
    @pytest.fixture
    def worker(self):
        """Create worker for error testing."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        update_bus.publish = MagicMock(return_value=True)
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
        )
        
        return worker, update_bus
    
    def test_refresh_signals_error_published(self, worker):
        """Test that refresh signal errors are published to UpdateBus."""
        worker_instance, update_bus = worker
        
        # Set up minimal data
        worker_instance.df = pd.DataFrame({
            "ts": [1700000000000],
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.0],
        })
        
        # Force an error
        with patch('automated_signals_worker.run_automated_signal_flow', side_effect=Exception("Test error")):
            worker_instance._refresh_signals()
            
            # Error should be published
            error_calls = [call for call in update_bus.publish.call_args_list 
                          if call[0][0].get('type') == 'signals_error']
            assert len(error_calls) > 0
    
    def test_execute_signal_error_handled(self, worker):
        """Test that signal execution errors are handled."""
        worker_instance, update_bus = worker
        
        # Create mock executor that will fail
        mock_executor = MagicMock()
        mock_executor.enabled = True
        mock_executor.execute_signal = MagicMock(side_effect=Exception("Execution failed"))
        
        worker_instance.signal_executor = mock_executor
        
        explicit_signal = {
            "signal": "BUY",
            "entries": [50000.0],
            "take_profits": {"tp1": 55000.0},
            "stop_loss": 48000.0,
        }
        
        # Should not raise
        worker_instance._execute_signal(explicit_signal, 1700000000000)


class TestAutomatedSignalsWorkerConfigUpdates:
    """Test configuration updates."""
    
    def test_update_config_changes_parameters(self):
        """Test that update_config changes worker parameters."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={"weights": {"technical": 0.1}},
            indicator_params={"rsi": {"period": 14}},
            signal_params={},
        )
        
        new_config = {"weights": {"technical": 0.5}}
        new_indicator_params = {"rsi": {"period": 21}}
        new_signal_params = {"param": "value"}
        
        with patch.object(worker, '_refresh_signals'):
            worker.update_config(new_config, new_indicator_params, new_signal_params)
        
        assert worker.signal_config_payload["weights"]["technical"] == 0.5
        assert worker.indicator_params["rsi"]["period"] == 21
        assert worker.signal_params["param"] == "value"
    
    def test_update_config_triggers_refresh(self):
        """Test that update_config triggers signal refresh."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
        )
        
        with patch.object(worker, '_refresh_signals') as mock_refresh:
            worker.update_config({}, {}, {})
            mock_refresh.assert_called_once()


class TestAutomatedSignalsWorkerSignalExecutorIntegration:
    """Test SignalExecutor integration."""
    
    def test_buy_signal_executed(self):
        """Test that BUY signals are executed."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        mock_executor = MagicMock()
        mock_executor.enabled = True
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
            signal_executor=mock_executor,
        )
        
        explicit_signal = {
            "signal": "BUY",
            "entries": [50000.0],
            "take_profits": {"tp1": 55000.0},
            "stop_loss": 48000.0,
        }
        
        worker._execute_signal(explicit_signal, 1700000000000)
        
        assert mock_executor.execute_signal.called
    
    def test_sell_signal_executed(self):
        """Test that SELL signals are executed."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        mock_executor = MagicMock()
        mock_executor.enabled = True
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
            signal_executor=mock_executor,
        )
        
        explicit_signal = {
            "signal": "SELL",
            "entries": [50000.0],
            "take_profits": {"tp1": 45000.0},
            "stop_loss": 52000.0,
        }
        
        worker._execute_signal(explicit_signal, 1700000000000)
        
        assert mock_executor.execute_signal.called
        
        # Check direction is SHORT
        call_args = mock_executor.execute_signal.call_args[0][0]
        assert call_args["direction"] == "SHORT"
    
    def test_hold_signal_not_executed(self):
        """Test that HOLD signals are not executed."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        mock_executor = MagicMock()
        mock_executor.enabled = True
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
            signal_executor=mock_executor,
        )
        
        explicit_signal = {
            "signal": "HOLD",
            "entries": [],
            "take_profits": {},
            "stop_loss": 0,
        }
        
        worker._execute_signal(explicit_signal, 1700000000000)
        
        assert not mock_executor.execute_signal.called
    
    def test_disabled_executor_skips_execution(self):
        """Test that disabled executor skips execution."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        mock_executor = MagicMock()
        mock_executor.enabled = False
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
            signal_executor=mock_executor,
        )
        
        explicit_signal = {
            "signal": "BUY",
            "entries": [50000.0],
            "take_profits": {"tp1": 55000.0},
            "stop_loss": 48000.0,
        }
        
        worker._execute_signal(explicit_signal, 1700000000000)
        
        assert not mock_executor.execute_signal.called


class TestAutomatedSignalsWorkerStateTransitions:
    """Test worker state transitions."""
    
    def test_is_running_reflects_state(self):
        """Test is_running method reflects WebSocket state."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={},
        )
        
        assert not worker.is_running()
        
        # Simulate started state
        worker.ws_client = MagicMock()
        assert worker.is_running()
        
        # Stop
        worker.stop()
        assert not worker.is_running()
    
    def test_double_start_noop(self):
        """Test that double start is a no-op."""
        from automated_signals_worker import AutomatedSignalsWorker
        
        update_bus = MagicMock()
        
        with patch('automated_signals_worker.BinanceWebSocketClient', MockWebSocketClient):
            worker = AutomatedSignalsWorker(
                symbol="BTCUSDT",
                timeframe="1h",
                update_bus=update_bus,
                signal_config_payload={},
                indicator_params={},
                signal_params={},
            )
            
            # Mock initial data load
            with patch('automated_signals_worker.get_binance_server_time_ms', return_value=1700000000000):
                worker.data_source = MagicMock()
                worker.data_source.load_candles.return_value = pd.DataFrame({
                    "ts": [1700000000000],
                    "open": [50000.0],
                    "high": [50100.0],
                    "low": [49900.0],
                    "close": [50050.0],
                    "volume": [100.0],
                })
                
                worker.start()
                first_ws = worker.ws_client
                
                worker.start()  # Second start
                
                assert worker.ws_client is first_ws  # Should not create new client
