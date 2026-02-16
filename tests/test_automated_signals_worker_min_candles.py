"""Tests for automated_signals_worker min_candles validation fix."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from automated_signals_worker import AutomatedSignalsWorker
from update_bus import UpdateBus


class TestAutomatedSignalsWorkerMinCandles:
    """Test AutomatedSignalsWorker min_candles validation."""

    def create_test_df(self, num_rows=100, start_ts=1700000000000):
        """Create a test DataFrame with candle data."""
        data = {
            "ts": np.arange(start_ts, start_ts + num_rows * 60000, 60000),
            "open": np.random.rand(num_rows) * 100 + 100,
            "high": np.random.rand(num_rows) * 100 + 105,
            "low": np.random.rand(num_rows) * 100 + 95,
            "close": np.random.rand(num_rows) * 100 + 100,
            "volume": np.random.rand(num_rows) * 1000,
        }
        df = pd.DataFrame(data)
        df["ts"] = df["ts"].astype("int64")
        return df

    def create_worker(self, df=None):
        """Create an AutomatedSignalsWorker instance."""
        update_bus = UpdateBus()
        worker = AutomatedSignalsWorker(
            symbol="BTCUSDT",
            timeframe="1h",
            start_ts=1700000000000,
            end_ts=1700003600000,
            update_bus=update_bus,
            signal_config_payload={},
            indicator_params={},
            signal_params={}
        )
        if df is not None:
            worker.df = df
        return worker

    def test_sufficient_candles_runs_signal_flow(self):
        """Test that signal flow runs when sufficient candles available."""
        # Create DF with 100 candles (more than required min_candles)
        df = self.create_test_df(num_rows=100)
        worker = self.create_worker(df=df)

        # Set up indicator params that require 30 candles
        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        # Mock run_automated_signal_flow
        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            mock_result = MagicMock()
            mock_result.candles = []
            mock_result.processed_payload = {}
            mock_result.explicit_signal = {}
            mock_flow.return_value = mock_result

            worker._process_last_closed_boundary(1700003600000)

            # Should have called run_automated_signal_flow
            mock_flow.assert_called_once()

    def test_insufficient_candles_skips_signal_flow(self):
        """Test that signal flow is skipped when insufficient candles available."""
        # Create DF with only 20 candles (less than required min_candles=30)
        df = self.create_test_df(num_rows=20)
        worker = self.create_worker(df=df)

        # Set up indicator params that require 30 candles
        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        # Mock run_automated_signal_flow
        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            worker._process_last_closed_boundary(1700003600000)

            # Should NOT have called run_automated_signal_flow
            mock_flow.assert_not_called()

    def test_min_candles_calculation_with_rsi(self):
        """Test min_candles calculation with RSI indicator."""
        df = self.create_test_df(num_rows=50)
        worker = self.create_worker(df=df)

        # RSI period = 20, so min_candles should be max(30, 20+2) = 22
        worker.indicator_params = {
            "rsi": {"period": 20},
            "atr": {"period": 10},
            "macd": {"slow": 12, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            mock_result = MagicMock()
            mock_result.candles = []
            mock_result.processed_payload = {}
            mock_result.explicit_signal = {}
            mock_flow.return_value = mock_result

            worker._process_last_closed_boundary(1700003600000)

            # Should have enough candles (50 > 22)
            mock_flow.assert_called_once()

    def test_min_candles_calculation_with_macd(self):
        """Test min_candles calculation with MACD indicator."""
        # Create DF with 40 candles (MACD slow=26, signal=9, so need at least 35)
        df = self.create_test_df(num_rows=40)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 10},
            "atr": {"period": 10},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            mock_result = MagicMock()
            mock_result.candles = []
            mock_result.processed_payload = {}
            mock_result.explicit_signal = {}
            mock_flow.return_value = mock_result

            worker._process_last_closed_boundary(1700003600000)

            # Should have enough candles (40 >= 35)
            mock_flow.assert_called_once()

    def test_min_candles_with_exactly_required_amount(self):
        """Test that signal flow runs with exactly the required min_candles."""
        # min_candles = max(30, 14+2, 14+2, 26+9) = max(30, 16, 16, 35) = 35
        df = self.create_test_df(num_rows=35)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            mock_result = MagicMock()
            mock_result.candles = []
            mock_result.processed_payload = {}
            mock_result.explicit_signal = {}
            mock_flow.return_value = mock_result

            worker._process_last_closed_boundary(1700003600000)

            # Should have exactly enough candles
            mock_flow.assert_called_once()

    def test_min_candles_one_less_than_required(self):
        """Test that signal flow is skipped with one less than required."""
        # min_candles = 35, but we only have 34
        df = self.create_test_df(num_rows=34)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            worker._process_last_closed_boundary(1700003600000)

            # Should NOT have called run_automated_signal_flow
            mock_flow.assert_not_called()

    def test_min_candles_with_none_dataframe(self):
        """Test that signal flow is skipped when df is None."""
        worker = self.create_worker(df=None)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            worker._process_last_closed_boundary(1700003600000)

            # Should NOT have called run_automated_signal_flow
            mock_flow.assert_not_called()

    def test_min_candles_logs_warning(self):
        """Test that a warning is logged when candles are insufficient."""
        # Create DF with only 10 candles
        df = self.create_test_df(num_rows=10)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            with patch('automated_signals_worker.logger') as mock_logger:
                worker._process_last_closed_boundary(1700003600000)

                # Should log warning about insufficient candles
                mock_logger.warning.assert_called_once()
                warning_msg = str(mock_logger.warning.call_args[0][0])
                assert "Insufficient candles" in warning_msg
                assert "10 available" in warning_msg
                assert "35 required" in warning_msg

    def test_min_candles_does_not_log_with_sufficient(self):
        """Test that no warning is logged when candles are sufficient."""
        df = self.create_test_df(num_rows=100)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            mock_result = MagicMock()
            mock_result.candles = []
            mock_result.processed_payload = {}
            mock_result.explicit_signal = {}
            mock_flow.return_value = mock_result

            with patch('automated_signals_worker.logger') as mock_logger:
                worker._process_last_closed_boundary(1700003600000)

                # Should NOT log warning
                mock_logger.warning.assert_not_called()

    def test_min_candles_returns_early_without_error(self):
        """Test that insufficient candles returns early without raising error."""
        df = self.create_test_df(num_rows=5)
        worker = self.create_worker(df=df)

        worker.indicator_params = {
            "rsi": {"period": 14},
            "atr": {"period": 14},
            "macd": {"slow": 26, "signal": 9}
        }

        with patch('automated_signals_worker.run_automated_signal_flow') as mock_flow:
            # Should not raise any exception
            worker._process_last_closed_boundary(1700003600000)

            # Should not call signal flow
            mock_flow.assert_not_called()

            # Should not publish update either
            updates = list(worker.update_bus.drain())
            assert len(updates) == 0
