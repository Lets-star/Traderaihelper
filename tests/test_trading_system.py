"""Tests for the trading system orchestrator."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from indicator_collector.trading_system import (
    MacroBlackoutConfig,
    TradingConfig,
    TradingState,
    TradingOrchestrator,
    MacroBlackoutFilter,
    create_trading_orchestrator,
    create_default_config,
)
from indicator_collector.trading_system.interfaces import (
    AnalyzerContext,
    SignalExplanation,
    TradingSignalPayload,
)
from indicator_collector.trading_system.position_manager import DiversificationGuard


class TestMacroBlackoutConfig:
    """Test MacroBlackoutConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MacroBlackoutConfig()
        
        assert config.enabled is True
        assert len(config.blackout_periods) == 2
        assert config.blackout_weekdays == {1, 2, 3, 4, 5}
        assert config.timezone == "UTC"
        assert len(config.custom_blackout_dates) == 0
    
    def test_serialization(self):
        """Test configuration serialization."""
        config = MacroBlackoutConfig(
            enabled=False,
            blackout_periods=[(time(9, 0), time(10, 0))],
            blackout_weekdays={1, 3, 5},
            timezone="EST",
            custom_blackout_dates={"2024-01-01", "2024-12-25"},
        )
        
        data = config.to_dict()
        restored = MacroBlackoutConfig.from_dict(data)
        
        assert restored.enabled == config.enabled
        assert len(restored.blackout_periods) == len(config.blackout_periods)
        assert restored.blackout_weekdays == config.blackout_weekdays
        assert restored.timezone == config.timezone
        assert restored.custom_blackout_dates == config.custom_blackout_dates


class TestMacroBlackoutFilter:
    """Test MacroBlackoutFilter class."""
    
    def test_disabled_filter(self):
        """Test disabled filter always returns False."""
        config = MacroBlackoutConfig(enabled=False)
        filter = MacroBlackoutFilter(config)
        
        # Any timestamp should return False when disabled
        assert filter.is_blackout_period(1640995200000) is False
    
    def test_weekday_filter(self):
        """Test weekday filtering."""
        config = MacroBlackoutConfig(
            enabled=True,
            blackout_weekdays={1, 2, 3, 4, 5},  # Monday-Friday
        )
        filter = MacroBlackoutFilter(config)
        
        # Monday (weekday 1)
        monday_timestamp = datetime(2024, 1, 1, 9, 0).timestamp() * 1000
        assert filter.is_blackout_period(monday_timestamp) is False  # No time period set
        
        # Saturday (weekday 6) - should not be filtered
        saturday_timestamp = datetime(2024, 1, 6, 9, 0).timestamp() * 1000
        assert filter.is_blackout_period(saturday_timestamp) is False
    
    def test_time_period_filter(self):
        """Test time period filtering."""
        config = MacroBlackoutConfig(
            enabled=True,
            blackout_periods=[(time(9, 0), time(10, 0))],
            blackout_weekdays={1, 2, 3, 4, 5},
        )
        filter = MacroBlackoutFilter(config)
        
        # Within blackout period
        blackout_time = datetime(2024, 1, 1, 9, 30).timestamp() * 1000  # Monday 9:30 AM
        assert filter.is_blackout_period(blackout_time) is True
        
        # Outside blackout period
        normal_time = datetime(2024, 1, 1, 11, 0).timestamp() * 1000  # Monday 11:00 AM
        assert filter.is_blackout_period(normal_time) is False
    
    def test_custom_date_filter(self):
        """Test custom date filtering."""
        config = MacroBlackoutConfig(
            enabled=True,
            custom_blackout_dates={"2024-01-01"},
        )
        filter = MacroBlackoutFilter(config)
        
        # Custom blackout date
        custom_date = datetime(2024, 1, 1, 14, 0).timestamp() * 1000
        assert filter.is_blackout_period(custom_date) is True
        
        # Normal date
        normal_date = datetime(2024, 1, 2, 14, 0).timestamp() * 1000
        assert filter.is_blackout_period(normal_date) is False


class TestTradingConfig:
    """Test TradingConfig class."""
    
    def test_default_config(self):
        """Test default trading configuration."""
        config = TradingConfig()
        
        assert config.account_balance == 10000.0
        assert config.mode == "live"
        assert config.data_dir == "./trading_data"
        assert config.state_file == "trading_state.json"
    
    def test_serialization(self):
        """Test configuration serialization."""
        config = TradingConfig(
            account_balance=50000.0,
            mode="backtest",
            data_dir="/tmp/trading",
        )
        
        data = config.to_dict()
        restored = TradingConfig.from_dict(data)
        
        assert restored.account_balance == config.account_balance
        assert restored.mode == config.mode
        assert restored.data_dir == config.data_dir


class TestTradingState:
    """Test TradingState class."""
    
    def test_default_state(self):
        """Test default trading state."""
        guard = DiversificationGuard()
        state = TradingState(diversification_guard=guard)
        
        assert state.diversification_guard == guard
        assert state.last_signal_timestamp is None
        assert state.last_optimization_timestamp is None
        assert state.signal_count == 0
        assert state.metadata == {}
    
    def test_serialization(self):
        """Test state serialization."""
        guard = DiversificationGuard()
        state = TradingState(
            diversification_guard=guard,
            last_signal_timestamp=1640995200000,
            signal_count=5,
            metadata={"test": "value"},
        )
        
        data = state.to_dict()
        restored = TradingState.from_dict(data)
        
        assert restored.last_signal_timestamp == state.last_signal_timestamp
        assert restored.signal_count == state.signal_count
        assert restored.metadata == state.metadata


class TestTradingOrchestrator:
    """Test TradingOrchestrator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return TradingConfig(
            data_dir=temp_dir,
            account_balance=10000.0,
            mode="backtest",
        )
    
    @pytest.fixture
    def orchestrator(self, config):
        """Create trading orchestrator for testing."""
        return TradingOrchestrator(config)
    
    @pytest.fixture
    def mock_collection_result(self):
        """Create mock collection result."""
        from indicator_collector.collector import CollectionResult, TimeframeSeries, TimeframeMetricSeries
        
        # Create mock data
        mock_payload = {
            "metadata": {"symbol": "BTCUSDT", "timeframe": "15m"},
            "latest": {
                "timestamp": 1640995200000,
                "open": 45000.0,
                "high": 45500.0,
                "low": 44500.0,
                "close": 45250.0,
                "volume": 1000.0,
                "trend_strength": 0.5,
                "rsi": 50.0,
                "macd": 100.0,
            },
            "advanced": {
                "volume_analysis": {"confidence": 0.7},
                "market_structure": {"trend": "up"},
            },
        }
        
        # Create minimal collection result
        result = Mock(spec=CollectionResult)
        result.payload = mock_payload
        
        return result
    
    def test_orchestrator_initialization(self, config, temp_dir):
        """Test orchestrator initialization."""
        orchestrator = TradingOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.data_dir == Path(temp_dir)
        assert orchestrator.state_file == Path(temp_dir) / config.state_file
        assert isinstance(orchestrator.state, TradingState)
        assert isinstance(orchestrator.macro_filter, MacroBlackoutFilter)
    
    def test_state_persistence(self, config, temp_dir):
        """Test state persistence."""
        orchestrator = TradingOrchestrator(config)
        
        # Modify state
        orchestrator.state.signal_count = 10
        orchestrator.state.last_signal_timestamp = 1640995200000
        
        # Save state
        orchestrator._save_state()
        
        # Create new orchestrator (should load saved state)
        new_orchestrator = TradingOrchestrator(config)
        assert new_orchestrator.state.signal_count == 10
        assert new_orchestrator.state.last_signal_timestamp == 1640995200000
    
    @patch('indicator_collector.trading_system.analyze_technical_factors')
    @patch('indicator_collector.trading_system.analyze_sentiment_factors')
    @patch('indicator_collector.trading_system.analyze_multitimeframe_factors')
    @patch('indicator_collector.trading_system.analyze_volume_orderbook')
    @patch('indicator_collector.trading_system.generate_trading_signal')
    def test_process_collection_result_normal(
        self,
        mock_generate_signal,
        mock_volume_analysis,
        mock_mtf_analysis,
        mock_sentiment_analysis,
        mock_technical_analysis,
        orchestrator,
        mock_collection_result,
    ):
        """Test normal processing of collection result."""
        # Mock analyzer responses
        mock_technical_analysis.return_value = [
            FactorScore("technical_analysis", 0.7, weight=1.0, metadata={"direction": "bullish"})
        ]
        mock_sentiment_analysis.return_value = [
            FactorScore("sentiment", 0.6, weight=0.8, metadata={"direction": "bullish"})
        ]
        mock_mtf_analysis.return_value = [
            FactorScore("multitimeframe_alignment", 0.5, weight=0.6, metadata={"direction": "bullish"})
        ]
        mock_volume_analysis.return_value = [
            FactorScore("volume_analysis", 0.8, weight=0.7, metadata={"direction": "bullish"})
        ]
        
        # Mock signal generation
        mock_signal = TradingSignalPayload(
            signal_type="BUY",
            confidence=0.75,
            timestamp=1640995200000,
            symbol="BTCUSDT",
            timeframe="15m",
        )
        mock_generate_signal.return_value = mock_signal
        
        # Process collection result
        result = orchestrator.process_collection_result(mock_collection_result)
        
        assert result.signal_type == "BUY"
        assert result.confidence == 0.75
        assert len(result.factors) >= 4  # All analyzer factors
        
        # Verify state was updated
        assert orchestrator.state.signal_count == 1
        assert orchestrator.state.last_signal_timestamp == 1640995200000
    
    def test_process_collection_result_blackout_period(self, orchestrator, mock_collection_result):
        """Test processing during macro blackout period."""
        # Set up blackout filter to always trigger
        orchestrator.macro_filter.config.enabled = True
        orchestrator.macro_filter.config.blackout_periods = [(time(0, 0), time(23, 59))]
        orchestrator.macro_filter.config.blackout_weekdays = {0, 1, 2, 3, 4, 5, 6}
        
        # Process collection result
        result = orchestrator.process_collection_result(mock_collection_result)
        
        assert result.signal_type == "NEUTRAL"
        assert result.confidence == 0.0
        assert "blackout" in result.explanation.primary_reason.lower()
    
    def test_signal_outcome_tracking(self, orchestrator):
        """Test signal outcome tracking."""
        from indicator_collector.trading_system import SignalOutcome
        
        outcome = SignalOutcome(
            signal_type="BUY",
            entry_price=45000.0,
            exit_price=46000.0,
            pnl_pct=2.22,
            success=True,
            timestamp=1640995200000,
            symbol="BTCUSDT",
        )
        
        orchestrator.add_signal_outcome(outcome)
        
        # Verify outcome was added
        assert len(orchestrator._recent_outcomes) == 1
        assert orchestrator._recent_outcomes[0] == outcome
        
        # Verify file was created
        assert orchestrator.outcomes_file.exists()
    
    def test_weight_optimization(self, orchestrator):
        """Test weight optimization."""
        current_weights = {
            "technical": 1.0,
            "sentiment": 0.8,
            "multitimeframe": 0.6,
            "volume": 0.7,
        }
        
        result = orchestrator.optimize_weights(current_weights)
        
        assert result is not None
        assert hasattr(result, 'can_meet_targets')
        assert hasattr(result, 'optimization_score')
    
    def test_performance_stats(self, orchestrator):
        """Test performance statistics retrieval."""
        stats = orchestrator.get_performance_stats()
        
        assert "kpis" in stats
        assert "state" in stats
        assert "recent_signals_count" in stats
        assert "recent_outcomes_count" in stats
    
    def test_historical_outcomes_loading(self, orchestrator):
        """Test loading historical outcomes."""
        # Create temporary outcomes file
        outcomes_data = [
            {
                "signal_type": "BUY",
                "entry_price": 45000.0,
                "exit_price": 46000.0,
                "pnl_pct": 2.22,
                "success": True,
                "timestamp": 1640995200000,
                "symbol": "BTCUSDT",
                "factors": [{"factor_name": "technical", "score": 0.8}],
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(outcomes_data, f)
            temp_file = f.name
        
        try:
            ingested = orchestrator.load_historical_outcomes(temp_file)
            assert ingested == 1
        finally:
            Path(temp_file).unlink()
    
    def test_state_export_import(self, orchestrator):
        """Test state export and import."""
        # Modify state
        orchestrator.state.signal_count = 15
        
        # Export state
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            orchestrator.export_state(export_file)
            assert Path(export_file).exists()
            
            # Create new orchestrator and import state
            new_config = TradingConfig(data_dir=orchestrator.data_dir)
            new_orchestrator = TradingOrchestrator(new_config)
            new_orchestrator.import_state(export_file)
            
            assert new_orchestrator.state.signal_count == 15
        finally:
            Path(export_file).unlink()


class TestTradingSystemIntegration:
    """Integration tests for the trading system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    def test_end_to_end_flow(self, temp_dir):
        """Test end-to-end trading flow."""
        config = TradingConfig(
            data_dir=temp_dir,
            account_balance=25000.0,
            mode="backtest",
        )
        
        orchestrator = create_trading_orchestrator(config)
        
        # Create mock collection result
        mock_payload = {
            "metadata": {"symbol": "ETHUSDT", "timeframe": "1h"},
            "latest": {
                "timestamp": 1640995200000,
                "open": 3000.0,
                "high": 3100.0,
                "low": 2950.0,
                "close": 3050.0,
                "volume": 5000.0,
                "trend_strength": 0.8,
                "rsi": 65.0,
                "macd": 150.0,
            },
            "advanced": {
                "volume_analysis": {"confidence": 0.9},
                "market_structure": {"trend": "strong_up"},
            },
        }
        
        mock_result = Mock()
        mock_result.payload = mock_payload
        
        # Process multiple signals
        signals = []
        for i in range(5):
            with patch('indicator_collector.trading_system.analyze_technical_factors') as mock_tech, \
                 patch('indicator_collector.trading_system.analyze_sentiment_factors') as mock_sent, \
                 patch('indicator_collector.trading_system.analyze_multitimeframe_factors') as mock_mtf, \
                 patch('indicator_collector.trading_system.analyze_volume_orderbook') as mock_vol, \
                 patch('indicator_collector.trading_system.generate_trading_signal') as mock_signal:
                
                # Mock analyzer responses
                mock_tech.return_value = [Mock(factor_name="technical", score=0.8, weight=1.0)]
                mock_sent.return_value = [Mock(factor_name="sentiment", score=0.7, weight=0.8)]
                mock_mtf.return_value = [Mock(factor_name="multitimeframe", score=0.6, weight=0.6)]
                mock_vol.return_value = [Mock(factor_name="volume", score=0.9, weight=0.7)]
                
                # Mock signal generation
                signal_payload = TradingSignalPayload(
                    signal_type="BUY" if i % 2 == 0 else "SELL",
                    confidence=0.75 + (i * 0.05),
                    timestamp=1640995200000 + (i * 3600000),
                    symbol="ETHUSDT",
                    timeframe="1h",
                )
                mock_signal.return_value = signal_payload
                
                signal = orchestrator.process_collection_result(mock_result)
                signals.append(signal)
        
        # Verify signals were generated
        assert len(signals) == 5
        assert orchestrator.state.signal_count == 5
        
        # Add some outcomes
        from indicator_collector.trading_system import SignalOutcome
        for i, signal in enumerate(signals):
            outcome = SignalOutcome(
                signal_type=signal.signal_type,
                entry_price=3000.0 + (i * 10),
                exit_price=3020.0 + (i * 10),
                pnl_pct=0.67,
                success=i < 3,  # First 3 successful
                timestamp=signal.timestamp + 3600000,
                symbol=signal.symbol,
            )
            orchestrator.add_signal_outcome(outcome)
        
        # Get performance stats
        stats = orchestrator.get_performance_stats()
        assert stats["recent_signals_count"] == 5
        assert stats["recent_outcomes_count"] == 5
        
        # Run optimization
        current_weights = {"technical": 1.0, "sentiment": 0.8, "multitimeframe": 0.6, "volume": 0.7}
        result = orchestrator.optimize_weights(current_weights)
        assert result is not None
    
    def test_diversification_enforcement(self, temp_dir):
        """Test diversification enforcement."""
        config = TradingConfig(
            data_dir=temp_dir,
            account_balance=10000.0,
            mode="backtest",
        )
        
        # Set low diversification limits for testing
        config.position_config.max_concurrent_same_direction = 2
        config.position_config.max_concurrent_total = 3
        
        orchestrator = create_trading_orchestrator(config)
        
        # Create mock collection result
        mock_payload = {
            "metadata": {"symbol": "BTCUSDT", "timeframe": "15m"},
            "latest": {
                "timestamp": 1640995200000,
                "open": 45000.0,
                "high": 45500.0,
                "low": 44500.0,
                "close": 45250.0,
                "volume": 1000.0,
                "trend_strength": 0.9,
                "rsi": 70.0,
                "macd": 200.0,
            },
            "advanced": {
                "volume_analysis": {"confidence": 0.9},
                "market_structure": {"trend": "up"},
            },
        }
        
        mock_result = Mock()
        mock_result.payload = mock_payload
        
        # Add multiple positions in same direction
        positions_added = 0
        for i in range(5):
            with patch('indicator_collector.trading_system.analyze_technical_factors') as mock_tech, \
                 patch('indicator_collector.trading_system.analyze_sentiment_factors') as mock_sent, \
                 patch('indicator_collector.trading_system.analyze_multitimeframe_factors') as mock_mtf, \
                 patch('indicator_collector.trading_system.analyze_volume_orderbook') as mock_vol, \
                 patch('indicator_collector.trading_system.generate_trading_signal') as mock_signal, \
                 patch('indicator_collector.trading_system.create_position_plan') as mock_position:
                
                # Mock analyzer responses
                mock_tech.return_value = [Mock(factor_name="technical", score=0.9, weight=1.0)]
                mock_sent.return_value = [Mock(factor_name="sentiment", score=0.8, weight=0.8)]
                mock_mtf.return_value = [Mock(factor_name="multitimeframe", score=0.7, weight=0.6)]
                mock_vol.return_value = [Mock(factor_name="volume", score=0.9, weight=0.7)]
                
                # Mock signal generation
                signal_payload = TradingSignalPayload(
                    signal_type="BUY",  # All same direction
                    confidence=0.8,
                    timestamp=1640995200000 + (i * 3600000),
                    symbol="BTCUSDT",
                    timeframe="15m",
                )
                mock_signal.return_value = signal_payload
                
                # Mock position plan - succeed for first 2, fail for rest
                from indicator_collector.trading_system.position_manager import PositionManagerResult
                if i < 2:
                    mock_position.return_value = PositionManagerResult(
                        can_trade=True,
                        position_plan=Mock(direction="long"),
                        holding_horizon_bars=20,
                    )
                    positions_added += 1
                else:
                    mock_position.return_value = PositionManagerResult(
                        can_trade=False,
                        cancellation_reasons=["Diversification limit exceeded"],
                    )
                
                signal = orchestrator.process_collection_result(mock_result)
                
                if i >= 2:
                    assert signal.signal_type == "NEUTRAL"
                    assert "diversification" in signal.explanation.primary_reason.lower()
        
        # Verify diversification limits were enforced
        assert positions_added == 2
        assert len(orchestrator.state.diversification_guard.get_positions_by_direction("long")) == 2


if __name__ == "__main__":
    pytest.main([__file__])