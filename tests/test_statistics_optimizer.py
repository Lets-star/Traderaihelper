"""Tests for statistics optimizer module."""

import unittest
from unittest.mock import patch, MagicMock

from indicator_collector.trading_system.statistics_optimizer import (
    SignalOutcome,
    PerformanceKPIs,
    WeightAdjustment,
    OptimizationResult,
    StatsOptimizerConfig,
    StatisticsOptimizer,
    create_stats_optimizer,
    create_synthetic_outcomes,
)


class TestSignalOutcome:
    """Test SignalOutcome dataclass."""
    
    def test_signal_outcome_creation(self):
        """Test creating a signal outcome."""
        outcome = SignalOutcome(
            signal_type="BUY",
            entry_price=100.0,
            exit_price=105.0,
            entry_timestamp=1640995200,
            exit_timestamp=1641081600,
            pnl_pct=5.0,
            holding_bars=24,
            success=True,
            factors=[{"factor_name": "technical", "score": 0.8}],
        )
        
        assert outcome.signal_type == "BUY"
        assert outcome.entry_price == 100.0
        assert outcome.exit_price == 105.0
        assert outcome.pnl_pct == 5.0
        assert outcome.success is True
        assert len(outcome.factors) == 1
    
    def test_signal_outcome_serialization(self):
        """Test signal outcome serialization."""
        outcome = SignalOutcome(
            signal_type="SELL",
            entry_price=200.0,
            exit_price=190.0,
            pnl_pct=-5.0,
            success=False,
        )
        
        data = outcome.to_dict()
        assert data["signal_type"] == "SELL"
        assert data["entry_price"] == 200.0
        assert data["exit_price"] == 190.0
        assert data["pnl_pct"] == -5.0
        assert data["success"] is False
        
        # Test deserialization
        restored = SignalOutcome.from_dict(data)
        assert restored.signal_type == "SELL"
        assert restored.entry_price == 200.0
        assert restored.exit_price == 190.0
        assert restored.pnl_pct == -5.0
        assert restored.success is False


class TestPerformanceKPIs:
    """Test PerformanceKPIs dataclass."""
    
    def test_kpis_creation(self):
        """Test creating KPIs."""
        kpis = PerformanceKPIs(
            total_signals=100,
            profitable_signals=60,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
        )
        
        assert kpis.total_signals == 100
        assert kpis.profitable_signals == 60
        assert kpis.losing_signals == 0  # Default
        assert kpis.win_rate == 0.6
        assert kpis.profit_factor == 1.5
        assert kpis.sharpe_ratio == 1.2
    
    def test_kpis_serialization(self):
        """Test KPIs serialization."""
        kpis = PerformanceKPIs(
            total_signals=50,
            profitable_signals=30,
            losing_signals=20,
            win_rate=0.6,
            profit_factor=1.8,
        )
        
        data = kpis.to_dict()
        assert data["total_signals"] == 50
        assert data["profitable_signals"] == 30
        assert data["losing_signals"] == 20
        assert data["win_rate"] == 0.6
        assert data["profit_factor"] == 1.8


class TestWeightAdjustment:
    """Test WeightAdjustment dataclass."""
    
    def test_weight_adjustment_creation(self):
        """Test creating weight adjustment."""
        adjustment = WeightAdjustment(
            factor_name="technical",
            current_weight=1.0,
            suggested_weight=1.2,
            adjustment_reason="Strong performance",
            confidence=0.8,
            performance_impact=0.5,
        )
        
        assert adjustment.factor_name == "technical"
        assert adjustment.current_weight == 1.0
        assert adjustment.suggested_weight == 1.2
        assert adjustment.adjustment_reason == "Strong performance"
        assert adjustment.confidence == 0.8
        assert adjustment.performance_impact == 0.5
    
    def test_weight_adjustment_serialization(self):
        """Test weight adjustment serialization."""
        adjustment = WeightAdjustment(
            factor_name="sentiment",
            current_weight=0.8,
            suggested_weight=0.6,
            adjustment_reason="Poor performance",
            confidence=0.7,
            performance_impact=-0.3,
        )
        
        data = adjustment.to_dict()
        assert data["factor_name"] == "sentiment"
        assert data["current_weight"] == 0.8
        assert data["suggested_weight"] == 0.6
        assert data["adjustment_reason"] == "Poor performance"
        assert data["confidence"] == 0.7
        assert data["performance_impact"] == -0.3


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creating optimization result."""
        current_kpis = PerformanceKPIs(total_signals=50, win_rate=0.5)
        target_kpis = PerformanceKPIs(total_signals=0, win_rate=0.6)
        projected_kpis = PerformanceKPIs(total_signals=0, win_rate=0.58)
        
        adjustment = WeightAdjustment(
            factor_name="technical",
            current_weight=1.0,
            suggested_weight=1.1,
            adjustment_reason="Good performance",
            confidence=0.7,
            performance_impact=0.2,
        )
        
        result = OptimizationResult(
            current_kpis=current_kpis,
            target_kpis=target_kpis,
            projected_kpis=projected_kpis,
            weight_adjustments=[adjustment],
            can_meet_targets=True,
            optimization_score=85.0,
            recommendations=["Consider weight adjustment"],
        )
        
        assert result.current_kpis == current_kpis
        assert result.target_kpis == target_kpis
        assert result.projected_kpis == projected_kpis
        assert len(result.weight_adjustments) == 1
        assert result.can_meet_targets is True
        assert result.optimization_score == 85.0
        assert len(result.recommendations) == 1


class TestStatsOptimizerConfig:
    """Test StatsOptimizerConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating config with defaults."""
        config = StatsOptimizerConfig()
        
        assert config.min_win_rate_target == 0.55
        assert config.min_profit_factor_target == 1.5
        assert config.max_drawdown_target == 0.20
        assert config.min_sharpe_target == 1.0
        assert config.min_signals_for_analysis == 30
        assert config.weight_adjustment_factor == 0.1
        assert config.factor_performance_window == 50
        assert config.enable_volatility_filter is True
        assert config.max_volatility_pct == 2.0
    
    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = StatsOptimizerConfig(
            min_win_rate_target=0.6,
            min_profit_factor_target=2.0,
            min_signals_for_analysis=50,
            weight_adjustment_factor=0.15,
        )
        
        assert config.min_win_rate_target == 0.6
        assert config.min_profit_factor_target == 2.0
        assert config.min_signals_for_analysis == 50
        assert config.weight_adjustment_factor == 0.15


class TestStatisticsOptimizer:
    """Test StatisticsOptimizer class."""
    
    def test_optimizer_creation(self):
        """Test creating optimizer."""
        optimizer = StatisticsOptimizer()
        assert optimizer.config.min_win_rate_target == 0.55
        assert len(optimizer._outcomes) == 0
        assert len(optimizer._factor_performance) == 0
    
    def test_optimizer_with_custom_config(self):
        """Test creating optimizer with custom config."""
        config = StatsOptimizerConfig(min_win_rate_target=0.7)
        optimizer = StatisticsOptimizer(config)
        assert optimizer.config.min_win_rate_target == 0.7
    
    def test_add_signal_outcome(self):
        """Test adding signal outcome."""
        optimizer = StatisticsOptimizer()
        
        outcome = SignalOutcome(
            signal_type="BUY",
            entry_price=100.0,
            exit_price=105.0,
            pnl_pct=5.0,
            success=True,
            factors=[
                {"factor_name": "technical", "score": 0.8},
                {"factor_name": "sentiment", "score": 0.6},
            ],
        )
        
        optimizer.add_signal_outcome(outcome)
        
        assert len(optimizer._outcomes) == 1
        assert "technical" in optimizer._factor_performance
        assert "sentiment" in optimizer._factor_performance
        assert optimizer._factor_performance["technical"] == [5.0]
        assert optimizer._factor_performance["sentiment"] == [5.0]
    
    def test_factor_performance_window(self):
        """Test factor performance window limiting."""
        config = StatsOptimizerConfig(factor_performance_window=3)
        optimizer = StatisticsOptimizer(config)
        
        # Add more outcomes than window size
        for i in range(5):
            outcome = SignalOutcome(
                signal_type="BUY",
                entry_price=100.0,
                pnl_pct=float(i),
                success=True,
                factors=[{"factor_name": "technical", "score": 0.8}],
            )
            optimizer.add_signal_outcome(outcome)
        
        # Should only keep last 3 performances
        assert len(optimizer._factor_performance["technical"]) == 3
        assert optimizer._factor_performance["technical"] == [2.0, 3.0, 4.0]
    
    def test_calculate_kpis_empty(self):
        """Test calculating KPIs with no outcomes."""
        optimizer = StatisticsOptimizer()
        kpis = optimizer.calculate_kpis()
        
        assert kpis.total_signals == 0
        assert kpis.win_rate == 0.0
        assert kpis.profit_factor == 0.0
        assert kpis.sharpe_ratio == 0.0
    
    def test_calculate_kpis_with_outcomes(self):
        """Test calculating KPIs with outcomes."""
        optimizer = StatisticsOptimizer()
        
        # Add test outcomes
        outcomes = [
            SignalOutcome(signal_type="BUY", entry_price=100.0, pnl_pct=5.0, success=True),
            SignalOutcome(signal_type="SELL", entry_price=100.0, pnl_pct=-3.0, success=False),
            SignalOutcome(signal_type="BUY", entry_price=100.0, pnl_pct=7.0, success=True),
        ]
        
        for outcome in outcomes:
            optimizer.add_signal_outcome(outcome)
        
        kpis = optimizer.calculate_kpis()
        
        assert kpis.total_signals == 3
        assert kpis.profitable_signals == 2
        assert kpis.losing_signals == 1
        assert kpis.win_rate == 2/3
        assert kpis.avg_profit_pct == 6.0  # (5 + 7) / 2
        assert kpis.avg_loss_pct == -3.0
        assert kpis.largest_win_pct == 7.0
        assert kpis.largest_loss_pct == -3.0
        assert kpis.total_return_pct == 9.0  # 5 - 3 + 7
    
    def test_suggest_weight_adjustments_no_data(self):
        """Test weight adjustments with insufficient data."""
        optimizer = StatisticsOptimizer()
        current_weights = {"technical": 1.0, "sentiment": 0.8}
        
        adjustments = optimizer.suggest_weight_adjustments(current_weights)
        assert len(adjustments) == 0
    
    def test_suggest_weight_adjustments_good_performer(self):
        """Test weight adjustments for good performing factor."""
        optimizer = StatisticsOptimizer()
        
        # Add outcomes with good technical factor performance
        for i in range(10):
            outcome = SignalOutcome(
                signal_type="BUY",
                entry_price=100.0,
                pnl_pct=2.0,  # Consistently good
                success=True,
                factors=[{"factor_name": "technical", "score": 0.8}],
            )
            optimizer.add_signal_outcome(outcome)
        
        current_weights = {"technical": 1.0}
        adjustments = optimizer.suggest_weight_adjustments(current_weights)
        
        assert len(adjustments) == 1
        adjustment = adjustments[0]
        assert adjustment.factor_name == "technical"
        assert adjustment.current_weight == 1.0
        assert adjustment.suggested_weight > 1.0
        assert "Strong average performance" in adjustment.adjustment_reason
        assert adjustment.confidence > 0
    
    def test_suggest_weight_adjustments_poor_performer(self):
        """Test weight adjustments for poor performing factor."""
        optimizer = StatisticsOptimizer()
        
        # Add outcomes with poor sentiment factor performance
        for i in range(10):
            outcome = SignalOutcome(
                signal_type="BUY",
                entry_price=100.0,
                pnl_pct=-1.0,  # Consistently poor
                success=False,
                factors=[{"factor_name": "sentiment", "score": -0.5}],
            )
            optimizer.add_signal_outcome(outcome)
        
        current_weights = {"sentiment": 1.0}
        adjustments = optimizer.suggest_weight_adjustments(current_weights)
        
        assert len(adjustments) == 1
        adjustment = adjustments[0]
        assert adjustment.factor_name == "sentiment"
        assert adjustment.current_weight == 1.0
        assert adjustment.suggested_weight < 1.0
        assert "Poor average performance" in adjustment.adjustment_reason
        assert adjustment.confidence > 0
    
    def test_optimize_weights_insufficient_data(self):
        """Test optimization with insufficient data."""
        optimizer = StatisticsOptimizer()
        current_weights = {"technical": 1.0}
        
        result = optimizer.optimize_weights(current_weights)
        
        assert result.can_meet_targets is False
        assert result.optimization_score == 0.0
        assert "Need at least 30 signals" in result.recommendations[0]
    
    def test_optimize_weights_with_data(self):
        """Test optimization with sufficient data."""
        optimizer = StatisticsOptimizer()
        
        # Add enough outcomes
        for i in range(35):
            outcome = SignalOutcome(
                signal_type="BUY",
                entry_price=100.0,
                pnl_pct=1.0 if i % 3 != 0 else -0.5,  # ~67% win rate
                success=i % 3 != 0,
                factors=[{"factor_name": "technical", "score": 0.7}],
            )
            optimizer.add_signal_outcome(outcome)
        
        current_weights = {"technical": 1.0}
        result = optimizer.optimize_weights(current_weights)
        
        assert result.current_kpis.total_signals == 35
        assert result.optimization_score > 0
        assert isinstance(result.can_meet_targets, bool)
        assert isinstance(result.recommendations, list)
    
    def test_ingest_historical_logs(self):
        """Test ingesting historical logs."""
        optimizer = StatisticsOptimizer()
        
        log_data = [
            {
                "signal": {
                    "signal_type": "BUY",
                    "entry_price": 100.0,
                    "timestamp": 1640995200,
                    "factors": [{"factor_name": "technical", "score": 0.8}],
                },
                "outcome": {
                    "exit_price": 105.0,
                    "pnl_pct": 5.0,
                    "success": True,
                    "holding_bars": 24,
                },
            },
            {
                "signal": {
                    "signal_type": "SELL",
                    "entry_price": 200.0,
                    "timestamp": 1641081600,
                    "factors": [{"factor_name": "sentiment", "score": -0.3}],
                },
                "outcome": {
                    "exit_price": 190.0,
                    "pnl_pct": -5.0,
                    "success": False,
                    "holding_bars": 12,
                },
            },
        ]
        
        ingested_count = optimizer.ingest_historical_logs(log_data)
        
        assert ingested_count == 2
        assert len(optimizer._outcomes) == 2
        assert optimizer._outcomes[0].signal_type == "BUY"
        assert optimizer._outcomes[1].signal_type == "SELL"
    
    def test_ingest_historical_logs_invalid_entries(self):
        """Test ingesting logs with invalid entries."""
        optimizer = StatisticsOptimizer()
        
        log_data = [
            {"invalid": "entry"},  # No signal data
            {
                "signal": {
                    "signal_type": "BUY",
                    "entry_price": "invalid",  # Invalid price
                },
            },
            {
                "signal": {
                    "signal_type": "SELL",
                    "entry_price": 100.0,
                    "timestamp": 1640995200,
                },
                "outcome": {
                    "exit_price": 95.0,
                    "pnl_pct": -5.0,
                    "success": False,
                },
            },
        ]
        
        ingested_count = optimizer.ingest_historical_logs(log_data)
        
        # Should only ingest the valid entry
        assert ingested_count == 1
        assert len(optimizer._outcomes) == 1
        assert optimizer._outcomes[0].signal_type == "SELL"
    
    def test_get_optimization_stats(self):
        """Test getting optimization stats."""
        optimizer = StatisticsOptimizer()
        
        # Add test outcomes
        for i in range(10):
            outcome = SignalOutcome(
                signal_type="BUY",
                entry_price=100.0,
                pnl_pct=2.0 if i % 2 == 0 else -1.0,
                success=i % 2 == 0,
            )
            optimizer.add_signal_outcome(outcome)
        
        stats = optimizer.get_optimization_stats()
        
        assert stats.total_signals == 10
        assert stats.profitable_signals == 5
        assert stats.losing_signals == 5
        assert stats.backtest_win_rate == 0.5
        assert stats.avg_profit_pct == 2.0
        assert stats.avg_loss_pct == -1.0
    
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        optimizer = StatisticsOptimizer()
        
        # Test with positive returns
        returns = [1.0, 2.0, 1.5, 3.0, 2.0]
        drawdown = optimizer._calculate_max_drawdown(returns)
        assert drawdown == 1.0  # Peak 4.0, low 3.0
        
        # Test with negative returns
        returns = [-1.0, -2.0, -0.5, -1.5]
        drawdown = optimizer._calculate_max_drawdown(returns)
        assert drawdown == 2.5  # Peak 0, low -2.5
        
        # Test with empty returns
        drawdown = optimizer._calculate_max_drawdown([])
        assert drawdown == 0.0
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        optimizer = StatisticsOptimizer()
        
        # Test with positive returns
        returns = [1.0, 2.0, 1.5, 3.0, 2.0]
        sharpe = optimizer._calculate_sharpe_ratio(returns)
        assert sharpe > 0
        
        # Test with negative returns
        returns = [-1.0, -2.0, -0.5, -1.5]
        sharpe = optimizer._calculate_sharpe_ratio(returns)
        assert sharpe < 0
        
        # Test with single return
        sharpe = optimizer._calculate_sharpe_ratio([1.0])
        assert sharpe == 0.0
        
        # Test with empty returns
        sharpe = optimizer._calculate_sharpe_ratio([])
        assert sharpe == 0.0
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        optimizer = StatisticsOptimizer()
        
        # Test with mixed returns
        returns = [1.0, 2.0, -0.5, 3.0, -1.0]
        sortino = optimizer._calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test with only positive returns
        returns = [1.0, 2.0, 1.5, 3.0]
        sortino = optimizer._calculate_sortino_ratio(returns)
        assert sortino == float('inf')
        
        # Test with single return
        sortino = optimizer._calculate_sortino_ratio([1.0])
        assert sortino == 0.0
        
        # Test with empty returns
        sortino = optimizer._calculate_sortino_ratio([])
        assert sortino == 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_stats_optimizer(self):
        """Test creating stats optimizer."""
        optimizer = create_stats_optimizer()
        assert isinstance(optimizer, StatisticsOptimizer)
        assert optimizer.config.min_win_rate_target == 0.55
        
        # Test with custom config
        config = StatsOptimizerConfig(min_win_rate_target=0.7)
        optimizer = create_stats_optimizer(config)
        assert optimizer.config.min_win_rate_target == 0.7
    
    def test_create_synthetic_outcomes(self):
        """Test creating synthetic outcomes."""
        outcomes = create_synthetic_outcomes(count=50)
        
        assert len(outcomes) == 50
        
        # Check structure of first outcome
        outcome = outcomes[0]
        assert isinstance(outcome, SignalOutcome)
        assert outcome.signal_type in ["BUY", "SELL", "NEUTRAL"]
        assert outcome.entry_price > 0
        assert outcome.exit_price is not None
        assert outcome.pnl_pct is not None
        assert outcome.success is not None
        assert outcome.holding_bars is not None
        assert len(outcome.factors) > 0
        
        # Check that we have both wins and losses
        wins = sum(1 for o in outcomes if o.success)
        losses = sum(1 for o in outcomes if not o.success)
        assert wins > 0
        assert losses > 0
    
    def test_create_synthetic_outcomes_default_count(self):
        """Test creating synthetic outcomes with default count."""
        outcomes = create_synthetic_outcomes()
        assert len(outcomes) == 100


class TestIntegration:
    """Integration tests for statistics optimizer."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create optimizer
        config = StatsOptimizerConfig(
            min_signals_for_analysis=20,
            min_win_rate_target=0.6,
            min_profit_factor_target=1.5,
        )
        optimizer = StatisticsOptimizer(config)
        
        # Add synthetic data
        outcomes = create_synthetic_outcomes(count=50)
        for outcome in outcomes:
            optimizer.add_signal_outcome(outcome)
        
        # Calculate current KPIs
        current_kpis = optimizer.calculate_kpis()
        assert current_kpis.total_signals == 50
        
        # Get weight adjustments
        current_weights = {
            "technical": 1.0,
            "sentiment": 0.8,
            "volume": 1.2,
            "multitimeframe": 0.9,
        }
        adjustments = optimizer.suggest_weight_adjustments(current_weights)
        assert isinstance(adjustments, list)
        
        # Run optimization
        result = optimizer.optimize_weights(current_weights)
        
        assert isinstance(result, OptimizationResult)
        assert result.current_kpis.total_signals == 50
        assert isinstance(result.weight_adjustments, list)
        assert isinstance(result.can_meet_targets, bool)
        assert isinstance(result.optimization_score, float)
        assert isinstance(result.recommendations, list)
        
        # Get optimization stats
        stats = optimizer.get_optimization_stats()
        assert stats.total_signals == 50
    
    def test_historical_data_integration(self):
        """Test integration with historical data."""
        optimizer = StatisticsOptimizer()
        
        # Create historical log format data
        historical_data = []
        for i in range(30):
            log_entry = {
                "signal": {
                    "signal_type": "BUY" if i % 2 == 0 else "SELL",
                    "entry_price": 100.0 + i,
                    "timestamp": 1640995200 + i * 3600,
                    "factors": [
                        {"factor_name": "technical", "score": 0.5 + (i % 3) * 0.2},
                        {"factor_name": "sentiment", "score": -0.2 + (i % 2) * 0.4},
                    ],
                },
                "outcome": {
                    "exit_price": 100.0 + i + (1 if i % 2 == 0 else -1),
                    "pnl_pct": 1.0 if i % 2 == 0 else -1.0,
                    "success": i % 2 == 0,
                    "holding_bars": 10 + i % 20,
                },
            }
            historical_data.append(log_entry)
        
        # Ingest historical data
        ingested = optimizer.ingest_historical_logs(historical_data)
        assert ingested == 30
        
        # Run optimization
        current_weights = {"technical": 1.0, "sentiment": 1.0}
        result = optimizer.optimize_weights(current_weights)
        
        assert result.current_kpis.total_signals == 30
        assert result.current_kpis.win_rate == 0.5  # 15 wins out of 30
        assert len(result.recommendations) > 0


if __name__ == "__main__":
    unittest.main()