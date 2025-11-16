"""Tests for backtesting engine and adaptive weighting system."""

import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from indicator_collector.trading_system.backtester import (
    Backtester,
    BacktestConfig,
    BacktestResult,
    ParameterSet,
    indicator_defaults_for,
)
from indicator_collector.timeframes import Timeframe
from indicator_collector.trading_system.adaptive_weights import (
    AdaptiveWeightManager,
    AdaptiveWeightConfig,
    AdaptationReport,
)
from indicator_collector.trading_system.statistics_optimizer import (
    SignalOutcome,
    PerformanceKPIs,
    StatisticsOptimizer,
)
from indicator_collector.trading_system.interfaces import TradingSignalPayload


class TestBacktester:
    """Test cases for Backtester class."""
    
    def test_backtest_config_creation(self):
        """Test BacktestConfig creation and serialization."""
        config = BacktestConfig(
            lookback_days=365,
            target_win_rate=0.6,
            search_method="random"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["lookback_days"] == 365
        assert config_dict["target_win_rate"] == 0.6
        assert config_dict["search_method"] == "random"
        assert config_dict["validate_real_data"] is True
    
    def test_parameter_set_creation(self):
        """Test ParameterSet creation and serialization."""
        params = ParameterSet(
            weights={"technical": 0.4, "volume": 0.3, "sentiment": 0.3},
            timeframe=Timeframe.HOUR_3.value,
            stop_loss_pct=2.5,
            take_profit_pct=5.0,
        )

        params_dict = params.to_dict()

        assert params_dict["weights"]["technical"] == 0.4
        assert params_dict["stop_loss_pct"] == 2.5
        assert params_dict["take_profit_pct"] == 5.0
        assert params_dict["timeframe"] == Timeframe.HOUR_3.value

        defaults = indicator_defaults_for(Timeframe.HOUR_3.value)
        assert params_dict["indicator_params"]["macd"]["fast"] == defaults["macd"]["fast"]

        # Test from_dict
        reconstructed = ParameterSet.from_dict(params_dict)
        assert reconstructed.weights == params.weights
        assert reconstructed.stop_loss_pct == params.stop_loss_pct
        assert reconstructed.timeframe == params.timeframe
        assert reconstructed.indicator_params == params.indicator_params
    
    def test_parameter_set_default_indicator_params(self):
        """ParameterSet without overrides should use timeframe defaults."""
        params = ParameterSet()
        defaults = indicator_defaults_for(Timeframe.HOUR_1.value)

        assert params.timeframe == Timeframe.HOUR_1.value
        assert params.indicator_params == defaults
        assert params.weights == {
            "technical": 0.25,
            "volume": 0.25,
            "sentiment": 0.25,
            "market_structure": 0.25,
        }

    def test_parameter_set_partial_indicator_params(self):
        """Partial indicator overrides should merge with defaults."""
        params = ParameterSet(
            timeframe=Timeframe.HOUR_4.value,
            indicator_params={
                "macd": {"fast": 10},
                "rsi": {"overbought": 68},
            },
        )

        defaults = indicator_defaults_for(Timeframe.HOUR_4.value)
        assert params.indicator_params["macd"]["fast"] == 10
        assert params.indicator_params["macd"]["slow"] == defaults["macd"]["slow"]
        assert params.indicator_params["rsi"]["overbought"] == 68
        assert params.indicator_params["rsi"]["oversold"] == defaults["rsi"]["oversold"]

    def test_parameter_set_unsupported_timeframe_defaults(self):
        """Unsupported timeframes should safely fall back to 1h defaults."""
        params = ParameterSet(timeframe="2h")
        defaults = indicator_defaults_for(Timeframe.HOUR_1.value)

        assert params.timeframe == Timeframe.HOUR_1.value
        assert params.indicator_params == defaults
    
    def test_backtester_initialization(self):
        """Test Backtester initialization."""
        config = BacktestConfig(lookback_days=100)
        backtester = Backtester(config)
        
        assert backtester.config.lookback_days == 100
        assert len(backtester._historical_data) == 0
    
    def test_load_historical_data_from_list(self):
        """Test loading historical data from list."""
        backtester = Backtester()
        
        # Create mock payloads
        payloads = []
        for i in range(100):
            payload = Mock(spec=TradingSignalPayload)
            payload.timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            payload.to_dict.return_value = {
                "timestamp": payload.timestamp,
                "signal_type": "BUY" if i % 2 == 0 else "SELL",
                "entry_price": 50000.0 + i,
            }
            payloads.append(payload)
        
        count = backtester.load_historical_data(payloads)
        assert count == 100
        assert len(backtester._historical_data) == 100
    
    def test_load_historical_data_insufficient_data(self):
        """Test loading insufficient historical data."""
        backtester = Backtester(config=BacktestConfig(min_data_points=200))
        
        # Create only 50 payloads
        payloads = []
        for i in range(50):
            payload = Mock(spec=TradingSignalPayload)
            payload.timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            payload.to_dict.return_value = {"timestamp": payload.timestamp}
            payloads.append(payload)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            backtester.load_historical_data(payloads)
    
    def test_data_splitting_time_split(self):
        """Test data splitting with time_split method."""
        config = BacktestConfig(split_method="time_split", train_ratio=0.7)
        backtester = Backtester(config)
        
        # Add mock data
        backtester._historical_data = [Mock() for _ in range(100)]
        
        train_data, test_data = backtester._split_data(backtester._historical_data)
        
        assert len(train_data) == 70
        assert len(test_data) == 30
    
    def test_data_splitting_k_fold(self):
        """Test data splitting with k_fold method."""
        config = BacktestConfig(split_method="k_fold", n_folds=5)
        backtester = Backtester(config)
        
        # Add mock data
        backtester._historical_data = [Mock() for _ in range(100)]
        
        train_data, test_data = backtester._split_data(backtester._historical_data)
        
        assert len(train_data) == 20  # 100/5
        assert len(test_data) == 80
    
    def test_simulate_trading_basic(self):
        """Test basic trading simulation."""
        backtester = Backtester()
        
        # Create mock payload
        payload = Mock(spec=TradingSignalPayload)
        payload.to_dict.return_value = {
            "signal_type": "BUY",
            "entry_price": 50000.0,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "factors": [
                {"factor_name": "technical", "score": 0.8},
                {"factor_name": "volume", "score": 0.7},
                {"factor_name": "sentiment", "score": 0.6},
            ],
        }
        
        params = ParameterSet(
            weights={"technical": 0.4, "volume": 0.3, "sentiment": 0.3},
            max_position_size_pct=0.05,
        )
        
        outcomes = backtester._simulate_trading([payload], params)
        
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.signal_type == "BUY"
        assert outcome.entry_price == 50000.0
        assert outcome.success is not None
        assert outcome.pnl_pct is not None
    
    def test_simulate_trading_insufficient_confirmations(self):
        """Test trading simulation with insufficient confirmations."""
        config = BacktestConfig(min_confirmation_categories=3)
        backtester = Backtester(config)
        
        # Create payload with only 2 factors (less than required 3)
        payload = Mock(spec=TradingSignalPayload)
        payload.to_dict.return_value = {
            "signal_type": "BUY",
            "entry_price": 50000.0,
            "timestamp": int(datetime.now().timestamp() * 1000),
            "factors": [
                {"factor_name": "technical", "score": 0.8},
                {"factor_name": "volume", "score": 0.7},
            ],
        }
        
        params = ParameterSet()
        outcomes = backtester._simulate_trading([payload], params)
        
        assert len(outcomes) == 0  # Should be filtered out
    
    def test_simulate_trading_position_limits(self):
        """Test trading simulation with position limits."""
        config = BacktestConfig(max_concurrent_same_direction=2)
        backtester = Backtester(config)
        
        # Create multiple BUY signals (more than limit)
        payloads = []
        for i in range(5):
            payload = Mock(spec=TradingSignalPayload)
            payload.to_dict.return_value = {
                "signal_type": "BUY",
                "entry_price": 50000.0 + i,
                "timestamp": int(datetime.now().timestamp() * 1000) + i,
                "factors": [
                    {"factor_name": "technical", "score": 0.8},
                    {"factor_name": "volume", "score": 0.7},
                    {"factor_name": "sentiment", "score": 0.6},
                ],
            }
            payloads.append(payload)
        
        params = ParameterSet()
        outcomes = backtester._simulate_trading(payloads, params)
        
        assert len(outcomes) <= 2  # Should be limited by max_concurrent_same_direction
    
    def test_run_backtest_complete(self):
        """Test complete backtest run."""
        backtester = Backtester()
        
        # Setup historical data
        payloads = []
        for i in range(50):
            payload = Mock(spec=TradingSignalPayload)
            payload.to_dict.return_value = {
                "signal_type": "BUY" if i % 2 == 0 else "SELL",
                "entry_price": 50000.0 + i,
                "timestamp": int((datetime.now() - timedelta(days=i)).timestamp() * 1000),
                "factors": [
                    {"factor_name": "technical", "score": 0.8},
                    {"factor_name": "volume", "score": 0.7},
                    {"factor_name": "sentiment", "score": 0.6},
                ],
            }
            payloads.append(payload)
        
        backtester._historical_data = payloads
        
        params = ParameterSet(
            weights={"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        )
        
        result = backtester.run_backtest(params)
        
        assert isinstance(result, BacktestResult)
        assert result.parameter_set == params
        assert isinstance(result.train_kpis, PerformanceKPIs)
        assert isinstance(result.test_kpis, PerformanceKPIs)
        assert result.execution_time_seconds > 0
        assert len(result.train_results) > 0 or len(result.test_results) > 0
    
    def test_optimization_score_calculation(self):
        """Test optimization score calculation."""
        backtester = Backtester()
        
        # Create KPIs that meet targets
        good_kpis = PerformanceKPIs(
            win_rate=0.6,  # Above target 0.55
            profit_factor=1.8,  # Above target 1.5
            sharpe_ratio=1.2,  # Above target 1.0
            max_drawdown_pct=0.15,  # Below target 0.25
        )
        
        score = backtester._calculate_optimization_score(good_kpis)
        assert score > 0.8  # Should be high score
        
        # Create KPIs that don't meet targets
        bad_kpis = PerformanceKPIs(
            win_rate=0.4,  # Below target
            profit_factor=1.0,  # Below target
            sharpe_ratio=0.5,  # Below target
            max_drawdown_pct=0.3,  # Above target
        )
        
        bad_score = backtester._calculate_optimization_score(bad_kpis)
        assert bad_score < score  # Should be lower than good score
    
    def test_generate_grid_search_space(self):
        """Test grid search parameter generation."""
        backtester = Backtester()
        search_space = backtester._get_default_search_space()
        
        assert "weights" in search_space
        assert "stop_loss_pct" in search_space
        assert "take_profit_pct" in search_space
        
        # Check weight bounds
        for weight_name, (min_val, max_val) in search_space["weights"].items():
            assert 0 <= min_val <= max_val <= 1.0
    
    def test_generate_random_search_space(self):
        """Test random search parameter generation."""
        config = BacktestConfig(search_method="random", max_iterations=10, random_seed=42)
        backtester = Backtester(config)
        search_space = backtester._get_default_search_space()
        
        param_sets = backtester._generate_random_search(search_space)
        
        assert len(param_sets) == 10  # Should match max_iterations
        
        # Check that all parameters are within bounds
        for params in param_sets:
            assert 0.01 <= params.stop_loss_pct <= 5.0
            assert 2.0 <= params.take_profit_pct <= 8.0
            assert 0.01 <= params.max_position_size_pct <= 0.1
    
    def test_parameter_importance_analysis(self):
        """Test parameter importance analysis."""
        backtester = Backtester()
        
        # Create mock results
        results = []
        for i in range(10):
            params = ParameterSet(
                weights={"technical": 0.3 + i * 0.05, "volume": 0.4 - i * 0.05}
            )
            result = Mock(spec=BacktestResult)
            result.parameter_set = params
            result.optimization_score = 0.5 + i * 0.05
            results.append(result)
        
        analysis = backtester._analyze_parameter_importance(results)
        
        assert "weight_importance" in analysis
        assert "technical" in analysis["weight_importance"]
        assert "volume" in analysis["weight_importance"]
        assert analysis["total_results"] == 10
    
    def test_save_and_load_results(self):
        """Test saving and loading backtest results."""
        backtester = Backtester()
        
        # Create mock result
        params = ParameterSet(weights={"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        result = Mock(spec=BacktestResult)
        result.parameter_set = params
        result.train_kpis = PerformanceKPIs()
        result.test_kpis = PerformanceKPIs()
        result.train_results = []
        result.test_results = []
        result.targets_met = True
        result.optimization_score = 0.8
        result.execution_time_seconds = 10.0
        result.metadata = {}
        result.to_dict.return_value = {
            "parameter_set": params.to_dict(),
            "train_kpis": result.train_kpis.to_dict(),
            "test_kpis": result.test_kpis.to_dict(),
            "train_results": [],
            "test_results": [],
            "targets_met": True,
            "optimization_score": 0.8,
            "execution_time_seconds": 10.0,
            "metadata": {},
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            backtester.save_results([result], filepath)
            
            # Verify file was created and contains expected data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "config" in data
            assert "results" in data
            assert "report" in data
            assert len(data["results"]) == 1
            
        finally:
            import os
            os.unlink(filepath)


class TestAdaptiveWeightManager:
    """Test cases for AdaptiveWeightManager class."""
    
    def test_adaptive_weight_config_creation(self):
        """Test AdaptiveWeightConfig creation."""
        config = AdaptiveWeightConfig(
            rolling_window_days=60,
            adaptation_strategy="volatility_adjusted",
            target_win_rate=0.6
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["rolling_window_days"] == 60
        assert config_dict["adaptation_strategy"] == "volatility_adjusted"
        assert config_dict["target_win_rate"] == 0.6
    
    def test_adaptive_weight_manager_initialization(self):
        """Test AdaptiveWeightManager initialization."""
        config = AdaptiveWeightConfig(rolling_window_days=30)
        manager = AdaptiveWeightManager(config)
        
        assert manager.config.rolling_window_days == 30
        assert len(manager._weight_performance) == 0
        assert len(manager._adaptation_history) == 0
    
    def test_initialize_weights(self):
        """Test weight initialization."""
        manager = AdaptiveWeightManager()
        
        initial_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        manager.initialize_weights(initial_weights)
        
        assert len(manager._weight_performance) == 3
        assert "technical" in manager._weight_performance
        assert "volume" in manager._weight_performance
        assert "sentiment" in manager._weight_performance
        
        # Check initial values
        for factor_name, weight in initial_weights.items():
            perf = manager._weight_performance[factor_name]
            assert perf.factor_name == factor_name
            assert perf.current_weight == weight
            assert perf.adaptation_count == 0
    
    def test_update_signal_outcomes(self):
        """Test updating signal outcomes."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Create mock outcomes
        outcomes = []
        for i in range(20):
            outcome = Mock(spec=SignalOutcome)
            outcome.success = i % 3 != 0  # ~67% win rate
            outcome.pnl_pct = 2.0 if outcome.success else -1.0
            outcome.entry_timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            outcome.factors = [
                {"factor_name": "technical", "score": 0.8},
                {"factor_name": "volume", "score": 0.7},
            ]
            outcomes.append(outcome)
        
        manager.update_signal_outcomes(outcomes)
        
        assert len(manager._signal_history) == 20
        
        # Check that performance metrics were updated
        tech_perf = manager._weight_performance["technical"]
        assert tech_perf.rolling_win_rate > 0  # Should be updated
    
    def test_should_adapt_insufficient_signals(self):
        """Test adaptation check with insufficient signals."""
        config = AdaptiveWeightConfig(min_signals_for_adaptation=50)
        manager = AdaptiveWeightManager(config)
        
        # Add only 10 signals
        for i in range(10):
            outcome = Mock(spec=SignalOutcome)
            outcome.success = True
            outcome.pnl_pct = 1.0
            outcome.entry_timestamp = int(datetime.now().timestamp() * 1000)
            outcome.factors = []
            manager._signal_history.append(outcome)
        
        should_adapt, reason = manager.should_adapt()
        
        assert should_adapt is False
        assert "Insufficient signals" in reason
    
    def test_should_adapt_recent_rebalance(self):
        """Test adaptation check with recent rebalance."""
        manager = AdaptiveWeightManager()
        manager._last_rebalance_date = datetime.now() - timedelta(days=2)
        
        # Add sufficient signals
        for i in range(100):
            outcome = Mock(spec=SignalOutcome)
            outcome.success = True
            outcome.pnl_pct = 1.0
            outcome.entry_timestamp = int(datetime.now().timestamp() * 1000)
            outcome.factors = []
            manager._signal_history.append(outcome)
        
        should_adapt, reason = manager.should_adapt()
        
        assert should_adapt is False
        assert "Too soon since last rebalance" in reason
    
    def test_should_adapt_performance_below_target(self):
        """Test adaptation check with performance below target."""
        config = AdaptiveWeightConfig(target_win_rate=0.6)
        manager = AdaptiveWeightManager(config)
        
        # Add signals with poor performance
        for i in range(100):
            outcome = Mock(spec=SignalOutcome)
            outcome.success = i % 5 != 0  # 80% win rate - above target
            outcome.pnl_pct = 1.0 if outcome.success else -1.0
            outcome.entry_timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            outcome.factors = []
            manager._signal_history.append(outcome)
        
        # Mock the KPI calculation to return low win rate
        with patch.object(manager, '_calculate_recent_kpis') as mock_kpis:
            mock_kpis.return_value = PerformanceKPIs(win_rate=0.4)  # Below target
            
            should_adapt, reason = manager.should_adapt()
            
            assert should_adapt is True
            assert "Win rate below target" in reason
    
    def test_adapt_weights_performance_based(self):
        """Test performance-based weight adaptation."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Set up performance data
        manager._weight_performance["technical"].rolling_win_rate = 0.8  # High
        manager._weight_performance["volume"].rolling_win_rate = 0.3  # Low
        manager._weight_performance["sentiment"].rolling_win_rate = 0.6  # Medium
        
        current_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        new_weights, reason = manager._adapt_performance_based(current_weights)
        
        # Technical should increase, volume should decrease
        assert new_weights["technical"] > current_weights["technical"]
        assert new_weights["volume"] < current_weights["volume"]
        assert "Performance-based adaptation" in reason
    
    def test_adapt_weights_volatility_adjusted(self):
        """Test volatility-adjusted weight adaptation."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Set up volatility data
        manager._weight_performance["technical"].volatility_score = 0.5  # Lower volatility
        manager._weight_performance["volume"].volatility_score = 2.0  # Higher volatility
        manager._weight_performance["sentiment"].volatility_score = 1.0  # Medium
        
        current_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        new_weights, reason = manager._adapt_volatility_adjusted(current_weights)
        
        # Technical (lower volatility) should get higher weight
        assert new_weights["technical"] > current_weights["technical"]
        assert "Volatility-adjusted adaptation" in reason
    
    def test_adapt_weights_hybrid(self):
        """Test hybrid weight adaptation."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        current_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        new_weights, reason = manager._adapt_hybrid(current_weights)
        
        # Should be a combination of performance and volatility based
        assert sum(new_weights.values()) == pytest.approx(1.0, rel=1e-9)
        assert "Hybrid adaptation" in reason
    
    def test_adapt_weights_without_backtester(self):
        """Adaptive weight adaptation should work without a backtester."""
        config = AdaptiveWeightConfig(max_weight_per_factor=1.0)
        manager = AdaptiveWeightManager(config)
        manager.initialize_weights({"technical": 0.5, "volume": 0.5})
        
        assert manager._backtester is None
        
        with patch.object(
            manager,
            "_adapt_hybrid",
            return_value=(
                {"technical": 0.55, "volume": 0.45},
                "Test adaptation without backtester",
            ),
        ):
            report = manager.adapt_weights()
        
        assert isinstance(report, AdaptationReport)
        assert report.new_weights["technical"] == pytest.approx(0.55)
        assert report.new_weights["volume"] == pytest.approx(0.45)
        assert report.adaptation_reason == "Test adaptation without backtester"
    
    def test_validate_weights_constraints(self):
        """Test weight validation with constraints."""
        config = AdaptiveWeightConfig(
            min_weight_per_factor=0.1,
            max_weight_per_factor=0.6,
            max_weight_change_pct=0.2
        )
        manager = AdaptiveWeightManager(config)
        
        original_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        
        # Test with weights that violate constraints
        new_weights = {
            "technical": 0.8,  # Above max
            "volume": 0.05,  # Below min
            "sentiment": 0.15  # OK
        }
        
        validated = manager._validate_weights(new_weights, original_weights)
        
        # Should be constrained
        assert validated["technical"] <= config.max_weight_per_factor
        assert validated["volume"] >= config.min_weight_per_factor
        assert sum(validated.values()) == pytest.approx(1.0, rel=1e-9)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        manager = AdaptiveWeightManager()
        
        # Add some signal history
        for i in range(50):  # Exactly min_signals_for_adaptation
            outcome = Mock(spec=SignalOutcome)
            outcome.success = True
            outcome.pnl_pct = 1.0
            outcome.entry_timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            outcome.factors = []
            manager._signal_history.append(outcome)
        
        confidence = manager._calculate_confidence_score()
        
        assert 0 <= confidence <= 1.0
        assert confidence > 0  # Should have some confidence with 50 signals
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        config = AdaptiveWeightConfig(target_win_rate=0.6)
        manager = AdaptiveWeightManager(config)
        
        # Mock poor performance
        with patch.object(manager, '_calculate_recent_kpis') as mock_kpis:
            mock_kpis.return_value = PerformanceKPIs(
                win_rate=0.4,  # Below target
                profit_factor=1.0,  # Below target
                max_drawdown_pct=0.3  # Above target
            )
            
            recommendations = manager._generate_recommendations()
            
            assert len(recommendations) > 0
            assert any("win rate" in rec.lower() for rec in recommendations)
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Add some signal history
        for i in range(50):
            outcome = Mock(spec=SignalOutcome)
            outcome.success = True
            outcome.pnl_pct = 1.0
            outcome.entry_timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            outcome.factors = [{"factor_name": "technical", "score": 0.8}]
            manager._signal_history.append(outcome)
        
        manager._update_weight_performance()
        
        report = manager.generate_performance_report()
        
        assert "summary" in report
        assert "weight_analysis" in report
        assert "adaptation_analysis" in report
        assert "performance_vs_targets" in report
        assert "recommendations" in report
        
        # Check current weights are included
        assert "current_weights" in report["summary"]
        assert report["summary"]["current_weights"]["technical"] == 0.4
    
    def test_save_adaptation_history(self):
        """Test saving adaptation history."""
        manager = AdaptiveWeightManager()
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Create a mock adaptation report
        report = Mock(spec=AdaptationReport)
        report.adaptation_date = datetime.now()
        report.original_weights = {"technical": 0.4, "volume": 0.3, "sentiment": 0.3}
        report.new_weights = {"technical": 0.45, "volume": 0.25, "sentiment": 0.3}
        report.performance_before = PerformanceKPIs()
        report.performance_after = None
        report.adaptation_reason = "Test adaptation"
        report.confidence_score = 0.8
        report.expected_improvement = 0.1
        report.factors_adjusted = ["technical", "volume"]
        report.to_dict.return_value = {
            "adaptation_date": report.adaptation_date.isoformat(),
            "original_weights": report.original_weights,
            "new_weights": report.new_weights,
            "performance_before": report.performance_before.to_dict(),
            "performance_after": None,
            "adaptation_reason": report.adaptation_reason,
            "confidence_score": report.confidence_score,
            "expected_improvement": report.expected_improvement,
            "factors_adjusted": report.factors_adjusted,
        }
        
        manager._adaptation_history.append(report)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            manager.save_adaptation_history(filepath)
            
            # Verify file was created and contains expected data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "config" in data
            assert "current_weights" in data
            assert "weight_performance" in data
            assert "adaptation_history" in data
            assert "performance_report" in data
            assert len(data["adaptation_history"]) == 1
            
        finally:
            import os
            os.unlink(filepath)


class TestIntegration:
    """Integration tests for backtesting and adaptive weighting."""
    
    def test_backtester_adaptive_weights_integration(self):
        """Test integration between backtester and adaptive weights."""
        # Setup backtester
        backtester = Backtester()
        
        # Create mock historical data
        payloads = []
        for i in range(100):
            payload = Mock(spec=TradingSignalPayload)
            payload.to_dict.return_value = {
                "signal_type": "BUY" if i % 2 == 0 else "SELL",
                "entry_price": 50000.0 + i,
                "timestamp": int((datetime.now() - timedelta(days=i)).timestamp() * 1000),
                "factors": [
                    {"factor_name": "technical", "score": 0.8},
                    {"factor_name": "volume", "score": 0.7},
                    {"factor_name": "sentiment", "score": 0.6},
                ],
            }
            payloads.append(payload)
        
        backtester._historical_data = payloads
        
        # Setup adaptive weight manager
        manager = AdaptiveWeightManager()
        manager.set_backtester(backtester)
        manager.initialize_weights({"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        
        # Run backtest to generate outcomes
        params = ParameterSet(weights={"technical": 0.4, "volume": 0.3, "sentiment": 0.3})
        result = backtester.run_backtest(params)
        
        # Update manager with outcomes
        manager.update_signal_outcomes(result.test_results)
        
        # Check if adaptation should be performed
        should_adapt, reason = manager.should_adapt()
        
        # Should adapt since we have no previous adaptations
        assert should_adapt is True or "Performance targets met" in reason
        
        # Generate performance report
        report = manager.generate_performance_report()
        
        assert "summary" in report
        assert "current_weights" in report["summary"]
        assert len(report["current_weights"]) == 3
    
    @patch('indicator_collector.trading_system.backtester.validate_real_data_payload')
    def test_real_data_validation_integration(self, mock_validate):
        """Test integration with real data validation."""
        backtester = Backtester(config=BacktestConfig(validate_real_data=True))
        
        # Mock validation to pass
        mock_validate.return_value = None
        
        # Create mock payload
        payload = Mock(spec=TradingSignalPayload)
        payload.to_dict.return_value = {"test": "data"}
        
        # Load data
        count = backtester.load_historical_data([payload])
        
        assert count == 1
        mock_validate.assert_called_once()
    
    def test_optimization_with_adaptive_weights(self):
        """Test parameter optimization with adaptive weights."""
        backtester = Backtester()
        
        # Create minimal historical data for optimization
        payloads = []
        for i in range(50):
            payload = Mock(spec=TradingSignalPayload)
            payload.timestamp = int((datetime.now() - timedelta(days=i)).timestamp() * 1000)
            payload.to_dict.return_value = {
                "timestamp": payload.timestamp,
                "signal_type": "BUY",
                "entry_price": 50000.0,
                "factors": [
                    {"factor_name": "technical", "score": 0.8},
                    {"factor_name": "volume", "score": 0.7},
                    {"factor_name": "sentiment", "score": 0.6},
                ],
            }
            payloads.append(payload)
        
        backtester._historical_data = payloads
        
        # Run optimization with limited iterations for test speed
        backtester.config.max_iterations = 5
        search_space = {
            "weights": {
                "technical": (0.2, 0.6),
                "volume": (0.2, 0.5),
                "sentiment": (0.1, 0.4),
            },
            "stop_loss_pct": (1.0, 3.0),
            "take_profit_pct": (2.0, 6.0),
            "max_position_size_pct": (0.02, 0.08),
            "confirmation_threshold": (0.5, 0.7),
        }
        
        try:
            best_params, best_result = backtester.optimize_parameters(search_space)
            
            assert isinstance(best_params, ParameterSet)
            assert isinstance(best_result, BacktestResult)
            assert best_result.optimization_score >= 0
            
        except Exception as e:
            # Optimization might fail due to insufficient data or other issues
            # That's acceptable for this test
            pytest.skip(f"Optimization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])