"""Tests for the position manager module."""

import unittest
from unittest.mock import Mock, patch

from indicator_collector.trading_system.position_manager import (
    PositionManagerConfig,
    DiversificationGuard,
    PositionSizingResult,
    PositionManagerResult,
    calculate_risk_based_position_size,
    assess_market_conditions,
    estimate_holding_horizon,
    create_position_plan,
    create_diversification_guard,
    validate_tp_sl_spacing,
)
from indicator_collector.trading_system.interfaces import (
    AnalyzerContext,
    PositionPlan,
)


class TestPositionManagerConfig(unittest.TestCase):
    """Test the PositionManagerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionManagerConfig()
        self.assertEqual(config.max_position_size_usd, 1000.0)
        self.assertEqual(config.max_risk_per_trade_pct, 0.02)
        self.assertEqual(config.default_leverage, 10.0)
        self.assertEqual(config.max_concurrent_same_direction, 3)
        self.assertEqual(config.tp1_multiplier, 1.5)
        self.assertEqual(config.tp2_multiplier, 3.0)
        self.assertEqual(config.tp3_multiplier, 5.0)
        self.assertEqual(config.sl_multiplier, 1.0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = PositionManagerConfig()
        config.validate()  # Should not raise
        
        # Invalid max_position_size_usd
        with self.assertRaises(ValueError):
            PositionManagerConfig(max_position_size_usd=0).validate()
        
        # Invalid max_risk_per_trade_pct
        with self.assertRaises(ValueError):
            PositionManagerConfig(max_risk_per_trade_pct=0).validate()
        
        with self.assertRaises(ValueError):
            PositionManagerConfig(max_risk_per_trade_pct=0.2).validate()  # Too high
        
        # Invalid leverage
        with self.assertRaises(ValueError):
            PositionManagerConfig(default_leverage=0).validate()
        
        # Invalid TP multipliers (not increasing)
        with self.assertRaises(ValueError):
            PositionManagerConfig(tp1_multiplier=2.0, tp2_multiplier=1.5).validate()
        
        # Invalid SL multiplier
        with self.assertRaises(ValueError):
            PositionManagerConfig(sl_multiplier=0).validate()
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PositionManagerConfig(
            max_position_size_usd=500.0,
            max_risk_per_trade_pct=0.01,
            default_leverage=5.0,
            max_concurrent_same_direction=2,
        )
        self.assertEqual(config.max_position_size_usd, 500.0)
        self.assertEqual(config.max_risk_per_trade_pct, 0.01)
        self.assertEqual(config.default_leverage, 5.0)
        self.assertEqual(config.max_concurrent_same_direction, 2)


class TestDiversificationGuard(unittest.TestCase):
    """Test the DiversificationGuard class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PositionManagerConfig()
        self.guard = DiversificationGuard()
    
    def test_empty_guard(self):
        """Test empty guard state."""
        self.assertEqual(len(self.guard.long_positions), 0)
        self.assertEqual(len(self.guard.short_positions), 0)
        self.assertEqual(self.guard.total_positions, 0)
    
    def test_can_add_long_position(self):
        """Test adding long positions within limits."""
        can_add, reason = self.guard.can_add_position("long", "BTC", self.config)
        self.assertTrue(can_add)
        self.assertIsNone(reason)
        
        # Add position
        self.guard.add_position("long", "BTC")
        self.assertEqual(len(self.guard.long_positions), 1)
        self.assertIn("BTC", self.guard.long_positions)
        self.assertEqual(self.guard.total_positions, 1)
    
    def test_can_add_short_position(self):
        """Test adding short positions within limits."""
        can_add, reason = self.guard.can_add_position("short", "ETH", self.config)
        self.assertTrue(can_add)
        self.assertIsNone(reason)
        
        # Add position
        self.guard.add_position("short", "ETH")
        self.assertEqual(len(self.guard.short_positions), 1)
        self.assertIn("ETH", self.guard.short_positions)
        self.assertEqual(self.guard.total_positions, 1)
    
    def test_max_concurrent_same_direction(self):
        """Test same-direction limit enforcement."""
        # Add max long positions
        for i in range(self.config.max_concurrent_same_direction):
            symbol = f"SYMBOL{i}"
            can_add, reason = self.guard.can_add_position("long", symbol, self.config)
            self.assertTrue(can_add)
            self.guard.add_position("long", symbol)
        
        # Try to add one more - should be blocked
        can_add, reason = self.guard.can_add_position("long", "EXTRA", self.config)
        self.assertFalse(can_add)
        self.assertIsNotNone(reason)
        self.assertIn("Max long positions", reason)
    
    def test_duplicate_symbol_prevention(self):
        """Test prevention of duplicate symbols in same direction."""
        # Add BTC long
        self.guard.add_position("long", "BTC")
        
        # Try to add BTC long again - should be blocked
        can_add, reason = self.guard.can_add_position("long", "BTC", self.config)
        self.assertFalse(can_add)
        self.assertIn("Already have long position", reason)
        
        # But BTC short should be allowed
        can_add, reason = self.guard.can_add_position("short", "BTC", self.config)
        self.assertTrue(can_add)
    
    def test_max_total_positions(self):
        """Test total position limit enforcement."""
        config = PositionManagerConfig(max_total_positions=2)
        guard = DiversificationGuard()
        
        # Add max total positions
        guard.add_position("long", "BTC")
        guard.add_position("short", "ETH")
        
        # Try to add one more - should be blocked
        can_add, reason = guard.can_add_position("long", "SOL", config)
        self.assertFalse(can_add)
        self.assertIn("Max total positions", reason)
    
    def test_remove_position(self):
        """Test position removal."""
        # Add positions
        self.guard.add_position("long", "BTC")
        self.guard.add_position("short", "ETH")
        self.assertEqual(self.guard.total_positions, 2)
        
        # Remove long position
        self.guard.remove_position("long", "BTC")
        self.assertEqual(len(self.guard.long_positions), 0)
        self.assertEqual(self.guard.total_positions, 1)
        
        # Remove short position
        self.guard.remove_position("short", "ETH")
        self.assertEqual(len(self.guard.short_positions), 0)
        self.assertEqual(self.guard.total_positions, 0)
    
    def test_remove_nonexistent_position(self):
        """Test removing non-existent positions."""
        # Should not raise error
        self.guard.remove_position("long", "NONEXISTENT")
        self.guard.remove_position("short", "NONEXISTENT")
        self.assertEqual(self.guard.total_positions, 0)


class TestCalculateRiskBasedPositionSize(unittest.TestCase):
    """Test risk-based position sizing calculations."""
    
    def test_basic_position_sizing(self):
        """Test basic position sizing calculation."""
        result = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            account_balance=10000.0,
            risk_per_trade_pct=0.02,
            leverage=10.0,
            commission_rate=0.0006,
        )
        
        self.assertIsInstance(result, PositionSizingResult)
        self.assertEqual(result.risk_amount_usd, 200.0)  # 2% of 10000
        self.assertEqual(result.leverage, 10.0)
        self.assertGreater(result.position_size_usd, 0)
        self.assertGreater(result.quantity, 0)
        self.assertGreater(result.commission_cost, 0)
        
        # Check sizing factors
        self.assertIn("risk_per_unit", result.sizing_factors)
        self.assertIn("risk_amount_usd", result.sizing_factors)
        self.assertIn("notional_value", result.sizing_factors)
        self.assertIn("commission_cost_pct", result.sizing_factors)
    
    def test_position_sizing_with_small_risk(self):
        """Test position sizing with very small price distance."""
        result = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=99.9,  # Very small distance
            account_balance=10000.0,
            risk_per_trade_pct=0.02,
        )
        
        # Should handle small risk gracefully
        self.assertGreater(result.position_size_usd, 0)
        self.assertGreater(result.sizing_factors["risk_per_unit"], 0)
    
    def test_position_sizing_zero_risk(self):
        """Test position sizing with zero price distance."""
        result = calculate_risk_based_position_size(
            entry_price=100.0,
            stop_loss=100.0,  # No distance
            account_balance=10000.0,
            risk_per_trade_pct=0.02,
        )
        
        # Should fall back to default 10% position
        self.assertEqual(result.position_size_usd, 1000.0)  # 10% of 10000
    
    def test_position_sizing_metrics(self):
        """Test position sizing metric calculations."""
        result = calculate_risk_based_position_size(
            entry_price=50.0,
            stop_loss=45.0,
            account_balance=5000.0,
            risk_per_trade_pct=0.01,
            leverage=5.0,
        )
        
        expected_risk = 50.0  # 1% of 5000
        self.assertEqual(result.risk_amount_usd, expected_risk)
        
        # Check notional value
        expected_notional = result.position_size_usd * 5.0
        self.assertAlmostEqual(result.sizing_factors["notional_value"], expected_notional, places=2)
        
        # Check quantity
        expected_quantity = expected_notional / 50.0
        self.assertAlmostEqual(result.quantity, expected_quantity, places=2)


class TestAssessMarketConditions(unittest.TestCase):
    """Test market condition assessment."""
    
    def create_context(self, atr=1.0, current_price=100.0, volume_confidence=0.5, risk_score=0.3):
        """Create a mock AnalyzerContext."""
        context = Mock(spec=AnalyzerContext)
        context.indicators = {"atr": atr}
        context.current_price = current_price
        context.volume_analysis = {"volume_confidence": volume_confidence}
        context.advanced_metrics = {
            "market_context": {"risk_score": risk_score}
        }
        context.metadata = {}
        context.extras = {}
        return context
    
    def test_normal_conditions(self):
        """Test assessment under normal market conditions."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        can_trade, fatal_reasons, warnings = assess_market_conditions(context, config)
        
        self.assertTrue(can_trade)
        self.assertEqual(fatal_reasons, [])
        self.assertEqual(warnings, [])
    
    def test_high_volatility_cancellation(self):
        """Test cancellation due to high volatility."""
        config = PositionManagerConfig()
        # High volatility: 10% ATR/price ratio
        context = self.create_context(atr=10.0, current_price=100.0)
        
        can_trade, fatal_reasons, warnings = assess_market_conditions(context, config)
        
        self.assertTrue(can_trade)
        self.assertEqual(fatal_reasons, [])
        self.assertTrue(any("High volatility" in warning for warning in warnings))
    
    def test_low_liquidity_cancellation(self):
        """Test cancellation due to low liquidity."""
        config = PositionManagerConfig()
        context = self.create_context(volume_confidence=0.1)  # Below threshold
        
        can_trade, fatal_reasons, warnings = assess_market_conditions(context, config)
        
        self.assertTrue(can_trade)
        self.assertEqual(fatal_reasons, [])
        self.assertTrue(any("Low liquidity" in warning for warning in warnings))
    
    def test_high_risk_score_cancellation(self):
        """Test cancellation due to high risk score."""
        config = PositionManagerConfig()
        context = self.create_context(risk_score=0.9)  # Above threshold
        
        can_trade, fatal_reasons, warnings = assess_market_conditions(context, config)
        
        self.assertTrue(can_trade)
        self.assertEqual(fatal_reasons, [])
        self.assertTrue(any("High risk score" in warning for warning in warnings))
    
    def test_multiple_cancellation_reasons(self):
        """Test multiple cancellation reasons."""
        config = PositionManagerConfig()
        context = self.create_context(
            atr=15.0,  # High volatility
            current_price=100.0,
            volume_confidence=0.1,  # Low liquidity
            risk_score=0.9,  # High risk
        )
        
        can_trade, fatal_reasons, warnings = assess_market_conditions(context, config)
        
        self.assertTrue(can_trade)
        self.assertEqual(fatal_reasons, [])
        self.assertGreaterEqual(len(warnings), 3)


class TestEstimateHoldingHorizon(unittest.TestCase):
    """Test holding horizon estimation."""
    
    def create_context(self, trend_strength=0.5, atr=2.0, current_price=100.0, structure_state="neutral"):
        """Create a mock AnalyzerContext."""
        context = Mock(spec=AnalyzerContext)
        context.indicators = {"trend_strength": trend_strength, "atr": atr}
        context.current_price = current_price
        context.market_structure = {"structure_state": structure_state}
        return context
    
    def test_baseline_holding_horizon(self):
        """Test baseline holding horizon calculation."""
        config = PositionManagerConfig(target_holding_bars=20)
        context = self.create_context()
        
        horizon = estimate_holding_horizon(context, config, "long")
        
        self.assertEqual(horizon, 20)  # Should be baseline with neutral conditions
    
    def test_strong_trend_adjustment(self):
        """Test holding horizon adjustment for strong trends."""
        config = PositionManagerConfig(target_holding_bars=20)
        context = self.create_context(trend_strength=0.8)  # Strong trend
        
        horizon = estimate_holding_horizon(context, config, "long")
        
        # Should be longer than baseline for strong trend
        self.assertGreater(horizon, 20)
    
    def test_weak_trend_adjustment(self):
        """Test holding horizon adjustment for weak trends."""
        config = PositionManagerConfig(target_holding_bars=20)
        context = self.create_context(trend_strength=0.2)  # Weak trend
        
        horizon = estimate_holding_horizon(context, config, "long")
        
        # Should be shorter than baseline for weak trend
        self.assertLess(horizon, 20)
    
    def test_volatility_adjustment(self):
        """Test holding horizon adjustment for volatility."""
        config = PositionManagerConfig(target_holding_bars=20)
        # High volatility: 5% ATR/price ratio
        context = self.create_context(atr=5.0, current_price=100.0)
        
        horizon = estimate_holding_horizon(context, config, "long")
        
        # Should be shorter for high volatility
        self.assertLess(horizon, 20)
    
    def test_structure_adjustment(self):
        """Test holding horizon adjustment for market structure."""
        config = PositionManagerConfig(target_holding_bars=20)
        
        # Trending market
        context_trending = self.create_context(structure_state="trending")
        horizon_trending = estimate_holding_horizon(context_trending, config, "long")
        self.assertGreater(horizon_trending, 20)
        
        # Ranging market
        context_ranging = self.create_context(structure_state="ranging")
        horizon_ranging = estimate_holding_horizon(context_ranging, config, "long")
        self.assertLess(horizon_ranging, 20)
    
    def test_horizon_clamping(self):
        """Test holding horizon clamping to valid range."""
        config = PositionManagerConfig(
            min_holding_bars=5,
            max_holding_bars=50,
            target_holding_bars=20,
        )
        
        # Test extreme values that should be clamped
        context = self.create_context(trend_strength=1.0, atr=20.0, current_price=100.0)  # Extreme adjustments
        
        horizon = estimate_holding_horizon(context, config, "long")
        
        self.assertGreaterEqual(horizon, config.min_holding_bars)
        self.assertLessEqual(horizon, config.max_holding_bars)


class TestValidateTpSlSpacing(unittest.TestCase):
    """Test TP/SL spacing validation."""
    
    def test_valid_spacing(self):
        """Test valid TP/SL spacing."""
        tp_levels = [105.0, 110.0, 115.0]
        stop_loss = 95.0
        entry_price = 100.0
        
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_tp_levels_too_close(self):
        """Test TP levels that are too close together."""
        tp_levels = [100.3, 100.4, 100.5]  # Very close
        stop_loss = 95.0
        entry_price = 100.0
        
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price, min_spacing_pct=0.01)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("TP1 and TP2 too close" in error for error in errors))
    
    def test_tp_to_sl_too_close(self):
        """Test TP level too close to SL."""
        tp_levels = [100.3, 105.0, 110.0]
        stop_loss = 100.2  # Very close to TP1
        entry_price = 100.0
        
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price, min_spacing_pct=0.01)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("TP1 and SL too close" in error for error in errors))
    
    def test_entry_to_sl_too_close(self):
        """Test entry price too close to SL."""
        tp_levels = [105.0, 110.0, 115.0]
        stop_loss = 100.1  # Very close to entry
        entry_price = 100.0
        
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price, min_spacing_pct=0.01)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Entry and SL too close" in error for error in errors))
    
    def test_custom_min_spacing(self):
        """Test custom minimum spacing."""
        tp_levels = [100.2, 100.4, 100.6]
        stop_loss = 99.8
        entry_price = 100.0
        
        # Should fail with 1% spacing
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price, min_spacing_pct=0.01)
        self.assertFalse(is_valid)
        
        # Should pass with 0.1% spacing
        is_valid, errors = validate_tp_sl_spacing(tp_levels, stop_loss, entry_price, min_spacing_pct=0.001)
        self.assertTrue(is_valid)


class TestCreatePositionPlan(unittest.TestCase):
    """Test comprehensive position plan creation."""
    
    def create_context(self, symbol="BTC", current_price=100.0, atr=2.0):
        """Create a mock AnalyzerContext."""
        context = Mock(spec=AnalyzerContext)
        context.symbol = symbol
        context.current_price = current_price
        context.indicators = {"atr": atr}
        context.volume_analysis = {"volume_confidence": 0.5}
        context.market_structure = {"structure_state": "neutral"}
        context.advanced_metrics = {"market_context": {"risk_score": 0.3}}
        context.metadata = {}
        context.extras = {}
        context.ohlcv = {
            "open": current_price,
            "close": current_price,
            "high": current_price * 1.01,
            "low": current_price * 0.99,
        }
        return context
    
    def test_successful_position_plan_creation(self):
        """Test successful position plan creation."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertIsInstance(result, PositionManagerResult)
        self.assertTrue(result.can_trade)
        self.assertIsNotNone(result.position_plan)
        self.assertIsNotNone(result.sizing_result)
        self.assertIsNotNone(result.holding_horizon_bars)
        self.assertEqual(len(result.cancellation_reasons), 0)
        
        # Check position plan details
        plan = result.position_plan
        self.assertEqual(plan.entry_price, 100.0)
        self.assertEqual(plan.direction, "long")
        self.assertEqual(plan.leverage, config.default_leverage)
        self.assertEqual(len(plan.take_profit_levels), 3)
        self.assertIsNotNone(plan.stop_loss)
        self.assertGreater(plan.position_size_usd, 0)
    
    def test_position_plan_with_diversification_guard(self):
        """Test position plan creation with diversification guard."""
        config = PositionManagerConfig()
        context = self.create_context()
        guard = create_diversification_guard()
        
        # Add one existing position
        guard.add_position("long", "ETH")
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
            diversification_guard=guard,
        )
        
        self.assertTrue(result.can_trade)
        self.assertEqual(len(guard.long_positions), 2)  # ETH + BTC
        self.assertIn("BTC", guard.long_positions)
    
    def test_diversification_limit_reached(self):
        """Test position plan blocked by diversification limits."""
        config = PositionManagerConfig(max_concurrent_same_direction=2)
        context = self.create_context()
        guard = create_diversification_guard()
        
        # Fill up long positions
        guard.add_position("long", "ETH")
        guard.add_position("long", "SOL")
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
            diversification_guard=guard,
        )
        
        self.assertFalse(result.can_trade)
        self.assertIsNone(result.position_plan)
        self.assertGreater(len(result.cancellation_reasons), 0)
        self.assertTrue(any("Max long positions" in reason for reason in result.cancellation_reasons))
    
    def test_market_conditions_cancellation(self):
        """Test position plan blocked by market conditions."""
        config = PositionManagerConfig()
        # Create context with high volatility
        context = self.create_context(atr=10.0, current_price=100.0)  # 10% volatility
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertTrue(result.can_trade)
        self.assertIsNotNone(result.position_plan)
        self.assertEqual(result.cancellation_reasons, [])
        self.assertTrue(any("High volatility" in warning for warning in result.warnings))
    
    def test_no_atr_cancellation(self):
        """Test position plan blocked by missing ATR."""
        config = PositionManagerConfig()
        context = self.create_context(atr=0)  # No ATR
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertTrue(result.can_trade)
        self.assertIsNotNone(result.position_plan)
        self.assertEqual(result.cancellation_reasons, [])
        self.assertTrue(any("ATR" in warning for warning in result.warnings))
    
    def test_position_size_limiting(self):
        """Test position size limiting by max position size."""
        config = PositionManagerConfig(max_position_size_usd=100.0)
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=100000.0,  # Large account
        )
        
        self.assertTrue(result.can_trade)
        self.assertIsNotNone(result.position_plan)
        
        # Position size should be limited
        self.assertLessEqual(result.position_plan.position_size_usd, config.max_position_size_usd)
        
        # Check sizing result indicates limitation
        self.assertTrue(result.sizing_result.sizing_factors.get("size_limited", False))
        self.assertIn("original_size", result.sizing_result.sizing_factors)
        self.assertTrue(any("limited" in warning.lower() for warning in result.warnings))
    
    def test_short_position_plan(self):
        """Test short position plan creation."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="short",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertTrue(result.can_trade)
        plan = result.position_plan
        self.assertEqual(plan.direction, "short")
        
        # For short: TP should be below entry, SL above entry
        self.assertLess(plan.take_profit_levels[0], plan.entry_price)
        self.assertGreater(plan.stop_loss, plan.entry_price)
    
    def test_long_position_plan(self):
        """Test long position plan creation."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertTrue(result.can_trade)
        plan = result.position_plan
        self.assertEqual(plan.direction, "long")
        
        # For long: TP should be above entry, SL below entry
        self.assertGreater(plan.take_profit_levels[0], plan.entry_price)
        self.assertLess(plan.stop_loss, plan.entry_price)
    
    def test_tp_sl_spacing_validation(self):
        """Test TP/SL spacing validation in position plan."""
        config = PositionManagerConfig(
            tp1_multiplier=0.1,  # Very small multiplier
            sl_multiplier=0.1,
        )
        context = self.create_context(atr=0.5)  # Small ATR
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        # Should still succeed but with very tight levels
        self.assertTrue(result.can_trade)
        plan = result.position_plan
        
        # Verify TP levels are in correct order
        self.assertLess(plan.take_profit_levels[0], plan.take_profit_levels[1])
        self.assertLess(plan.take_profit_levels[1], plan.take_profit_levels[2])
    
    def test_holding_horizon_inclusion(self):
        """Test holding horizon inclusion in position plan."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        self.assertIsNotNone(result.holding_horizon_bars)
        self.assertGreater(result.holding_horizon_bars, 0)
        
        # Check that holding horizon is included in plan metadata
        plan = result.position_plan
        self.assertIn("holding_horizon_bars", plan.metadata)
        self.assertEqual(plan.metadata["holding_horizon_bars"], result.holding_horizon_bars)
    
    def test_metadata_completeness(self):
        """Test completeness of metadata in position plan."""
        config = PositionManagerConfig()
        context = self.create_context()
        
        result = create_position_plan(
            context=context,
            signal_direction="long",
            config=config,
            account_balance=10000.0,
        )
        
        plan = result.position_plan
        metadata = plan.metadata
        
        # Check required metadata fields
        self.assertIn("atr", metadata)
        self.assertIn("tp_sl_multipliers", metadata)
        self.assertIn("holding_horizon_bars", metadata)
        self.assertIn("sizing_factors", metadata)
        
        # Check TP/SL multipliers
        tp_sl_multipliers = metadata["tp_sl_multipliers"]
        self.assertEqual(tp_sl_multipliers["tp1"], config.tp1_multiplier)
        self.assertEqual(tp_sl_multipliers["tp2"], config.tp2_multiplier)
        self.assertEqual(tp_sl_multipliers["tp3"], config.tp3_multiplier)
        self.assertEqual(tp_sl_multipliers["sl"], config.sl_multiplier)


class TestCreateDiversificationGuard(unittest.TestCase):
    """Test diversification guard creation."""
    
    def test_create_guard(self):
        """Test creating a new diversification guard."""
        guard = create_diversification_guard()
        
        self.assertIsInstance(guard, DiversificationGuard)
        self.assertEqual(len(guard.long_positions), 0)
        self.assertEqual(len(guard.short_positions), 0)
        self.assertEqual(guard.total_positions, 0)


if __name__ == "__main__":
    unittest.main()